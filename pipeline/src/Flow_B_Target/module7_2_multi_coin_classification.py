"""Modulo 7.2 (Flow B): Classificazione Monete Multiple (Soft-CSP Multi-Modale: Geometria, Colore e ZNCC).

Questo modulo è dedicato alla classificazione delle immagini che contengono MONETE MULTIPLE (> 1 moneta),
integrando in un unico framework probabilistico:
1. Invarianza di Scala Relativa e Coerenza Fisica (Hard/Dominant Constraint):
   - Tutte le monete della stessa immagine condividono la stessa distanza fotocamera-piano.
   - Il fattore di scala S = raggio_pixel / raggio_nominale_mm deve essere costante (minimizzazione di CV(S)).
   - Vincolo di Monotonicità: se r_i > r_j + tolleranza, allora D_i deve essere >= D_j.
2. Rilassamento Cromatico (Soft Pruning):
   - Per monete con classificazione cromatica sicura ('confident' == True), lo spazio dei candidati
     è ristretto al proprio gruppo ('1 euro', '2 euro', 'gold', 'copper').
   - Per monete con confidenza cromatica bassa ('confident' == False, es. monete ossidate/sporche),
     lo spazio dei candidati viene allargato al gruppo cromatico vicino (es. monometalliche sia gold che copper).
   - Se una tupla seleziona una classe appartenente a un gruppo cromatico diverso da quello predetto,
     subisce una penalità morbida pesata per la confidenza residua.
3. Termine di Regolarizzazione ZNCC Dinamico (Bordi e Bassorilievi):
   - Per ciascuna moneta e ciascun candidato, confronta la matrice polare target con la matrice
     di riferimento del Flusso A tramite Circular ZNCC.
   - Il peso del bonus ZNCC viene modulato dinamicamente:
     * Aumenta se la moneta ha bassa confidenza di colore (aiuta a recuperare errori cromatici).
     * Aumenta se la moneta ha raggio simile ad altre monete nella scena (risolve l'ambiguità di scala/tie-breaker).
   - Il peso base ZNCC è calibrato per essere ridotto rispetto alle penalità geometriche, garantendo
     che la consistenza fisica e la monotonicità non vengano mai violate.
"""

import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import funzioni ZNCC e caricamento reference dal Modulo 7.1
try:
    from module7_1_single_coin_classification import (
        load_reference_polar_matrices,
        compute_circular_zncc,
        get_default_reference_polar_dir
    )
except ImportError:
    from pipeline.src.Flow_B_Target.module7_1_single_coin_classification import (
        load_reference_polar_matrices,
        compute_circular_zncc,
        get_default_reference_polar_dir
    )


CLASS_TO_KIND: Dict[str, str] = {
    "1cent": "copper",
    "2cent": "copper",
    "5cent": "copper",
    "10cent": "gold",
    "20cent": "gold",
    "50cent": "gold",
    "1euro": "1 euro",
    "2euro": "2 euro"
}


def get_default_conf_dir() -> Path:
    """Restituisce il percorso della cartella pipeline/conf/Flow_B_Target."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "conf" / "Flow_B_Target"


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module7_2_multi_coin_classification."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module7_2_multi_coin_classification"


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json unificato in artifacts/Flow_B_Target."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "annotated_images.json"


def load_or_create_multi_coin_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di classificazione monete multiple da file JSON o crea i default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "multi_coin_classification_params.json"
    else:
        json_path = Path(json_path)
        if not json_path.exists():
            proj_root = Path(__file__).resolve().parents[3]
            if (proj_root / json_path).exists():
                json_path = proj_root / json_path

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"Parametri Modulo 7.2 caricati da: {json_path.name}")
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path}: {e}. Uso parametri default.")

    params = {
        "diameters_mm": {
            "1cent": 16.25,
            "2cent": 18.75,
            "10cent": 19.75,
            "5cent": 21.25,
            "20cent": 22.25,
            "1euro": 23.25,
            "50cent": 24.25,
            "2euro": 25.75
        },
        "kind_to_candidates": {
            "1 euro": ["1euro"],
            "2 euro": ["2euro"],
            "gold": ["10cent", "20cent", "50cent"],
            "copper": ["1cent", "2cent", "5cent"]
        },
        "monotonicity_tolerance_px": 3.0,
        "weight_ratio_error": 10.0,
        "weight_scale_cv": 20.0,
        "weight_monotonicity_violation": 50.0,
        "use_soft_csp": True,
        "weight_color_penalty": 3.0,
        "weight_base_zncc": 1.5,
        "alpha_color_uncertainty": 1.0,
        "beta_dim_uncertainty": 1.0,
        "radius_similarity_px": 4.0
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Creato file di configurazione: {json_path.name}")
    return params


def load_coin_polar_matrix(coin: Dict[str, Any], img_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Carica la matrice polare (360x50) della moneta target."""
    proj_root = Path(__file__).resolve().parents[3]
    polar_path_str = coin.get("polar_unwarp_path", "")
    if polar_path_str:
        p = proj_root / polar_path_str
        if p.exists():
            try:
                return np.load(str(p)).astype(np.float32)
            except Exception:
                pass

    if img_name:
        coin_id = coin.get("coin_id", 0)
        p = Path(__file__).resolve().parents[2] / "artifacts" / "Flow_B_Target" / "module6_polar_unwarp" / f"{img_name}_{coin_id}_polar.npy"
        if p.exists():
            try:
                return np.load(str(p)).astype(np.float32)
            except Exception:
                pass

    return None


def solve_multi_coin_assignment(
    coins: List[Dict[str, Any]],
    params: Dict[str, Any],
    reference_matrices: Optional[Dict[str, np.ndarray]] = None,
    img_name: Optional[str] = None
) -> Tuple[List[str], float, Dict[str, Any]]:
    """
    Risolve l'assegnazione ottima delle classi per le monete di una singola immagine
    tramite Soft-CSP Multi-Modale (Geometria + Colore Morbido + Polar ZNCC).
    
    1. Soft Pruning dei Candidati:
       - Se la moneta ha confidenza cromatica sicura ('confident' == True),
         usa il set ristretto del gruppo colore ('kind').
       - Se la moneta ha colore incerto ('confident' == False), allarga lo spazio
         dei candidati ai gruppi vicini (es. monometalliche copper + gold).
    2. Valutazione Combinatoria Multi-Modale:
       - Termine Geometrico: dispersione della scala CV(S), errore rapporti e violazioni di monotonicità.
       - Termine Cromatico: penalità per deviazioni dal gruppo colore predetto pesata sulla confidenza.
       - Termine ZNCC: bonus pesato dinamicamente, più influente in presenza di incertezza di colore
         o di raggi dimensionalmente vicini (tie-breaker).
    3. Restituisce la tupla vincente con lo score massimo e i dettagli completi.
    """
    diameters = params.get("diameters_mm", {})
    kind_to_candidates = params.get("kind_to_candidates", {})
    mono_tol = float(params.get("monotonicity_tolerance_px", 3.0))
    w_ratio = float(params.get("weight_ratio_error", 10.0))
    w_cv = float(params.get("weight_scale_cv", 20.0))
    w_viol = float(params.get("weight_monotonicity_violation", 50.0))

    # Parametri Soft-CSP e ZNCC
    use_soft_csp = bool(params.get("use_soft_csp", True))
    w_color_pen = float(params.get("weight_color_penalty", 3.0))
    w_base_zncc = float(params.get("weight_base_zncc", 1.5))
    alpha_color = float(params.get("alpha_color_uncertainty", 1.0))
    beta_dim = float(params.get("beta_dim_uncertainty", 1.0))
    rad_sim_px = float(params.get("radius_similarity_px", 4.0))

    n_coins = len(coins)
    radii = [float(c["radius"]) for c in coins]

    # 1. Candidati per ciascuna moneta
    candidates_per_coin = []
    for c in coins:
        knd = c.get("kind", "gold")
        confident = bool(c.get("confident", True))

        if use_soft_csp and not confident:
            # Espansione dei candidati per monete con colore incerto:
            if knd in ["gold", "copper"]:
                # Monometallica incerta: ammette sia rame che oro
                cands = ["1cent", "2cent", "5cent", "10cent", "20cent", "50cent"]
            elif knd in ["1 euro", "2 euro"]:
                # Bimetallica incerta: ammette entrambi i tagli bimetallici
                cands = ["1euro", "2euro"]
            else:
                cands = kind_to_candidates.get(knd, ["10cent", "20cent", "50cent"])
        else:
            cands = kind_to_candidates.get(knd, ["10cent", "20cent", "50cent"])

        candidates_per_coin.append(cands)

    # Safety check sullo spazio combinatorio: se il numero di tuple supera 50.000,
    # mantiene l'espansione solo sulle 2 monete con confidenza più bassa
    num_tuples_est = int(np.prod([len(c) for c in candidates_per_coin]))
    if num_tuples_est > 50000:
        confs_list = [(i, float(coins[i].get("confidence", 0.70))) for i in range(n_coins)]
        confs_list.sort(key=lambda x: x[1])
        unconstrained_indices = {confs_list[0][0], confs_list[1][0]}
        for i in range(n_coins):
            if i not in unconstrained_indices:
                knd = coins[i].get("kind", "gold")
                candidates_per_coin[i] = kind_to_candidates.get(knd, ["10cent", "20cent", "50cent"])

    all_tuples = list(itertools.product(*candidates_per_coin))

    # 2. Incertezza dimensionale (monete con raggio molto vicino ad altre monete nella scena)
    dim_uncertainty = []
    for i in range(n_coins):
        is_close = any(abs(radii[i] - radii[j]) <= rad_sim_px for j in range(n_coins) if j != i)
        dim_uncertainty.append(1.0 if is_close else 0.0)

    # 3. Precalcolo ZNCC per ciascuna moneta e ciascun candidato
    zncc_scores_per_coin: List[Dict[str, float]] = []
    for i in range(n_coins):
        coin_zncc: Dict[str, float] = {}
        target_polar = load_coin_polar_matrix(coins[i], img_name=img_name) if (use_soft_csp and reference_matrices) else None

        for cand in candidates_per_coin[i]:
            if target_polar is not None and reference_matrices and cand in reference_matrices:
                try:
                    score, _ = compute_circular_zncc(target_polar, reference_matrices[cand])
                    coin_zncc[cand] = float(score)
                except Exception:
                    coin_zncc[cand] = 0.0
            else:
                coin_zncc[cand] = 0.0
        zncc_scores_per_coin.append(coin_zncc)

    # 4. Pesi dinamici ZNCC per moneta:
    # Aumentano se il colore è incerto (1 - conf) e/o se i raggi sono simili ad altre monete
    zncc_weights = []
    for i in range(n_coins):
        if use_soft_csp:
            conf_i = min(float(coins[i].get("confidence", 0.70)), 1.0)
            color_unc = max(0.0, 1.0 - conf_i)
            w_i = w_base_zncc * (1.0 + alpha_color * color_unc + beta_dim * dim_uncertainty[i])
        else:
            w_i = 0.0
        zncc_weights.append(w_i)

    # 5. Valutazione di tutte le tuple candidate
    best_tuple = None
    best_score = -1e9
    best_details = {}

    for tup in all_tuples:
        # 5.1 Stima scala px/mm
        scales = [radii[i] / (diameters[tup[i]] / 2.0) for i in range(n_coins)]
        mean_scale = float(np.mean(scales))
        std_scale = float(np.std(scales))
        scale_cv = float(std_scale / (mean_scale + 1e-5))

        # 5.2 Rapporti a coppie e monotonicità
        ratio_errors = []
        violations = 0
        for i in range(n_coins):
            for j in range(i + 1, n_coins):
                r_ratio = radii[i] / max(radii[j], 1e-5)
                d_ratio = diameters[tup[i]] / max(diameters[tup[j]], 1e-5)
                rel_err = abs(r_ratio - d_ratio) / d_ratio
                ratio_errors.append(rel_err)

                # Violazione di monotonicità
                if radii[i] > radii[j] + mono_tol and diameters[tup[i]] < diameters[tup[j]]:
                    violations += 1
                elif radii[j] > radii[i] + mono_tol and diameters[tup[j]] < diameters[tup[i]]:
                    violations += 1

        mean_ratio_err = float(np.mean(ratio_errors)) if ratio_errors else 0.0
        geom_score = - (w_ratio * mean_ratio_err) - (w_cv * scale_cv) - (w_viol * violations)

        # 5.3 Penalità Colore (solo per assegnazioni a gruppo diverso da quello osservato)
        color_penalties = []
        for i in range(n_coins):
            pred_kind = coins[i].get("kind", "gold")
            cand_kind = CLASS_TO_KIND.get(tup[i], pred_kind)
            if cand_kind != pred_kind:
                conf_i = min(float(coins[i].get("confidence", 0.70)), 1.0)
                pen = w_color_pen * conf_i
                color_penalties.append(pen)
            else:
                color_penalties.append(0.0)
        total_color_pen = sum(color_penalties) if use_soft_csp else 0.0

        # 5.4 Bonus ZNCC
        zncc_bonuses = []
        for i in range(n_coins):
            z_score = zncc_scores_per_coin[i].get(tup[i], 0.0)
            z_val = max(0.0, z_score)
            zncc_bonuses.append(zncc_weights[i] * z_val)
        total_zncc_bonus = sum(zncc_bonuses) if use_soft_csp else 0.0

        # Score totale combinato
        total_score = geom_score - total_color_pen + total_zncc_bonus

        if total_score > best_score:
            best_score = total_score
            best_tuple = tup
            best_details = {
                "mean_scale_px_per_mm": round(mean_scale, 2),
                "scale_cv_pct": round(scale_cv * 100, 2),
                "mean_ratio_error_pct": round(mean_ratio_err * 100, 2),
                "monotonicity_violations": violations,
                "geom_score": round(geom_score, 3),
                "color_penalty": round(total_color_pen, 3),
                "zncc_bonus": round(total_zncc_bonus, 3),
                "total_score": round(total_score, 3),
                "scales_per_coin": [round(s, 2) for s in scales],
                "zncc_per_coin": [round(zncc_scores_per_coin[i].get(tup[i], 0.0), 3) for i in range(n_coins)],
                "zncc_scores_all": zncc_scores_per_coin
            }

    if best_tuple is None and all_tuples:
        best_tuple = all_tuples[0]

    return list(best_tuple), best_score, best_details


def create_labeled_denomination_image(
    roi_bgr: np.ndarray,
    predicted_class: str,
    scale_val: float = 0.0,
    zncc_score: Optional[float] = None,
    canvas_size: int = 250
) -> np.ndarray:
    """
    Crea l'immagine della ROI con banner in sovraimpressione che indica
    la classe finale ('class') e lo score ZNCC (se disponibile).
    Es: '50CENT (0.35)', '2CENT', '1EURO (0.91)'.
    """
    labeled = roi_bgr.copy()

    # Mappa colori per denominazione (BGR)
    color_map = {
        "1euro": (0, 215, 255),    # Giallo brillante
        "2euro": (255, 200, 50),   # Argento / azzurro
        "10cent": (0, 215, 255),   # Oro
        "20cent": (0, 215, 255),
        "50cent": (0, 215, 255),
        "1cent": (50, 127, 230),   # Rame
        "2cent": (50, 127, 230),
        "5cent": (50, 127, 230)
    }
    label_color = color_map.get(predicted_class, (255, 255, 255))

    # Banner rettangolare semitrasparente in alto (0..34 px)
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_size, 34), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, labeled, 0.25, 0, labeled)

    # Bordo inferiore sottile al banner
    cv2.line(labeled, (0, 34), (canvas_size, 34), (80, 80, 80), 1)

    # Testo con la classe predetta ed eventuale score ZNCC
    if zncc_score is not None and zncc_score > 0:
        label_text = f"{predicted_class.upper()} ({zncc_score:.2f})"
        font_scale = 0.58
    else:
        label_text = predicted_class.upper()
        font_scale = 0.65

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    tx = max(8, (canvas_size - tw) // 2)
    ty = 24

    # Ombra nera per contrasto e testo principale
    cv2.putText(labeled, label_text, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(labeled, label_text, (tx, ty), font, font_scale, label_color, thickness, cv2.LINE_AA)

    return labeled


def extract_clean_coin_roi(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    canvas_size: int = 250,
    target_radius: float = 100.0
) -> np.ndarray:
    """
    Estrae la ROI quadrata e centrata della moneta, normalizzata a canvas_size x canvas_size
    con raggio normalizzato target_radius, applicando una maschera circolare pulita (senza testi pregressi).
    """
    center = canvas_size / 2.0
    scale = target_radius / max(radius, 1.0)
    T = np.float32([
        [scale, 0, center - cx * scale],
        [0, scale, center - cy * scale]
    ])
    roi = cv2.warpAffine(
        img_bgr,
        T,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    yy, xx = np.ogrid[:canvas_size, :canvas_size]
    circle_mask = ((xx - center) ** 2 + (yy - center) ** 2) <= (target_radius ** 2)
    roi_masked = roi.copy()
    roi_masked[~circle_mask] = 0
    return roi_masked


def load_target_image(
    img_name: str,
    denoised_images: Optional[Dict[str, np.ndarray]] = None,
    image_cache: Optional[Dict[str, np.ndarray]] = None
) -> Optional[np.ndarray]:
    """Carica l'immagine target a colori (privilegiando il denoised di Modulo 4)."""
    if denoised_images and img_name in denoised_images:
        return denoised_images[img_name]

    if image_cache is not None and img_name in image_cache:
        return image_cache[img_name]

    # Prova da artifacts Modulo 4 (module4_denoise_colored)
    pipeline_dir = Path(__file__).resolve().parents[2]
    denoised_path = pipeline_dir / "artifacts" / "Flow_B_Target" / "module4_denoise_colored" / f"{img_name}.png"
    if denoised_path.exists():
        img = cv2.imread(str(denoised_path))
        if image_cache is not None:
            image_cache[img_name] = img
        return img

    # Fallback su target_set originale
    project_root = Path(__file__).resolve().parents[3]
    target_path = project_root / "coin_dataset" / "target_set" / f"{img_name}.jpg"
    if target_path.exists():
        img = cv2.imread(str(target_path))
        if image_cache is not None:
            image_cache[img_name] = img
        return img

    return None


def classify_multi_coin_images(
    annotated_images: Dict[str, Any],
    reference_matrices: Optional[Dict[str, np.ndarray]] = None,
    denoised_images: Optional[Dict[str, np.ndarray]] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Esegue la classificazione per tutte le immagini che contengono MONETE MULTIPLE (len(coins) > 1)
    tramite Soft-CSP Multi-Modale (Invarianza di Scala Relativa, Colore Morbido e Polar ZNCC):
    1. Per ciascuna immagine con monete multiple:
       - Risolve l'assegnazione ottima delle classi tramite Soft-CSP multi-modale.
       - Assegna il campo 'class', 'polar_zncc_score' e metadati a ogni moneta dell'immagine.
       - Salva le ROI etichettate ({nome_file}_{coin_id}.png) negli artifacts.
    2. Aggiorna in append 'annotated_images.json'.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
            - updated_annotated: Data structure aggiornata con la classificazione di tutte le monete multiple.
            - labeled_multi_rois: Dizionario {f"{image_name}_{coin_id}": roi_bgr_etichettata}.
    """
    if params is None:
        params = load_or_create_multi_coin_params()

    if reference_matrices is None and params.get("use_soft_csp", True):
        try:
            reference_matrices = load_reference_polar_matrices()
        except Exception as e:
            print(f"[Warning] Impossibile caricare reference polar matrices: {e}")
            reference_matrices = None

    if output_dir is not None:
        out_path = Path(output_dir)
        if not out_path.is_absolute() and not out_path.exists():
            proj_root = Path(__file__).resolve().parents[3]
            if (proj_root / out_path).parent.exists():
                out_path = proj_root / out_path
    else:
        out_path = get_default_artifacts_dir()

    if save_artifacts:
        out_path.mkdir(parents=True, exist_ok=True)

    updated_annotated: Dict[str, Any] = json.loads(json.dumps(annotated_images))
    labeled_multi_rois: Dict[str, np.ndarray] = {}
    image_cache: Dict[str, np.ndarray] = {}

    multi_coin_images = [img for img, data in updated_annotated.items() if len(data.get("coins", [])) > 1]
    total_multi_coins = sum(len(updated_annotated[img]["coins"]) for img in multi_coin_images)

    mode_label = "Soft-CSP Multi-Modale (Geometria + Colore + ZNCC)" if params.get("use_soft_csp", True) else "Hard-CSP Geometrico"
    print(f"\n--- Modulo 7.2: Classificazione Monete Multiple [{mode_label}] ---")
    print(f"  - Immagini con monete multiple: {len(multi_coin_images)}")
    print(f"  - Totale monete da classificare: {total_multi_coins}")

    class_counts: Dict[str, int] = {}
    processed_images = 0

    for img_name in multi_coin_images:
        entry = updated_annotated[img_name]
        coins_list = entry["coins"]

        # Carica l'immagine target sorgente pulita (denoised)
        full_img = load_target_image(img_name, denoised_images=denoised_images, image_cache=image_cache)

        # 1. Risoluzione dell'assegnazione ottimale Soft-CSP
        best_classes, score, details = solve_multi_coin_assignment(
            coins=coins_list,
            params=params,
            reference_matrices=reference_matrices,
            img_name=img_name
        )

        # 2. Assegnazione delle classi e salvataggio metadati
        for idx, coin in enumerate(coins_list):
            coin_id = coin.get("coin_id", 0)
            coin_key = f"{img_name}_{coin_id}"
            assigned_class = best_classes[idx]
            scale_val = details.get("scales_per_coin", [0.0] * len(coins_list))[idx]
            zncc_val = details.get("zncc_per_coin", [0.0] * len(coins_list))[idx]

            coin["class"] = assigned_class
            coin["classification_method"] = "multi_coin_soft_csp_zncc" if params.get("use_soft_csp", True) else "multi_coin_relative_scale"
            coin["estimated_scale_px_per_mm"] = scale_val
            coin["assignment_score"] = round(score, 3)
            coin["polar_zncc_score"] = zncc_val
            if "zncc_scores_all" in details and idx < len(details["zncc_scores_all"]):
                coin["polar_matching_scores"] = {c: round(s, 4) for c, s in details["zncc_scores_all"][idx].items()}

            class_counts[assigned_class] = class_counts.get(assigned_class, 0) + 1

            # 3. Creazione e salvataggio ROI etichettata pulita
            if full_img is not None:
                cx, cy = float(coin["center"][0]), float(coin["center"][1])
                radius = float(coin["radius"])
                base_roi = extract_clean_coin_roi(full_img, cx, cy, radius, canvas_size=250, target_radius=100.0)
            else:
                base_roi = np.zeros((250, 250, 3), dtype=np.uint8)

            labeled_roi = create_labeled_denomination_image(
                base_roi,
                assigned_class,
                scale_val=scale_val,
                zncc_score=zncc_val,
                canvas_size=250
            )
            labeled_multi_rois[coin_key] = labeled_roi

            if save_artifacts:
                cv2.imwrite(str(out_path / f"{coin_key}.png"), labeled_roi)

        processed_images += 1
        if processed_images % 30 == 0 or processed_images == len(multi_coin_images):
            print(f"  [{processed_images}/{len(multi_coin_images)}] Immagini multiple classificate...")

    # 4. Aggiornamento finale del manifest unificato annotated_images.json
    if save_artifacts:
        ann_main_path = get_default_annotated_images_path()
        ann_main_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_main_path, "w", encoding="utf-8") as f:
            json.dump(updated_annotated, f, indent=2)
        print(f"Aggiornato manifest unificato 'annotated_images.json' in: {ann_main_path}")
        print(f"Salvati artefatti Modulo 7.2 in: {out_path.name}/")

    print(f"\nRiepilogo Classificazione Monete Multiple (Modulo 7.2 - {total_multi_coins} monete):")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_multi_coins * 100) if total_multi_coins > 0 else 0
        print(f"  - {cls_name:7s}: {count:3d} monete ({pct:5.1f}%)")

    return updated_annotated, labeled_multi_rois


def plot_multi_coin_classification_sample(
    updated_annotated: Dict[str, Any],
    labeled_rois: Dict[str, np.ndarray],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (16, 8),
    show_plot: bool = True
):
    """
    Mostra a schermo una griglia di campioni di monete multiple classificate,
    con la denominazione scelta in sovraimpressione.
    """
    if not labeled_rois:
        print("Nessuna moneta multipla classificata disponibile per la visualizzazione.")
        return

    if seed is not None:
        random.seed(seed)

    keys = list(labeled_rois.keys())
    sample_keys = random.sample(keys, min(n_samples, len(keys)))

    ncols = 4
    nrows = int(np.ceil(len(sample_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, key in enumerate(sample_keys):
        ax = axes[idx]
        roi_bgr = labeled_rois[key]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(roi_rgb)
        ax.set_title(f"Moneta: {key}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(len(sample_keys), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Modulo 7.2: Campione Monete Multiple Classificate ({len(sample_keys)} monete)", fontsize=14, y=0.98)
    plt.tight_layout()
    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys

    ann_path = get_default_annotated_images_path()
    if not ann_path.exists():
        print(f"Errore: {ann_path} non trovato. Esegui prima i Moduli precedenti.")
        sys.exit(1)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotated_images = json.load(f)

    updated_ann, labeled_rois = classify_multi_coin_images(
        annotated_images=annotated_images,
        save_artifacts=True
    )
