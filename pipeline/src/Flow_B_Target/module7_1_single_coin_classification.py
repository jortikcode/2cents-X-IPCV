"""Modulo 7.1 (Flow B): Classificazione Moneta Singola (Pruning Colore + Polar ZNCC Matching).

Questo modulo è dedicato alla classificazione delle immagini che contengono una SINGOLA moneta.
(Il caso delle monete multiple è riservato al Modulo 7.2).

Logica della pipeline:
1. Filtraggio:
   - Isola le immagini con esattamente 1 moneta (len(coins) == 1).
   - Le immagini con monete multiple vengono saltate temporaneamente in attesa del Modulo 7.2.
2. Pruning dei candidati tramite la tipologia di colore ('kind') estratta nel Modulo 5:
   - '1 euro' -> candidati: ['1euro'] (classificazione immediata)
   - '2 euro' -> candidati: ['2euro'] (classificazione immediata)
   - 'gold'   -> candidati: ['10cent', '20cent', '50cent']
   - 'copper' -> candidati: ['1cent', '2cent', '5cent']
3. Matching Polar Unwarp tramite ZNCC Circolare (Zero-Normalized Cross-Correlation):
   - Per ciascun candidato, confronta la matrice polare (360x50) del target con la matrice
     di riferimento del Flusso A lungo tutti i 360 shift angolari (invarianza a rotazione).
   - Seleziona il candidato con il punteggio massimo di ZNCC.
4. Output e Artefatti:
   - Aggiunge in append il campo "class": "<taglio>" alla moneta in 'annotated_images.json'.
   - Salva negli artifacts (artifacts/Flow_B_Target/module7_1_single_coin_classification/):
     * Immagine ROI etichettata '{nome_file}_{coin_id}.png' con il taglio classificato e score ZNCC.
     * Manifest 'single_coin_classified_manifest.json'.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_default_conf_dir() -> Path:
    """Restituisce il percorso della cartella pipeline/conf/Flow_B_Target."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "conf" / "Flow_B_Target"


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module7_1_single_coin_classification."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module7_1_single_coin_classification"


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module3_coin_detection" / "annotated_images.json"


def get_default_reference_polar_dir() -> Path:
    """Restituisce il percorso della cartella delle matrici polari di riferimento (Flusso A, Modulo 5)."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module5_polar_unwarp"


def load_or_create_single_coin_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di classificazione singola moneta da file JSON o crea i default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "single_coin_classification_params.json"
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
            print(f"Parametri Modulo 7.1 caricati da: {json_path.name}")
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path}: {e}. Uso parametri default.")

    params = {
        "reference_polar_dir": "pipeline/artifacts/Flow_A_Reference/module5_polar_unwarp",
        "metric": "circular_zncc",
        "kind_to_candidates": {
            "1 euro": ["1euro"],
            "2 euro": ["2euro"],
            "gold": ["10cent", "20cent", "50cent"],
            "copper": ["1cent", "2cent", "5cent"]
        }
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Creato file di configurazione: {json_path.name}")
    return params


def load_reference_polar_matrices(ref_dir: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """
    Carica le matrici polar unwarp delle 8 monete di riferimento dal Flusso A.
    """
    if ref_dir is None:
        ref_path = get_default_reference_polar_dir()
    else:
        ref_path = Path(ref_dir)
        if not ref_path.exists():
            proj_root = Path(__file__).resolve().parents[3]
            if (proj_root / ref_path).exists():
                ref_path = proj_root / ref_path
            else:
                ref_path = get_default_reference_polar_dir()

    if not ref_path.exists():
        raise FileNotFoundError(f"Cartella matrici polari di riferimento non trovata: {ref_path}")

    # Prova prima dal file aggregato polar_matrices.npz
    npz_path = ref_path / "polar_matrices.npz"
    if npz_path.exists():
        npz_data = np.load(str(npz_path))
        matrices = {key: npz_data[key].astype(np.float32) for key in npz_data.files}
        print(f"Caricate {len(matrices)} matrici polari di riferimento da: {npz_path.name}")
        return matrices

    # Fallback: caricamento dai singoli file .npy
    matrices = {}
    for npy_file in ref_path.glob("*_polar.npy"):
        coin_name = npy_file.stem.replace("_polar", "")
        matrices[coin_name] = np.load(str(npy_file)).astype(np.float32)

    print(f"Caricate {len(matrices)} matrici polari di riferimento da: {ref_path.name}")
    return matrices


def compute_circular_zncc(target: np.ndarray, reference: np.ndarray) -> Tuple[float, int]:
    """
    Calcola la Zero-Normalized Cross-Correlation circolare lungo l'asse angolare (360 bins).
    
    Supporta matrici polari orizzontali (50 righe r x 360 colonne theta) ed effettua
    la correlazione circolare lungo l'asse angolare (asse 1, colonne).
    (Supporta retrocompatibilmente anche matrici verticali 360x50 se fornite).
    
    Usando la FFT lungo l'asse angolare:
    - Zero-mean: sottrae la media globale per eliminare bias di illuminazione / densità bordi
    - Normalizzazione: divide per deviazione standard
    - Cross-correlazione circolare calcolata per tutti i 360 angoli contemporaneamente
    
    Returns:
        Tuple[float, int]:
            - best_score: punteggio ZNCC massimo (compreso tra -1.0 e 1.0)
            - best_shift_deg: angolo di rotazione migliore in gradi (0..359°)
    """
    # 1. Normalizzazione a media zero e varianza unitaria (ZNCC)
    t = target.astype(np.float64)
    r = reference.astype(np.float64)

    t_mean = np.mean(t)
    t_std = np.std(t)
    if t_std < 1e-7:
        return 0.0, 0
    t_norm = (t - t_mean) / (t_std * np.sqrt(t.size))

    r_mean = np.mean(r)
    r_std = np.std(r)
    if r_std < 1e-7:
        return 0.0, 0
    r_norm = (r - r_mean) / (r_std * np.sqrt(r.size))

    # Individuazione dell'asse angolare (la dimensione con 360 bins)
    # Default matrice orizzontale (50 righe r, 360 colonne theta): ang_axis = 1, rad_axis = 0
    if t.shape[1] == 360 or t.shape[0] == 50:
        ang_axis = 1
        rad_axis = 0
    else:
        ang_axis = 0
        rad_axis = 1

    # 2. Correlazione circolare tramite FFT lungo l'asse angolare
    T_fft = np.fft.fft(t_norm, axis=ang_axis)
    R_fft = np.fft.fft(r_norm, axis=ang_axis)

    # Teorema della correlazione: IFFT(T * conj(R))
    corr_matrix = np.fft.ifft(T_fft * np.conj(R_fft), axis=ang_axis).real

    # Somma lungo l'asse radiale per ottenere la correlazione angolare aggregata (360 bins)
    corr_profile = np.sum(corr_matrix, axis=rad_axis)

    best_shift = int(np.argmax(corr_profile))
    best_score = float(corr_profile[best_shift])

    return best_score, best_shift


def create_labeled_denomination_image(
    roi_bgr: np.ndarray,
    predicted_class: str,
    score: float,
    canvas_size: int = 250
) -> np.ndarray:
    """
    Crea l'immagine della ROI con banner in sovraimpressione che indica
    la classe finale ('class') e il punteggio ZNCC di matching.
    Es: '10CENT (0.28)', '1EURO (0.95)'.
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

    # Testo: CLASSE (score)
    label_text = f"{predicted_class.upper()} ({score:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    tx = max(8, (canvas_size - tw) // 2)
    ty = 23

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


def classify_single_coin_images(
    annotated_images: Dict[str, Any],
    reference_matrices: Optional[Dict[str, np.ndarray]] = None,
    denoised_images: Optional[Dict[str, np.ndarray]] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Esegue la classificazione vera e propria per le sole immagini che contengono una SINGOLA moneta:
    1. Filtra le immagini con len(coins) == 1 (le multiple vengono saltate).
    2. Restringe i candidati tramite il colore 'kind':
       - '1 euro' -> ['1euro']
       - '2 euro' -> ['2euro']
       - 'gold'   -> ['10cent', '20cent', '50cent']
       - 'copper' -> ['1cent', '2cent', '5cent']
    3. Carica la matrice polare della moneta e calcola la cross-correlazione ZNCC con le reference.
    4. Sceglie il candidato con lo score più alto e assegna il campo 'class': <taglio>.
    5. Salva le immagini ROI etichettate negli artifacts e aggiorna 'annotated_images.json'.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
            - updated_annotated: Data structure aggiornata con il campo 'class' sulle monete singole.
            - labeled_single_rois: Dizionario {f"{image_name}_{coin_id}": roi_bgr_etichettata}.
    """
    if params is None:
        params = load_or_create_single_coin_params()

    if reference_matrices is None:
        ref_dir = params.get("reference_polar_dir")
        reference_matrices = load_reference_polar_matrices(ref_dir)

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

    kind_to_candidates = params.get("kind_to_candidates", {
        "1 euro": ["1euro"],
        "2 euro": ["2euro"],
        "gold": ["10cent", "20cent", "50cent"],
        "copper": ["1cent", "2cent", "5cent"]
    })

    proj_root = Path(__file__).resolve().parents[3]
    updated_annotated: Dict[str, Any] = json.loads(json.dumps(annotated_images))
    labeled_single_rois: Dict[str, np.ndarray] = {}
    image_cache: Dict[str, np.ndarray] = {}

    single_coin_images = [img for img, data in updated_annotated.items() if len(data.get("coins", [])) == 1]
    multi_coin_images = [img for img, data in updated_annotated.items() if len(data.get("coins", [])) > 1]

    print(f"\n--- Ramo Decisionale Modulo 7 ---")
    print(f"  - Immagini con moneta singola (elaborate da Modulo 7.1): {len(single_coin_images)}")
    print(f"  - Immagini con monete multiple (saltate, riservate a Modulo 7.2): {len(multi_coin_images)}")

    class_counts: Dict[str, int] = {}
    classified_single_coins = 0

    for img_name in single_coin_images:
        entry = updated_annotated[img_name]
        coin = entry["coins"][0]
        coin_id = coin.get("coin_id", 0)
        coin_key = f"{img_name}_{coin_id}"

        # 1. Recupero della tipologia di colore dal Modulo 5
        kind = coin.get("kind", "gold")
        candidates = kind_to_candidates.get(kind, ["10cent", "20cent", "50cent"])

        # 2. Caricamento della matrice polare estratta dal Modulo 6
        polar_path_str = coin.get("polar_unwarp_path", "")
        polar_file = proj_root / polar_path_str
        if not polar_file.exists():
            # Fallback
            polar_file = Path(__file__).resolve().parents[2] / "artifacts" / "Flow_B_Target" / "module6_polar_unwarp" / f"{coin_key}_polar.npy"

        if not polar_file.exists():
            print(f"[Warning] Matrice polare non trovata per {coin_key}: {polar_file}. Skip.")
            continue

        target_polar = np.load(str(polar_file))

        # 3. Matching ZNCC contro ciascun candidato del gruppo ristretto
        candidate_scores: Dict[str, float] = {}
        candidate_shifts: Dict[str, int] = {}

        for cand in candidates:
            if cand in reference_matrices:
                ref_polar = reference_matrices[cand]
                score, shift = compute_circular_zncc(target_polar, ref_polar)
                candidate_scores[cand] = round(score, 4)
                candidate_shifts[cand] = shift
            else:
                candidate_scores[cand] = 0.0
                candidate_shifts[cand] = 0

        # Selezione del candidato vincente con ZNCC più alto
        best_cand = max(candidate_scores, key=candidate_scores.get)
        best_score = candidate_scores[best_cand]
        best_shift = candidate_shifts[best_cand]

        # 4. Assegnazione della classe definitiva alla moneta
        coin["class"] = best_cand
        coin["polar_zncc_score"] = best_score
        coin["polar_rotation_deg"] = best_shift
        coin["polar_matching_scores"] = candidate_scores
        coin["classification_method"] = "single_coin_polar_zncc"

        class_counts[best_cand] = class_counts.get(best_cand, 0) + 1
        classified_single_coins += 1

        # 5. Creazione e salvataggio dell'immagine ROI con etichetta pulita negli artifacts
        # Estrae la ROI pulita direttamente dall'immagine sorgente denoised (evitando scritte pregresse)
        full_img = load_target_image(img_name, denoised_images=denoised_images, image_cache=image_cache)
        if full_img is not None:
            cx, cy = float(coin["center"][0]), float(coin["center"][1])
            radius = float(coin["radius"])
            base_roi = extract_clean_coin_roi(full_img, cx, cy, radius, canvas_size=250, target_radius=100.0)
        else:
            base_roi = np.zeros((250, 250, 3), dtype=np.uint8)

        labeled_roi = create_labeled_denomination_image(base_roi, best_cand, best_score, canvas_size=250)
        labeled_single_rois[coin_key] = labeled_roi

        if save_artifacts:
            cv2.imwrite(str(out_path / f"{coin_key}.png"), labeled_roi)

    # 6. Aggiornamento in append di annotated_images.json principale e manifest Modulo 7.1
    if save_artifacts:
        ann_main_path = get_default_annotated_images_path()
        if ann_main_path.parent.exists():
            with open(ann_main_path, "w", encoding="utf-8") as f:
                json.dump(updated_annotated, f, indent=2)
            print(f"Aggiornato 'annotated_images.json' in: {ann_main_path.name}")

        with open(out_path / "single_coin_classified_manifest.json", "w", encoding="utf-8") as f:
            json.dump(updated_annotated, f, indent=2)
        print(f"Salvati artefatti Modulo 7.1 in: {out_path.name}/")

    print(f"\nRiepilogo Classificazione Monete Singole (Modulo 7.1 - {classified_single_coins} monete):")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / classified_single_coins * 100) if classified_single_coins > 0 else 0
        print(f"  - {cls_name:7s}: {count:2d} monete ({pct:5.1f}%)")

    return updated_annotated, labeled_single_rois


def plot_single_coin_classification_sample(
    updated_annotated: Dict[str, Any],
    labeled_rois: Dict[str, np.ndarray],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (16, 8),
    show_plot: bool = True
):
    """
    Mostra a schermo una griglia di campioni di monete singole classificate,
    con la denominazione scelta e il punteggio ZNCC.
    """
    if not labeled_rois:
        print("Nessuna moneta singola classificata disponibile per la visualizzazione.")
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

    plt.suptitle(f"Modulo 7.1: Campione Monete Singole Classificate ({len(sample_keys)} monete)", fontsize=14, y=0.98)
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

    updated_ann, labeled_rois = classify_single_coin_images(
        annotated_images=annotated_images,
        save_artifacts=True
    )
