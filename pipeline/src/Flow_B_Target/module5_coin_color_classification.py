"""Modulo 5 (Flow B): Estrazione ROI mascherata, classificazione cromatica e calcolo confidence.

Questo modulo:
1. Per OGNI IMMAGINE e per OGNI MONETA rilevata (da 'annotated_images.json'):
   - Estrae la ROI mascherata circolare (normalizzata a 250x250 px, R=100 px).
   - Estrae le caratteristiche cromatiche:
     * Delta b* (anello esterno - centro) per la discriminazione bimetallica (1 euro vs 2 euro).
     * Coordinate cromatiche (a*, b*) e tonalità Hue con White-Patch per la separazione
       monometallica (copper vs gold).
   - Confronta i valori con i profili di riferimento estratti dal Flusso A
     (pipeline/artifacts/Flow_A_Reference/module3_color/color_profiles.json).
   - Determina la tipologia cromatico-metallica ('kind') in una delle 4 classi:
     * '1 euro'
     * '2 euro'
     * 'gold'
     * 'copper'
   - Calcola un punteggio di confidence continuo in [0.50, 1.00] (Metodo 1: Margine di Decisione Normalizzato)
     in base alla distanza dai margini decisionali e dai centroidi di riferimento.
2. Aggiunge in append il campo "kind": "<kind>" e "confidence": <valore> a ogni oggetto moneta
   nella data structure di 'annotated_images.json' e aggiorna il file JSON.
3. Salva negli artifacts di output (pipeline/artifacts/Flow_B_Target/module5_coin_color_classification/):
   - Una immagine della ROI mascherata per ciascuna moneta: '{nome_file}_{coin_id}.png'
     con una scritta in sovraimpressione indicante la tipologia e la percentuale di confidenza (es. 'GOLD (92%)').
   - Il file JSON aggregato 'classified_annotated_images.json'.
4. Restituisce in memoria la data structure aggiornata e il dizionario delle ROI estratte.
5. Mostra a video una griglia di campioni rappresentativi delle monete classificate.
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
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module5_coin_color_classification."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module5_coin_color_classification"


def get_default_flow_a_color_profiles_path() -> Path:
    """Restituisce il percorso del file color_profiles.json estratto dal Flusso A."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module3_color" / "color_profiles.json"


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json generato dal Modulo 3."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module3_coin_detection" / "annotated_images.json"


def load_or_create_classification_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di configurazione dal file JSON se esiste, altrimenti inizializza con i default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "coin_color_classification_params.json"
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
            print(f"Parametri Modulo 5 caricati da: {json_path.name}")
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path}: {e}. Uso parametri default.")

    params = {
        "color_profiles_path": "pipeline/artifacts/Flow_A_Reference/module3_color/color_profiles.json",
        "bimetal_tau": 7.69,
        "hue_threshold": 17.5,
        "roi_canvas_size": 250,
        "roi_target_radius": 100.0,
        "kinds": ["1 euro", "2 euro", "gold", "copper"]
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Creato file di configurazione: {json_path.name}")
    return params


def load_reference_color_profiles(profiles_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i profili di colore delle reference generati dal Flusso A (Modulo 3)."""
    if profiles_path is None:
        profiles_path = get_default_flow_a_color_profiles_path()
    else:
        profiles_path = Path(profiles_path)
        if not profiles_path.exists():
            # Risoluzione robusta rispetto alla root di progetto o alla cartella pipeline
            proj_root = Path(__file__).resolve().parents[3]
            candidate_root = proj_root / profiles_path
            if candidate_root.exists():
                profiles_path = candidate_root
            else:
                candidate_pipeline = Path(__file__).resolve().parents[2] / profiles_path
                if candidate_pipeline.exists():
                    profiles_path = candidate_pipeline
                else:
                    profiles_path = get_default_flow_a_color_profiles_path()

    if not profiles_path.exists():
        raise FileNotFoundError(f"File profili colore del Flusso A non trovato: {profiles_path}")

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"Profili colore Flusso A caricati da: {profiles_path.name}")
    return profiles


def extract_masked_coin_roi(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    canvas_size: int = 250,
    target_radius: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae la ROI quadrata e centrata della moneta, normalizzata a canvas_size x canvas_size
    con raggio normalizzato target_radius, applicando una maschera circolare (pixel esterni neri).
    
    Args:
        img_bgr: Immagine intera a colori (formato BGR).
        cx, cy: Coordinate del centro del cerchio identificato da Hough.
        radius: Raggio della moneta in pixel.
        canvas_size: Dimensione del canvas quadrato di output (default: 250 px).
        target_radius: Raggio normalizzato nel canvas (default: 100.0 px).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - roi_masked: ROI mascherata (canvas_size x canvas_size x 3, BGR).
            - mask: Maschera booleana (canvas_size x canvas_size).
    """
    center = canvas_size / 2.0
    scale = target_radius / max(radius, 1.0)

    # Matrice di trasformazione affine: centratura e ridimensionamento
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

    # Maschera circolare binaria per azzerare lo sfondo oltre il disco della moneta
    yy, xx = np.ogrid[:canvas_size, :canvas_size]
    circle_mask = ((xx - center) ** 2 + (yy - center) ** 2) <= (target_radius ** 2)

    roi_masked = roi.copy()
    roi_masked[~circle_mask] = 0

    return roi_masked, circle_mask


def classify_coin_color(
    roi_bgr: np.ndarray,
    profiles: Dict[str, Any],
    cx: float = 125.0,
    cy: float = 125.0,
    r: float = 100.0,
    hue_threshold: float = 17.5
) -> Tuple[str, float, Dict[str, float]]:
    """
    Estrae i vettori cromatici dalla ROI mascherata e classifica la tipologia ('kind') della moneta
    in una delle 4 classi: '1 euro', '2 euro', 'copper', 'gold', con relativo punteggio di confidence.
    
    Confidence (Metodo 1: Margine di Decisione Normalizzato):
    - Bimetalliche: margine di Delta b* oltre la soglia tau, rispetto al valore di riferimento atteso:
        conf = 0.50 + 0.49 * clip((|Delta b*| - tau) / (Delta_ref - tau), 0.0, 1.0)
    - Monometalliche: margine relativo tra le distanze euclidee dai centroidi (d_lose - d_win)/(d_lose + d_win)
      combinato con la separazione Hue e modulato dalla sicurezza di essere monometallica (|Delta b*| <= tau):
        conf = (0.50 + 0.48 * combined_margin) * (0.85 + 0.15 * bimetal_safety)
        
    Returns:
        Tuple[str, float, Dict[str, float]]:
            - predicted_kind: '1 euro', '2 euro', 'copper', 'gold'
            - confidence: float in [0.50, 0.99] arrotondato a 3 decimali
            - metrics: dizionario con tutti i parametri cromatici estratti
    """
    bimetal_tau = float(profiles.get("decision_rules", {}).get("bimetal_threshold_tau", 7.69))
    delta_b_1euro_ref = float(profiles.get("classes", {}).get("1euro", {}).get("delta_b", 14.08))
    delta_b_2euro_ref = float(profiles.get("classes", {}).get("2euro", {}).get("delta_b", -13.13))

    # Centroidi di riferimento per copper e gold dallo spazio (a*, b*) di Flow A
    copper_centroid = profiles.get("classes", {}).get("copper", {}).get("centroid", {"a_star": 149.18, "b_star": 145.1})
    gold_centroid = profiles.get("classes", {}).get("gold", {}).get("centroid", {"a_star": 141.68, "b_star": 148.49})

    # Calcolo coordinate polari/distanze dal centro della ROI
    ys, xs = np.ogrid[:roi_bgr.shape[0], :roi_bgr.shape[1]]
    dist = np.hypot(xs - cx, ys - cy)

    centre_mask = dist < (0.60 * r)
    ring_mask = (dist >= 0.80 * r) & (dist < 0.90 * r)
    disc_mask = dist < (0.85 * r)

    # 1. Spazio CIELAB per Delta b*
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_star_plane = lab[:, :, 2]
    a_star_plane = lab[:, :, 1]

    if np.sum(ring_mask) > 0 and np.sum(centre_mask) > 0:
        delta_b = float(b_star_plane[ring_mask].mean() - b_star_plane[centre_mask].mean())
    else:
        delta_b = 0.0

    mean_a = float(a_star_plane[disc_mask].mean()) if np.sum(disc_mask) > 0 else 128.0
    mean_b = float(b_star_plane[disc_mask].mean()) if np.sum(disc_mask) > 0 else 128.0

    # 2. White-Patch per eliminare dominanti cromatiche e calcolare l'Hue
    disc_pixels_bgr = roi_bgr[disc_mask]
    if len(disc_pixels_bgr) > 0:
        brightest = np.percentile(disc_pixels_bgr, 99, axis=0)
        wb_bgr = np.clip(roi_bgr / (brightest + 1e-5) * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(wb_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
        hue_px = hsv[:, :, 0][disc_mask]
        # Hue circolare su scala 0-360: i valori > 180° rappresentano rossi negativi
        hue_px = np.where(hue_px > 180, hue_px - 360, hue_px)
        mean_hue = float(hue_px.mean())
    else:
        mean_hue = 25.0

    # Distanze euclidee dai centroidi di riferimento
    d_copper = float(np.hypot(mean_a - copper_centroid["a_star"], mean_b - copper_centroid["b_star"]))
    d_gold = float(np.hypot(mean_a - gold_centroid["a_star"], mean_b - gold_centroid["b_star"]))

    # Decision Tree: Bimetalliche vs Monometalliche + Calcolo Confidence (Metodo 1)
    if delta_b > bimetal_tau:
        predicted_kind = "1 euro"
        span_1euro = max(delta_b_1euro_ref - bimetal_tau, 1.0)
        raw_margin = min(1.0, max(0.0, (delta_b - bimetal_tau) / span_1euro))
        confidence = 0.50 + 0.49 * raw_margin
    elif delta_b < -bimetal_tau:
        predicted_kind = "2 euro"
        span_2euro = max(abs(delta_b_2euro_ref) - bimetal_tau, 1.0)
        raw_margin = min(1.0, max(0.0, (abs(delta_b) - bimetal_tau) / span_2euro))
        confidence = 0.50 + 0.49 * raw_margin
    else:
        # Monometallica: combinazione di Hue e vicinanza ai centroidi Lab
        if mean_hue < hue_threshold:
            predicted_kind = "copper"
        else:
            predicted_kind = "gold"

        d_win = d_copper if predicted_kind == "copper" else d_gold
        d_lose = d_gold if predicted_kind == "copper" else d_copper
        color_margin = (d_lose - d_win) / (d_lose + d_win + 1e-5)
        color_margin = float(np.clip(color_margin, -0.5, 1.0))

        hue_dist = abs(mean_hue - hue_threshold)
        hue_margin = float(np.clip(hue_dist / 15.0, 0.0, 1.0))

        combined_margin = 0.5 * max(0.0, color_margin) + 0.5 * hue_margin
        bimetal_safety = max(0.0, 1.0 - (abs(delta_b) / bimetal_tau))

        raw_conf = 0.50 + 0.48 * combined_margin
        confidence = raw_conf * (0.85 + 0.15 * bimetal_safety)

    confidence = round(float(np.clip(confidence, 0.50, 0.99)), 3)

    metrics = {
        "delta_b": round(delta_b, 2),
        "mean_a_star": round(mean_a, 2),
        "mean_b_star": round(mean_b, 2),
        "mean_hue": round(mean_hue, 2),
        "dist_copper": round(d_copper, 2),
        "dist_gold": round(d_gold, 2),
        "kind_confidence": confidence
    }

    return predicted_kind, confidence, metrics


def create_labeled_roi_image(
    roi_bgr: np.ndarray,
    predicted_kind: str,
    confidence: float,
    canvas_size: int = 250
) -> np.ndarray:
    """
    Crea l'immagine della ROI con una scritta in sovraimpressione che indica
    la tipologia predetta ('kind') e la relativa confidence percentuale.
    Es: 'GOLD (92%)', '1 EURO (98%)', 'COPPER (75%)'.
    
    Usa un banner scuro semitrasparente in alto e testo colorato per massima leggibilità.
    """
    labeled = roi_bgr.copy()

    # Mappa colori per tipologia (BGR)
    color_map = {
        "1 euro": (0, 215, 255),    # Oro/Giallo brillante
        "2 euro": (255, 200, 50),   # Argento/Azzurro
        "gold": (0, 215, 255),      # Giallo oro
        "copper": (50, 127, 230)    # Rame/Arancio acceso
    }
    label_color = color_map.get(predicted_kind, (255, 255, 255))

    # Banner rettangolare semitrasparente in alto (0..34 px)
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_size, 34), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, labeled, 0.25, 0, labeled)

    # Bordo inferiore sottile al banner
    cv2.line(labeled, (0, 34), (canvas_size, 34), (80, 80, 80), 1)

    # Testo: Tipologia + Confidence (%)
    pct = int(round(confidence * 100))
    label_text = f"{predicted_kind.upper()} ({pct}%)"
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


def classify_all_coins(
    denoised_images: Dict[str, np.ndarray],
    annotated_images: Dict[str, Any],
    profiles: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Esegue la classificazione cromatica su tutte le monete di tutte le immagini target.
    
    1. Per ogni immagine e per ogni moneta:
       - Estrae la ROI mascherata circolare.
       - Classifica la tipologia ('kind') in: '1 euro', '2 euro', 'gold', 'copper'.
       - Calcola la confidence in base al margine di decisione normalizzato.
       - Aggiunge in append il campo 'kind': <kind> e 'confidence': <conf> a ciascun oggetto coin.
       - Salva l'immagine della ROI '{image_name}_{coin_id}.png' con etichetta e percentuale.
    2. Aggiorna 'annotated_images.json' in place e salva 'classified_annotated_images.json'.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
            - updated_annotated_images: Data structure con i campi 'kind' e 'confidence' aggiunti a ogni coin.
            - labeled_rois: Dizionario {f"{image_name}_{coin_id}": roi_etichettata_bgr}.
    """
    if params is None:
        params = load_or_create_classification_params()

    if profiles is None:
        flow_a_profiles_path = params.get("color_profiles_path")
        profiles = load_reference_color_profiles(flow_a_profiles_path)

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

    canvas_size = int(params.get("roi_canvas_size", 250))
    target_radius = float(params.get("roi_target_radius", 100.0))
    hue_threshold = float(params.get("hue_threshold", 17.5))

    updated_annotated: Dict[str, Any] = json.loads(json.dumps(annotated_images))
    labeled_rois: Dict[str, np.ndarray] = {}

    total_coins = sum(len(entry.get("coins", [])) for entry in annotated_images.values())
    print(f"Avvio classificazione cromatica su {total_coins} monete ({len(annotated_images)} immagini)...")

    kind_counts: Dict[str, int] = {"1 euro": 0, "2 euro": 0, "gold": 0, "copper": 0}
    processed_count = 0

    for img_name, entry in updated_annotated.items():
        # Recupero dell'immagine a colori denoised
        if img_name in denoised_images:
            img = denoised_images[img_name]
        else:
            # Fallback: caricamento da file artifacts di Modulo 4
            png_path = Path(__file__).resolve().parents[2] / "artifacts" / "Flow_B_Target" / "module4_denoise_colored" / f"{img_name}.png"
            if png_path.exists():
                img = cv2.imread(str(png_path))
            else:
                img_path = Path(__file__).resolve().parents[3] / entry.get("relative_image_path", "")
                img = cv2.imread(str(img_path))

        # L'immagine è già in formato standard OpenCV BGR (da Modulo 4 o cv2.imread)
        img_bgr = img.copy()

        coins_list = entry.get("coins", [])
        for coin in coins_list:
            coin_id = coin.get("coin_id", 0)
            cx, cy = float(coin["center"][0]), float(coin["center"][1])
            radius = float(coin["radius"])

            # 1. Estrazione ROI mascherata
            roi_masked, _ = extract_masked_coin_roi(
                img_bgr, cx, cy, radius,
                canvas_size=canvas_size,
                target_radius=target_radius
            )

            # 2. Classificazione della tipologia di colore con i profili del Flusso A e calcolo confidence
            pred_kind, conf, metrics = classify_coin_color(
                roi_masked,
                profiles,
                cx=canvas_size / 2.0,
                cy=canvas_size / 2.0,
                r=target_radius,
                hue_threshold=hue_threshold
            )

            # 3. Aggiornamento dell'oggetto coin: rimuove vecchio campo 'class' e aggiunge 'kind' e 'confidence'
            coin.pop("class", None)
            coin["kind"] = pred_kind
            coin["confidence"] = conf
            coin["color_metrics"] = metrics

            kind_counts[pred_kind] = kind_counts.get(pred_kind, 0) + 1
            processed_count += 1

            # 4. Creazione dell'immagine ROI con etichetta sovraimpressa (Tipologia + Confidence %)
            labeled_roi = create_labeled_roi_image(roi_masked, pred_kind, conf, canvas_size=canvas_size)
            roi_key = f"{img_name}_{coin_id}"
            labeled_rois[roi_key] = labeled_roi

            # Salvataggio artefatto: {nome_file}_{coin_id}.png
            if save_artifacts:
                roi_filename = f"{img_name}_{coin_id}.png"
                cv2.imwrite(str(out_path / roi_filename), labeled_roi)

        if processed_count % 50 == 0 or processed_count == total_coins:
            print(f"  [{processed_count}/{total_coins}] Monete classificate...")

    # Aggiornamento di annotated_images.json (nella cartella di Modulo 3 e come copia in Modulo 5)
    if save_artifacts:
        # 1. Aggiorna in append annotated_images.json principale (Modulo 3)
        ann_main_path = get_default_annotated_images_path()
        if ann_main_path.parent.exists():
            with open(ann_main_path, "w", encoding="utf-8") as f:
                json.dump(updated_annotated, f, indent=2)
            print(f"Aggiornato 'annotated_images.json' in: {ann_main_path.name}")

        # 2. Salva una copia nel manifest di Modulo 5
        out_meta_file = out_path / "classified_annotated_images.json"
        with open(out_meta_file, "w", encoding="utf-8") as f:
            json.dump(updated_annotated, f, indent=2)
        print(f"Salvati artefatti Modulo 5 in: {out_path.name}/")

    print("\nRiepilogo Classificazione Cromatica (Kind) delle Monete:")
    for knd_name, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_coins * 100) if total_coins > 0 else 0
        print(f"  - {knd_name:7s}: {count:3d} monete ({pct:5.1f}%)")

    return updated_annotated, labeled_rois


def plot_classified_coins_sample(
    updated_annotated: Dict[str, Any],
    labeled_rois: Dict[str, np.ndarray],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (16, 8)
):
    """
    Mostra a schermo una griglia di campioni di ROI mascherate con etichette e confidence,
    cercando di includere esempi di tutte e 4 le tipologie.
    
    Args:
        updated_annotated: Data structure annotata e classificata.
        labeled_rois: Dizionario {f"{image_name}_{coin_id}": roi_bgr}.
        n_samples: Numero di monete da mostrare (default: 8).
        seed: Seed opzionale.
        figsize: Dimensioni figura matplotlib.
    """
    if not labeled_rois:
        print("Nessuna ROI classificata disponibile per la visualizzazione.")
        return

    if seed is not None:
        random.seed(seed)

    # Raggruppa le chiavi per tipologia ('kind') per garantire un campione eterogeneo
    by_kind: Dict[str, List[str]] = {"1 euro": [], "2 euro": [], "gold": [], "copper": []}
    for img_name, entry in updated_annotated.items():
        for coin in entry.get("coins", []):
            cid = coin.get("coin_id", 0)
            key = f"{img_name}_{cid}"
            knd = coin.get("kind", coin.get("class", "gold"))
            if key in labeled_rois and knd in by_kind:
                by_kind[knd].append(key)

    # Seleziona monete bilanciando le tipologie
    selected_keys: List[str] = []
    for knd, keys in by_kind.items():
        if keys:
            k_sample = min(2, len(keys))
            selected_keys.extend(random.sample(keys, k_sample))

    # Completa fino a n_samples se necessario
    remaining = [k for k in labeled_rois.keys() if k not in selected_keys]
    if len(selected_keys) < n_samples and remaining:
        selected_keys.extend(random.sample(remaining, min(n_samples - len(selected_keys), len(remaining))))

    selected_keys = selected_keys[:n_samples]

    ncols = 4
    nrows = int(np.ceil(len(selected_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, key in enumerate(selected_keys):
        ax = axes[idx]
        roi_bgr = labeled_rois[key]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(roi_rgb)
        ax.set_title(f"Moneta: {key}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(len(selected_keys), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Modulo 5: Campione ROI Monete con Tipologia (Kind) e Confidence ({len(selected_keys)} monete)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys

    # Caricamento annotated_images da Modulo 3
    ann_path = get_default_annotated_images_path()
    if not ann_path.exists():
        print(f"Errore: {ann_path} non trovato. Esegui prima il Modulo 3.")
        sys.exit(1)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotated_images = json.load(f)

    # Caricamento immagini denoised da Modulo 4
    denoised_dir = Path(__file__).resolve().parents[2] / "artifacts" / "Flow_B_Target" / "module4_denoise_colored"
    denoised_images = {}
    if denoised_dir.exists():
        for p in denoised_dir.glob("*.png"):
            denoised_images[p.stem] = cv2.imread(str(p))

    classified_annotated, labeled_rois = classify_all_coins(
        denoised_images=denoised_images,
        annotated_images=annotated_images,
        save_artifacts=True
    )
