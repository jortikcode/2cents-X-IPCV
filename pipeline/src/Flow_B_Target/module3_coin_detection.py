"""Modulo 3 (Flow B): Detection delle monete tramite Hough Circles e generazione delle annotated_image.

Questo modulo:
1. Prende in input le immagini pre-elaborate dal Modulo 2 (denoised_target_images)
   e le immagini originali BGR dal Modulo 1 (target_images).
2. Esegue Hough Circles per identificare i cerchi candidati di ogni moneta.
3. Valuta il supporto radiale dei gradienti (edge support) per abbattere eventuali falsi positivi.
4. Costruisce per ogni immagine la data structure 'annotated_image':
   {
       "image_name": str,
       "relative_image_path": str,
       "width": int,
       "height": int,
       "n_coins": int,
       "coins": [
           {
               "coin_id": int,
               "center": [cx, cy],
               "radius": float,
               "edge_score": float
           }, ...
       ]
   }
5. Genera e salva negli artifacts le immagini annotate con i cerchi sovraimpressi ({image_name}.png)
   e il file aggregato 'annotated_images.json'.
6. Salva i parametri di configurazione in pipeline/conf/Flow_B_Target/coin_detection_params.json.
7. Mostra a video un subsample casuale di immagini annotate.
"""

import json
import random
import re
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
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module3_coin_detection."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module3_coin_detection"


def get_default_detection_params() -> Dict[str, Any]:
    """Parametri di default calibrati per Hough Circles e Edge Support Filter."""
    return {
        "dp": 1.3,
        "min_dist": 100.0,
        "param1": 50.0,
        "param2": 30.0,
        "min_radius": 50,
        "max_radius": 130,
        "use_edge_support_filter": True,
        "min_edge_score": 0.85,
        "n_samples_edge_support": 72
    }


def load_or_create_detection_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di detection dal file JSON se esiste, altrimenti crea il default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "coin_detection_params.json"
    else:
        json_path = Path(json_path)

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path} ({e}). Caricamento default.")

    params = get_default_detection_params()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return params


def save_detection_params(params: Dict[str, Any], json_path: Optional[Union[str, Path]] = None) -> Path:
    """Serializza i parametri di detection nel file JSON."""
    if json_path is None:
        json_path = get_default_conf_dir() / "coin_detection_params.json"
    else:
        json_path = Path(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return json_path


def compute_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcola magnitudo e orientazione dei gradienti tramite operatore Sobel."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return mag, angle


def edge_support_score(
    cx: float,
    cy: float,
    radius: float,
    mag: np.ndarray,
    angle: np.ndarray,
    n_samples: int = 72
) -> float:
    """
    Calcola l'allineamento dei gradienti lungo la circonferenza del cerchio.
    I valori vicini a 1.0 indicano che il gradiente è prettamente radiale (bordo circolare effettivo).
    """
    h, w = mag.shape
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    px = cx + radius * np.cos(thetas)
    py = cy + radius * np.sin(thetas)

    px_i = np.clip(np.round(px).astype(int), 0, w - 1)
    py_i = np.clip(np.round(py).astype(int), 0, h - 1)

    mags = mag[py_i, px_i]
    angles = angle[py_i, px_i]
    radial = np.arctan2(py - cy, px - cx)
    diff = np.abs(angles - radial) % np.pi
    alignment = np.cos(diff) ** 2

    total_mag = mags.sum()
    if total_mag < 1e-8:
        return 0.0
    return float((alignment * mags).sum() / total_mag)


def detect_coins_single_image(
    denoised_gray: np.ndarray,
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Esegue Hough Circles e l'eventuale filtro edge support su una singola immagine.
    
    Returns:
        List[Dict[str, Any]]: Lista di monete identificate, ordinate per raggio decrescente.
    """
    dp = float(params.get("dp", 1.3))
    min_dist = float(params.get("min_dist", 100.0))
    param1 = float(params.get("param1", 50.0))
    param2 = float(params.get("param2", 30.0))
    min_r = int(params.get("min_radius", 50))
    max_r = int(params.get("max_radius", 130))

    raw_circles = cv2.HoughCircles(
        denoised_gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r
    )

    if raw_circles is None:
        return []

    circles = [(float(cx), float(cy), float(r)) for cx, cy, r in raw_circles[0]]

    # Filtro di supporto dei gradienti radiali (opzionale)
    use_filter = params.get("use_edge_support_filter", True)
    min_score = float(params.get("min_edge_score", 0.85))
    n_samples = int(params.get("n_samples_edge_support", 72))

    if use_filter and circles:
        mag, angle = compute_gradients(denoised_gray)
        filtered_coins = []
        for cx, cy, r in circles:
            score = edge_support_score(cx, cy, r, mag, angle, n_samples=n_samples)
            if score >= min_score:
                filtered_coins.append((cx, cy, r, score))
    else:
        filtered_coins = [(cx, cy, r, 1.0) for cx, cy, r in circles]

    # Ordinamento deterministico (da sinistra a destra o per raggio)
    filtered_coins.sort(key=lambda item: (item[1], item[0]))  # ordinato per posizione (y, x)

    coins_data = []
    for idx, (cx, cy, r, sc) in enumerate(filtered_coins):
        coins_data.append({
            "coin_id": idx,
            "center": [round(cx, 1), round(cy, 1)],
            "radius": round(r, 1),
            "edge_score": round(sc, 3)
        })

    return coins_data


def draw_coin_annotations(img_bgr: np.ndarray, coins: List[Dict[str, Any]]) -> np.ndarray:
    """
    Disegna cerchi e label identificativi sopra l'immagine BGR.
    """
    annotated = img_bgr.copy()
    for coin in coins:
        cx, cy = int(coin["center"][0]), int(coin["center"][1])
        r = int(coin["radius"])
        coin_id = coin["coin_id"]

        # Circonferenza verde
        cv2.circle(annotated, (cx, cy), r, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        # Punto centrale rosso
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), thickness=-1)

        # Label con ID e raggio
        label = f"#{coin_id} (r={r}px)"
        cv2.putText(
            annotated,
            label,
            (cx - r, cy - r - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            label,
            (cx - r, cy - r - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return annotated


def detect_coins_in_target_images(
    target_images: Dict[str, np.ndarray],
    denoised_images: Dict[str, np.ndarray],
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Esegue la detection su tutte le immagini target, costruisce la struttura dati 'annotated_image'
    e salva su file sia le immagini con cerchi disegnati che il manifesto JSON.
    
    Args:
        target_images: Immagini originali BGR dal Modulo 1.
        denoised_images: Immagini grayscale pre-elaborate dal Modulo 2.
        params: Parametri di configurazione (se None, caricati da conf).
        output_dir: Cartella artifacts di destinazione.
        save_artifacts: Se True, serializza su disco.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dizionario {nome_immagine: annotated_image_data}.
    """
    if params is None:
        params = load_or_create_detection_params()

    out_path = Path(output_dir) if output_dir is not None else get_default_artifacts_dir()
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()

    if save_artifacts:
        out_path.mkdir(parents=True, exist_ok=True)

    annotated_images_dict: Dict[str, Dict[str, Any]] = {}
    total_coins_detected = 0

    print(f"Avvio Hough Circles detection su {len(denoised_images)} immagini...")

    for name, denoised_gray in denoised_images.items():
        orig_bgr = target_images.get(name)
        if orig_bgr is None:
            continue

        h, w = orig_bgr.shape[:2]

        # 1. Rileva i cerchi
        coins = detect_coins_single_image(denoised_gray, params)
        total_coins_detected += len(coins)

        # 2. Crea la data structure 'annotated_image'
        annotated_image_data = {
            "image_name": name,
            "relative_image_path": f"coin_dataset/target_set/{name}.jpg",
            "width": int(w),
            "height": int(h),
            "n_coins": len(coins),
            "coins": coins
        }
        annotated_images_dict[name] = annotated_image_data

        # 3. Disegna e salva l'immagine annotata negli artifacts
        if save_artifacts:
            annotated_bgr = draw_coin_annotations(orig_bgr, coins)
            cv2.imwrite(str(out_path / f"{name}.png"), annotated_bgr)

    # 4. Serializzazione globale del JSON degli annotated_images
    if save_artifacts:
        json_file = out_path / "annotated_images.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(annotated_images_dict, f, indent=2)

        print(f"Detection completata!")
        print(f"  - Monete totali identificate: {total_coins_detected}")
        print(f"  - Data structure salvata in: {json_file.name}")
        print(f"  - Immagini annotate salvate in: {out_path}")

    return annotated_images_dict


def plot_annotated_sample(
    annotated_dict: Dict[str, Dict[str, Any]],
    target_images: Dict[str, np.ndarray],
    n_samples: int = 4,
    seed: Optional[int] = 42,
    figsize: tuple = (16, 12),
    save_plot: bool = True
):
    """
    Mostra a schermo un subsample casuale di immagini target con i cerchi rilevati sovraimpressi.
    """
    keys = sorted(
        list(annotated_dict.keys()),
        key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k
    )
    if not keys:
        print("Nessuna immagine annotata da mostrare.")
        return

    if seed is not None:
        random.seed(seed)

    sample_size = min(n_samples, len(keys))
    sampled_keys = sorted(
        random.sample(keys, sample_size),
        key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k
    )

    ncols = 2
    nrows = int(np.ceil(sample_size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, key in enumerate(sampled_keys):
        ax = axes[idx]
        data = annotated_dict[key]
        orig_bgr = target_images[key]
        annotated_bgr = draw_coin_annotations(orig_bgr, data["coins"])
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(annotated_rgb)
        ax.set_title(f"{key} - {data['n_coins']} moneta/e rilevata/e", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(sample_size, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Modulo 3: Detection Cerchi Hough su Subsample Randomico ({sample_size} Immagini)", fontsize=14, y=0.99)
    plt.tight_layout()

    if save_plot:
        artifacts_dir = get_default_artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifacts_dir / "annotated_detection_summary.png"
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"Grafico riassuntivo della detection salvato in: {summary_path}")

    plt.show()
    plt.close(fig)
