"""Modulo 6 (Flow B): Estrazione Edge e Polar Unwarp delle Monete Target.

Questo modulo implementa per il set target la stessa identica tecnica di estrazione
utilizzata per le monete di riferimento nel Flusso A (Moduli 4 e 5):
1. Per OGNI IMMAGINE e per OGNI MONETA in 'annotated_images.json':
   - Estrae la ROI centrata e normalizzata a 250x250 px con raggio target normalizzato R=100 px.
   - Applica la pipeline di edge extraction calibrata nel Flusso A:
     * Conversione in Grayscale
     * CLAHE (clipLimit=2.0, tileGridSize=(8, 8)) per esaltare i dettagli in bassorilievo
     * Bilateral Filter (d=7, sigma=75) per abbattere il rumore superficiale
     * Canny Edge Detection (t1=50, t2=140)
     * Mascheratura circolare esterna a raggio 99 px
   - Applica Gaussian Blur (5x5) per rendere continue le transizioni dei bordi
   - Esegue lo srotolamento in coordinate polari (theta, r) tramite cv2.warpPolar:
     * 360 bins angolari (1 grado per riga da 0 a 359 gradi)
     * 50 bins radiali (campionamento da r=0 a r=100 px)
     * Normalizzazione dell'energia totale (somma unitaria, float32)
   - Salva negli artifacts (artifacts/Flow_B_Target/module6_polar_unwarp/):
     * Matrice grezza NumPy lossless: '{nome_file}_{coin_id}_polar.npy' (float32, 360x50)
     * Immagine visiva per ispezione: '{nome_file}_{coin_id}_polar.png' (8-bit, 360x50)
     * Immagine mappa edge: '{nome_file}_{coin_id}_edge.png' (250x250)
2. Come output, aggiorna il manifest unificato 'annotated_images.json' (in artifacts/Flow_B_Target/annotated_images.json) in append aggiungendo a ogni moneta:
   - "polar_unwarp_path": percorso al file .npy della matrice polare
   - "polar_image_path": percorso al file .png dell'immagine polare
3. Restituisce in memoria la data structure aggiornata e il dizionario delle matrici polari.
4. Fornisce la funzione di visualizzazione a griglia per il notebook.
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
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module6_polar_unwarp."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module6_polar_unwarp"


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json unificato in artifacts/Flow_B_Target."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "annotated_images.json"


def load_or_create_polar_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di estrazione edge e polar unwarp dal file JSON o inizializza con i default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "polar_unwarp_params.json"
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
            print(f"Parametri Modulo 6 caricati da: {json_path.name}")
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path}: {e}. Uso parametri default.")

    params = {
        "roi_canvas_size": 250,
        "roi_target_radius": 100.0,
        "center": [125.0, 125.0],
        "max_radius": 100.0,
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "use_bilateral": True,
        "bilateral_d": 7,
        "bilateral_sigma": 75.0,
        "canny_t1": 50,
        "canny_t2": 140,
        "use_adaptive_canny": True,
        "adaptive_canny_low_pct": 75.0,
        "adaptive_canny_high_pct": 92.0,
        "target_edge_min": 800,
        "target_edge_max": 2800,
        "cluttered_canny_low_pct": 82.0,
        "cluttered_canny_high_pct": 95.0,
        "faint_canny_low_pct": 65.0,
        "faint_canny_high_pct": 85.0,
        "blur_ksize": [5, 5],
        "polar_dims": [50, 360],
        "normalize_energy": True
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Creato file di configurazione: {json_path.name}")
    return params


def extract_coin_edge_map(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae la mappa di bordi (Canny) dalla ROI normalizzata della moneta
    usando la stessa pipeline calibrata nel Flusso A:
    1. Warp affine centratura e riscalatura (canvas 250x250, raggio normalizzato 100 px).
    2. Conversione Grayscale.
    3. CLAHE.
    4. Bilateral Filter.
    5. Canny Edge Detector.
    6. Mascheratura circolare esterna (r > 99.0 px).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - roi_gray: ROI grayscale normalizzata (250x250).
            - edge_img: Mappa edge binaria uint8 (250x250).
    """
    canvas_size = int(params.get("roi_canvas_size", 250))
    target_radius = float(params.get("roi_target_radius", 100.0))
    center = canvas_size / 2.0
    scale = target_radius / max(radius, 1.0)

    # 1. Centratura e normalizzazione della scala
    T = np.float32([
        [scale, 0, center - cx * scale],
        [0, scale, center - cy * scale]
    ])
    roi_bgr = cv2.warpAffine(
        img_bgr,
        T,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 2. Grayscale
    if len(roi_bgr.shape) == 3:
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_bgr.copy()

    # 3. CLAHE
    img_proc = roi_gray.copy()
    if params.get("use_clahe", True):
        clip_limit = float(params.get("clahe_clip_limit", 2.0))
        tile_grid = tuple(params.get("clahe_tile_grid", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        img_proc = clahe.apply(img_proc)

    # 4. Bilateral Filter
    if params.get("use_bilateral", True):
        d = int(params.get("bilateral_d", 7))
        sigma = float(params.get("bilateral_sigma", 75.0))
        if d > 0:
            img_proc = cv2.bilateralFilter(img_proc, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # 5. Canny Edge Detection (Adattivo o Statico)
    yy, xx = np.ogrid[:canvas_size, :canvas_size]
    mask_outside = ((xx - center) ** 2 + (yy - center) ** 2) > ((target_radius - 1.0) ** 2)
    mask_inside = ~mask_outside

    use_adaptive = params.get("use_adaptive_canny", True)
    if use_adaptive:
        gx = cv2.Sobel(img_proc, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_proc, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        vals = mag[mask_inside]

        p_low = float(params.get("adaptive_canny_low_pct", 75.0))
        p_high = float(params.get("adaptive_canny_high_pct", 92.0))
        t1 = int(max(25, np.percentile(vals, p_low)))
        t2 = int(max(t1 + 25, np.percentile(vals, p_high)))
        edge_img = cv2.Canny(img_proc, t1, t2)
        edge_img[mask_outside] = 0
        cnt = int(np.count_nonzero(edge_img))

        target_max = int(params.get("target_edge_max", 2800))
        target_min = int(params.get("target_edge_min", 800))

        if cnt > target_max:
            # Sovra-segmentazione / clutter: alza le soglie per ripulire il rumore superficiale
            p_low_high = float(params.get("cluttered_canny_low_pct", 82.0))
            p_high_high = float(params.get("cluttered_canny_high_pct", 95.0))
            t1 = int(max(30, np.percentile(vals, p_low_high)))
            t2 = int(max(t1 + 30, np.percentile(vals, p_high_high)))
            edge_img = cv2.Canny(img_proc, t1, t2)
            edge_img[mask_outside] = 0
        elif cnt < target_min:
            # Sotto-segmentazione / rilievi deboli: abbassa le soglie per catturare il conio
            p_low_faint = float(params.get("faint_canny_low_pct", 65.0))
            p_high_faint = float(params.get("faint_canny_high_pct", 85.0))
            t1 = int(max(18, np.percentile(vals, p_low_faint)))
            t2 = int(max(t1 + 20, np.percentile(vals, p_high_faint)))
            edge_img = cv2.Canny(img_proc, t1, t2)
            edge_img[mask_outside] = 0
    else:
        t1 = int(params.get("canny_t1", 50))
        t2 = int(params.get("canny_t2", 140))
        edge_img = cv2.Canny(img_proc, t1, t2)
        edge_img[mask_outside] = 0

    return roi_gray, edge_img


def compute_polar_unwarp(
    edge_img: np.ndarray,
    center: Tuple[float, float] = (125.0, 125.0),
    max_radius: float = 100.0,
    dims: Tuple[int, int] = (50, 360),
    blur_ksize: Optional[Tuple[int, int]] = (5, 5),
    normalize_energy: bool = True
) -> np.ndarray:
    """
    Trasforma la mappa degli edge in una matrice di coordinate polari (r, theta) orizzontale.
    
    Args:
        edge_img: Immagine 2D grayscale degli edge (250x250 px).
        center: Centro della moneta normalizzata (cx, cy).
        max_radius: Raggio massimo di campionamento.
        dims: Dimensioni della matrice polare (rad_bins, ang_bins) o viceversa, default (50, 360).
        blur_ksize: Kernel gaussiano per ammorbidire le transizioni dei bordi.
        normalize_energy: Se True, normalizza la somma totale dell'energia a 1.0.
        
    Returns:
        np.ndarray: Matrice float32 orizzontale di forma (50, 360).
    """
    d1, d2 = dims
    if d1 < d2:
        rad_bins, ang_bins = d1, d2
    else:
        ang_bins, rad_bins = d1, d2

    img = edge_img.copy()

    # 1. Gaussian Blur per ammorbidire le transizioni binarie
    if blur_ksize is not None and blur_ksize[0] > 0 and blur_ksize[1] > 0:
        img = cv2.GaussianBlur(img, tuple(blur_ksize), 0)

    # 2. Warp polare lineare (cv2.WARP_POLAR_LINEAR)
    polar_full = cv2.warpPolar(
        img,
        (int(max_radius), int(ang_bins)),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )

    # 3. Ridimensionamento radiale a rad_bins (50 bins) con INTER_AREA
    if polar_full.shape[1] != rad_bins:
        polar_matrix = cv2.resize(polar_full, (rad_bins, ang_bins), interpolation=cv2.INTER_AREA)
    else:
        polar_matrix = polar_full

    # 4. Trasposizione per ottenere l'organizzazione orizzontale richiesta: (50 righe raggio x 360 colonne theta)
    polar_matrix = polar_matrix.T.astype(np.float32)

    # 5. Normalizzazione dell'energia totale
    if normalize_energy:
        total_energy = float(np.sum(polar_matrix))
        if total_energy > 0:
            polar_matrix /= total_energy

    return polar_matrix


def process_all_coins_polar(
    denoised_images: Dict[str, np.ndarray],
    annotated_images: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Esegue l'estrazione degli edge e il polar unwarp su tutte le monete di tutte le immagini target.
    
    1. Per ogni immagine e per ogni moneta:
       - Estrae la mappa edge Canny con CLAHE + Bilateral.
       - Calcola la matrice polar unwarp (360x50 float32, energia normalizzata).
       - Salva negli artifacts:
         * '{nome_file}_{coin_id}_polar.npy' (matrice NumPy grezza float32)
         * '{nome_file}_{coin_id}_polar.png' (immagine visiva normalizzata)
         * '{nome_file}_{coin_id}_edge.png' (mappa Canny edge)
       - Aggiunge a ciascun oggetto coin i campi:
         * 'polar_unwarp_path': percorso al file .npy
         * 'polar_image_path': percorso al file .png
    2. Aggiorna in append 'annotated_images.json' e salva 'polar_unwarp_annotated_images.json'.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
            - updated_annotated: Data structure aggiornata con i puntatori alle matrici polari.
            - polar_matrices: Dizionario {f"{image_name}_{coin_id}": polar_matrix (360x50 float32)}.
    """
    if params is None:
        params = load_or_create_polar_params()

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

    center = tuple(params.get("center", [125.0, 125.0]))
    max_radius = float(params.get("max_radius", 100.0))
    dims = tuple(params.get("polar_dims", [360, 50]))
    blur_ksize = tuple(params.get("blur_ksize", [5, 5])) if params.get("blur_ksize") else None
    normalize_energy = bool(params.get("normalize_energy", True))

    updated_annotated: Dict[str, Any] = json.loads(json.dumps(annotated_images))
    polar_matrices: Dict[str, np.ndarray] = {}

    total_coins = sum(len(entry.get("coins", [])) for entry in annotated_images.values())
    print(f"Avvio estrazione Polar Unwarp su {total_coins} monete ({len(annotated_images)} immagini)...")

    processed_count = 0

    for img_name, entry in updated_annotated.items():
        # Recupero dell'immagine denoised
        if img_name in denoised_images:
            img_bgr = denoised_images[img_name]
        else:
            # Fallback: caricamento da file artifacts di Modulo 4
            png_path = Path(__file__).resolve().parents[2] / "artifacts" / "Flow_B_Target" / "module4_denoise_colored" / f"{img_name}.png"
            if png_path.exists():
                img_bgr = cv2.imread(str(png_path))
            else:
                img_path = Path(__file__).resolve().parents[3] / entry.get("relative_image_path", "")
                img_bgr = cv2.imread(str(img_path))

        coins_list = entry.get("coins", [])
        for coin in coins_list:
            coin_id = coin.get("coin_id", 0)
            cx, cy = float(coin["center"][0]), float(coin["center"][1])
            radius = float(coin["radius"])

            # 1. Estrazione mappa edge con pipeline Canny + CLAHE + Bilateral
            roi_gray, edge_img = extract_coin_edge_map(img_bgr, cx, cy, radius, params)

            # 2. Calcolo polar unwarp
            polar_mat = compute_polar_unwarp(
                edge_img,
                center=center,
                max_radius=max_radius,
                dims=dims,
                blur_ksize=blur_ksize,
                normalize_energy=normalize_energy
            )

            coin_key = f"{img_name}_{coin_id}"
            polar_matrices[coin_key] = polar_mat

            # 3. Nomi file per gli artefatti
            npy_rel_path = f"pipeline/artifacts/Flow_B_Target/module6_polar_unwarp/{coin_key}_polar.npy"
            png_rel_path = f"pipeline/artifacts/Flow_B_Target/module6_polar_unwarp/{coin_key}_polar.png"
            edge_rel_path = f"pipeline/artifacts/Flow_B_Target/module6_polar_unwarp/{coin_key}_edge.png"

            # 4. Aggiunta campi puntatore all'oggetto coin
            coin["polar_unwarp_path"] = npy_rel_path
            coin["polar_image_path"] = png_rel_path
            coin["edge_pixels"] = int(np.count_nonzero(edge_img))

            # 5. Salvataggio su disco negli artifacts
            if save_artifacts:
                # Array numpy grezzo lossless (.npy)
                np.save(str(out_path / f"{coin_key}_polar.npy"), polar_mat)

                # Immagine visualizzabile (.png) normalizzata a [0, 255]
                polar_vis = cv2.normalize(polar_mat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(str(out_path / f"{coin_key}_polar.png"), polar_vis)

                # Mappa di bordi Canny (.png)
                cv2.imwrite(str(out_path / f"{coin_key}_edge.png"), edge_img)

            processed_count += 1

        if processed_count % 50 == 0 or processed_count == total_coins:
            print(f"  [{processed_count}/{total_coins}] Matrici polari estratte...")

    # 6. Aggiornamento in append del manifest unificato annotated_images.json
    if save_artifacts:
        ann_main_path = get_default_annotated_images_path()
        ann_main_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_main_path, "w", encoding="utf-8") as f:
            json.dump(updated_annotated, f, indent=2)
        print(f"Aggiornato manifest unificato 'annotated_images.json' in: {ann_main_path}")
        print(f"Salvati artefatti Modulo 6 in: {out_path.name}/")

    return updated_annotated, polar_matrices


def plot_polar_unwarp_sample(
    updated_annotated: Dict[str, Any],
    polar_matrices: Dict[str, np.ndarray],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (16, 7),
    show_plot: bool = True
):
    """
    Visualizza a schermo una griglia con le matrici di polar unwarp estratte in orizzontale
    (asse X: angolo theta [0..360°], asse Y: raggio r [0..50 bins]), mostrando
    anche la tipologia ('kind') e la confidenza determinata dal Modulo 5.
    
    Args:
        updated_annotated: Data structure con i metadati delle monete.
        polar_matrices: Dizionario {f"{image_name}_{coin_id}": polar_matrix (50x360)}.
        n_samples: Numero di campioni da visualizzare (default: 8).
        seed: Seed opzionale.
        figsize: Dimensioni figura matplotlib.
        show_plot: Se True invoca plt.show(), altrimenti chiude la figura.
    """
    if not polar_matrices:
        print("Nessuna matrice polare disponibile per la visualizzazione.")
        return

    if seed is not None:
        random.seed(seed)

    # Raggruppa per tipologia per garantire una visualizzazione eterogenea
    by_kind: Dict[str, List[Tuple[str, str, float]]] = {"1 euro": [], "2 euro": [], "gold": [], "copper": []}
    for img_name, entry in updated_annotated.items():
        for coin in entry.get("coins", []):
            cid = coin.get("coin_id", 0)
            key = f"{img_name}_{cid}"
            knd = coin.get("kind", "gold")
            conf = coin.get("confidence", 0.0)
            if key in polar_matrices:
                by_kind.setdefault(knd, []).append((key, knd, conf))

    selected: List[Tuple[str, str, float]] = []
    for knd, items in by_kind.items():
        if items:
            k_sample = min(2, len(items))
            selected.extend(random.sample(items, k_sample))

    all_items = [(f"{im}_{c.get('coin_id', 0)}", c.get("kind", "gold"), c.get("confidence", 0.0))
                 for im, e in updated_annotated.items() for c in e.get("coins", [])]
    remaining = [it for it in all_items if it not in selected and it[0] in polar_matrices]
    if len(selected) < n_samples and remaining:
        selected.extend(random.sample(remaining, min(n_samples - len(selected), len(remaining))))

    selected = selected[:n_samples]

    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, (key, knd, conf) in enumerate(selected):
        ax = axes[idx]
        mat = polar_matrices[key]

        # Visualizzazione della matrice polare orizzontale (Asse X: theta 0..360°, Asse Y: r 0..50)
        mat_disp = mat if mat.shape[0] == 50 else mat.T
        im = ax.imshow(mat_disp, aspect="auto", cmap="viridis", origin="lower", extent=[0, 360, 0, 50])
        pct = int(round(conf * 100)) if conf > 0 else 0
        conf_str = f" ({pct}%)" if conf > 0 else ""
        ax.set_title(f"{key}\n{knd.upper()}{conf_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Angolo $\theta$ (gradi: 0° - 360°)", fontsize=9)
        ax.set_ylabel("Raggio r (bins)", fontsize=9)

    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Modulo 6: Campione Matrici Polar Unwarp (50x360) ({len(selected)} monete)", fontsize=14, y=0.98)
    plt.tight_layout()
    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys

    # Caricamento annotated_images
    ann_path = get_default_annotated_images_path()
    if not ann_path.exists():
        print(f"Errore: {ann_path} non trovato. Esegui prima i Moduli precedenti.")
        sys.exit(1)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotated_images = json.load(f)

    # Caricamento immagini denoised se presenti
    denoised_dir = Path(__file__).resolve().parents[2] / "artifacts" / "Flow_B_Target" / "module4_denoise_colored"
    denoised_images = {}
    if denoised_dir.exists():
        for p in denoised_dir.glob("*.png"):
            denoised_images[p.stem] = cv2.imread(str(p))

    updated_ann, polar_mats = process_all_coins_polar(
        denoised_images=denoised_images,
        annotated_images=annotated_images,
        save_artifacts=True
    )
    print(f"Completata estrazione di {len(polar_mats)} matrici polari!")
