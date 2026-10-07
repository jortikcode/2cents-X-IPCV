"""Modulo 4 (Flow B): Denoising a Colori (fastNlMeansDenoisingColored + Median Filter).

Questo modulo:
1. Applica la pipeline di denoising a colori concordata a ogni immagine target:
   - Inversione iniziale dei canali cromatici (R e B) per operare nello spazio colore corretto.
   - fastNlMeansDenoisingColored (parametri da config: nlm_h=12.0, nlm_h_color=20.0).
   - Median Filter 3x3 per rimuovere residui ad alta frequenza e rumore impulsivo.
2. Salva le configurazioni in:
   pipeline/conf/Flow_B_Target/denoise_colored_params.json
3. Salva i risultati intermedi (immagini denoised) e i metadati negli artifacts in:
   pipeline/artifacts/Flow_B_Target/module4_denoise_colored/
4. Rende le immagini denoised direttamente riconducibili alle 'annotated_images' del Modulo 3
   tramite la chiave primaria/foreign key 'image_name' (es. 'image_1'), preparando il dataset
   per l'estrazione delle ROI delle singole monete nel Modulo 5.
5. Restituisce in memoria il dizionario {image_name: immagine_denoised_rgb}.
6. Visualizza a video un subsample casuale di immagini (Originale vs Denoised Colored).
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
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module4_denoise_colored."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module4_denoise_colored"


def get_default_denoise_colored_params() -> Dict[str, Any]:
    """Parametri di default per il denoising a colori concordati per il Modulo 4."""
    return {
        "invert_channels": True,
        "nlm_h": 12.0,
        "nlm_h_color": 20.0,
        "template_window_size": 7,
        "search_window_size": 21,
        "median_ksize": 3
    }


def load_or_create_denoise_colored_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di denoising dal file JSON se esiste, altrimenti salva i default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "denoise_colored_params.json"
    else:
        json_path = Path(json_path)

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"Parametri Modulo 4 caricati da: {json_path.name}")
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path}: {e}. Carico parametri di default.")

    params = get_default_denoise_colored_params()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Creato file di configurazione con parametri default: {json_path.name}")
    return params


def save_denoise_colored_params(
    params: Dict[str, Any],
    json_path: Optional[Union[str, Path]] = None
) -> Path:
    """Salva i parametri su file JSON."""
    if json_path is None:
        json_path = get_default_conf_dir() / "denoise_colored_params.json"
    else:
        json_path = Path(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Parametri salvati con successo in: {json_path.name}")
    return json_path


def denoise_single_image_colored(
    img: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Applica la pipeline di denoising a una singola immagine:
    1. Inversione canali cromatici (R/B) per preparare il formato corretto.
    2. fastNlMeansDenoisingColored (spazio CIELAB con nlm_h e nlm_h_color).
    3. Median Filter (median_ksize) per rimuovere residui ad alta frequenza.
    
    Args:
        img: Immagine target in memoria (shape HxWx3).
        params: Parametri di denoising.
        
    Returns:
        np.ndarray: Immagine denoised a colori in spazio RGB (colori naturali).
    """
    invert_channels = bool(params.get("invert_channels", True))
    nlm_h = float(params.get("nlm_h", 12.0))
    nlm_h_color = float(params.get("nlm_h_color", 20.0))
    template_ws = int(params.get("template_window_size", 7))
    search_ws = int(params.get("search_window_size", 21))
    median_k = int(params.get("median_ksize", 3))

    # 1. Inversione canali: l'array caricato da cv2.imread ha canali cromatici invertiti
    # Per passare a fastNlMeansDenoisingColored (che in OpenCV richiede BGR nativo), convertiamo opportunamente
    if invert_channels:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img.copy()

    # 2. Fast Non-Local Means a colori (CIELAB)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=nlm_h,
        hColor=nlm_h_color,
        templateWindowSize=template_ws,
        searchWindowSize=search_ws
    )

    # 3. Median Filter 3x3 post-processing
    if median_k > 1:
        denoised_bgr = cv2.medianBlur(denoised_bgr, median_k)

    # Restituisce l'immagine denoised in formato BGR standard (consistente sia in memoria che su disco)
    return denoised_bgr


def median_target_images(
    target_images: Dict[str, np.ndarray],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, np.ndarray]:
    """
    Denoising leggero per la classificazione cromatica (Modulo 5): inversione dei canali R/B e
    solo Median Filter, senza NLM. Il Median Filter rimuove i pixel rumorosi isolati senza spostare
    il colore medio della moneta, mentre l'NLM a colori lo altera.
    Usa 'invert_channels' e 'median_ksize' di denoise_colored_params.json.

    Returns:
        Dict[str, np.ndarray]: Dizionario in memoria {image_name: immagine_bgr}.
    """
    if params is None:
        params = load_or_create_denoise_colored_params()
    median_k = int(params.get("median_ksize", 3))

    median_images: Dict[str, np.ndarray] = {}
    for img_name, img in target_images.items():
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if params.get("invert_channels", True) else img
        median_images[img_name] = cv2.medianBlur(img_bgr, median_k) if median_k > 1 else img_bgr
    return median_images


def white_patch(image: np.ndarray) -> np.ndarray:
    """
    Bilanciamento del bianco white-patch: i pixel piu' luminosi (99-esimo percentile di ogni canale)
    diventano bianchi, cosi' il colore della luce non sposta la tonalita' (Hue) delle monete.
    Usato sulle target (white_patch_images) e sulle reference (Flusso A, Modulo 3).
    """
    brightest = np.percentile(image, 99, axis=(0, 1))
    return np.clip(image / brightest * 255, 0, 255).astype(np.uint8)


def white_patch_images(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Applica il white-patch a ogni immagine BGR (es. l'output di median_target_images).
    Serve al Modulo 5 per la misura della tonalita' (rame vs oro).

    Returns:
        Dict[str, np.ndarray]: Dizionario in memoria {image_name: immagine_bgr_bilanciata}.
    """
    return {img_name: white_patch(img) for img_name, img in images.items()}


def denoise_target_images_colored(
    target_images: Dict[str, np.ndarray],
    params: Optional[Dict[str, Any]] = None,
    conf_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Dict[str, np.ndarray]:
    """
    Esegue il denoising a colori su tutte le immagini target fornite.
    
    Salva negli artifacts:
    - Ciascuna immagine denoised in formato PNG ({image_name}.png)
    - Il file di metadati 'denoised_images_metadata.json' con la foreign key
      verso 'annotated_images.json'
      
    Args:
        target_images: Dizionario {image_name: immagine_originale}.
        params: Parametri opzionali (se None vengono caricati da conf).
        conf_path: Percorso file JSON parametri.
        output_dir: Cartella artifacts di output.
        save_artifacts: Se True salva le immagini e i metadati su disco.
        
    Returns:
        Dict[str, np.ndarray]: Dizionario in memoria {image_name: immagine_denoised_rgb}.
    """
    if params is None:
        params = load_or_create_denoise_colored_params(conf_path)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = get_default_artifacts_dir()

    if save_artifacts:
        out_path.mkdir(parents=True, exist_ok=True)

    denoised_images: Dict[str, np.ndarray] = {}
    metadata_records: Dict[str, Dict[str, Any]] = {}

    total_images = len(target_images)
    print(f"Avvio denoising a colori su {total_images} immagini target...")
    print(f"Parametri applicati: NLM_h={params.get('nlm_h')}, NLM_hColor={params.get('nlm_h_color')}, Median={params.get('median_ksize')}x{params.get('median_ksize')}")

    for idx, (img_name, raw_img) in enumerate(target_images.items(), start=1):
        denoised_bgr = denoise_single_image_colored(raw_img, params)
        denoised_images[img_name] = denoised_bgr

        h, w = denoised_bgr.shape[:2]
        png_filename = f"{img_name}.png"
        artifact_rel_path = f"pipeline/artifacts/Flow_B_Target/module4_denoise_colored/{png_filename}"

        if save_artifacts:
            # Salvataggio lossless PNG in formato BGR standard
            cv2.imwrite(str(out_path / png_filename), denoised_bgr)

        # Record metadati con 'foreign_key' identica alla chiave di 'annotated_images.json'
        metadata_records[img_name] = {
            "image_name": img_name,
            "foreign_key": img_name,
            "relative_image_path": f"coin_dataset/target_set/{img_name}.jpg",
            "artifact_path": artifact_rel_path,
            "width": w,
            "height": h,
            "color_space": "BGR",
            "denoise_params": {
                "nlm_h": params.get("nlm_h"),
                "nlm_h_color": params.get("nlm_h_color"),
                "median_ksize": params.get("median_ksize")
            }
        }

        if idx % 20 == 0 or idx == total_images:
            print(f"  [{idx}/{total_images}] Elaborate {idx} immagini su {total_images}...")

    # Salvataggio del file manifest JSON per il Modulo 5
    if save_artifacts:
        meta_file = out_path / "denoised_images_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2)
        print(f"Denoising completato con successo!")
        print(f"  - Immagini salvate in: {out_path}")
        print(f"  - Manifest metadati con foreign key salvato in: {meta_file.name}")

    return denoised_images


def plot_denoise_colored_sample(
    target_images: Dict[str, np.ndarray],
    denoised_images: Dict[str, np.ndarray],
    n_samples: int = 4,
    seed: Optional[int] = None,
    figsize: tuple = (16, 9)
):
    """
    Visualizza a schermo un subsample casuale affiancando l'immagine originale e la versione denoised a colori.
    
    Args:
        target_images: Dizionario immagini originali.
        denoised_images: Dizionario immagini denoised.
        n_samples: Numero di immagini campione da visualizzare (default: 4).
        seed: Seed opzionale per riproducibilità.
        figsize: Dimensioni della figura matplotlib.
    """
    if not denoised_images:
        print("Nessuna immagine denoised disponibile da visualizzare.")
        return

    if seed is not None:
        random.seed(seed)

    common_keys = [k for k in target_images.keys() if k in denoised_images]
    sample_size = min(n_samples, len(common_keys))
    sampled_keys = sorted(
        random.sample(common_keys, sample_size),
        key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k
    )

    fig, axes = plt.subplots(sample_size, 2, figsize=figsize)
    if sample_size == 1:
        axes = np.array([axes])

    for i, key in enumerate(sampled_keys):
        orig_img = target_images[key]
        denoised_img = denoised_images[key]

        # Colonna 1: Originale
        ax_orig = axes[i, 0]
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f"Originale: {key}", fontsize=11, fontweight="bold")
        ax_orig.axis("off")

        # Colonna 2: Denoised a Colori (NLM + Median)
        ax_den = axes[i, 1]
        ax_den.imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
        ax_den.set_title(f"Denoised a Colori (NLM h=12, hColor=20 + Med 3x3): {key}", fontsize=11, fontweight="bold")
        ax_den.axis("off")

    plt.suptitle(
        f"Modulo 4: Confronto Denoising a Colori ({sample_size} Immagini Target Campione)",
        fontsize=14,
        y=0.98
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)
