"""Modulo 2 (Flow B): Denoising delle immagini target ottimizzato per Hough Circles.

Questo modulo:
1. Applica una pipeline di denoising calibrata specificamente per la detection dei cerchi:
   - Estrazione canale di luminanza L (da spazio colore LAB) per separare luminosità da colore.
   - Median Blur (filtro per rumore sale e pepe/impulsivo).
   - Fast Non-Local Means Denoising (NLM, abbattimento tessiture di sfondo preservando i bordi).
   - Gaussian Blur (ammorbidimento finale per stabilizzare i gradienti).
   - Normalizzazione del contrasto locale tramite CLAHE.
2. Esegue il loop su tutte le immagini target caricate dal Modulo 1.
3. Salva la configurazione dei parametri in:
   pipeline/conf/Flow_B_Target/denoise_for_detection_params.json
4. Salva le immagini pre-elaborate negli artifacts in:
   pipeline/artifacts/Flow_B_Target/module2_denoise_for_detection/
5. Restituisce in memoria il dizionario {nome_immagine: immagine_denoised_gray} per il Modulo 3.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_default_conf_dir() -> Path:
    """Restituisce il percorso della cartella pipeline/conf/Flow_B_Target."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "conf" / "Flow_B_Target"


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module2_denoise_for_detection."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module2_denoise_for_detection"


def get_default_denoise_params() -> Dict[str, Any]:
    """Restituisce i parametri di default calibrati per il denoising da detection."""
    return {
        "channel": "LAB_L",
        "median_ksize": 3,
        "use_nlm": True,
        "nlm_h": 45.0,
        "template_window_size": 7,
        "search_window_size": 21,
        "gaussian_ksize": 11,
        "normalize": "clahe",
        "clahe_clip_limit": 1.2,
        "clahe_tile_grid_size": [16, 16]
    }


def load_or_create_denoise_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica i parametri di denoising dal file JSON se esiste, altrimenti salva i default."""
    if json_path is None:
        json_path = get_default_conf_dir() / "denoise_for_detection_params.json"
    else:
        json_path = Path(json_path)

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            return params
        except Exception as e:
            print(f"[Warning] Errore lettura {json_path} ({e}). Caricamento parametri di default.")

    params = get_default_denoise_params()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return params


def save_denoise_params(params: Dict[str, Any], json_path: Optional[Union[str, Path]] = None) -> Path:
    """Serializza il dizionario dei parametri nel file JSON di configurazione."""
    if json_path is None:
        json_path = get_default_conf_dir() / "denoise_for_detection_params.json"
    else:
        json_path = Path(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return json_path


def denoise_single_image(img_bgr: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Applica la pipeline di denoising ad una singola immagine BGR.
    
    Args:
        img_bgr: Immagine target BGR.
        params: Dizionario dei parametri di denoising.
        
    Returns:
        np.ndarray: Immagine in scala di grigi filtrata (uint8) pronta per Hough Circles.
    """
    channel = params.get("channel", "LAB_L")
    if img_bgr.ndim == 3:
        if channel == "LAB_L":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # 1. Median filter (rimozione rumore impulsivo)
    median_k = int(params.get("median_ksize", 3))
    if median_k > 1:
        if median_k % 2 == 0:
            median_k += 1
        gray = cv2.medianBlur(gray, median_k)

    # 2. Fast Non-Local Means (abbatte il rumore di fondo preservando i contorni netti)
    if params.get("use_nlm", True):
        nlm_h = float(params.get("nlm_h", 45.0))
        t_win = int(params.get("template_window_size", 7))
        s_win = int(params.get("search_window_size", 21))
        gray = cv2.fastNlMeansDenoising(
            gray,
            h=nlm_h,
            templateWindowSize=t_win,
            searchWindowSize=s_win
        )

    # 3. Gaussian Blur (omogeneizzazione dei gradienti)
    gauss_k = int(params.get("gaussian_ksize", 11))
    if gauss_k > 1:
        if gauss_k % 2 == 0:
            gauss_k += 1
        gray = cv2.GaussianBlur(gray, (gauss_k, gauss_k), 0)

    # 4. Normalizzazione contrasto locale (CLAHE)
    normalize_mode = params.get("normalize", "clahe")
    if normalize_mode == "clahe":
        clip = float(params.get("clahe_clip_limit", 1.2))
        tile = tuple(params.get("clahe_tile_grid_size", [16, 16]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        gray = clahe.apply(gray)

    return gray


def denoise_target_images(
    target_images: Dict[str, np.ndarray],
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Dict[str, np.ndarray]:
    """
    Esegue il loop di denoising su tutte le immagini target fornite.
    
    Args:
        target_images: Dizionario {nome_immagine: img_bgr}.
        params: Dizionario dei parametri (se None, caricati da conf).
        output_dir: Cartella per salvare gli artifacts.
        save_artifacts: Flag per abilitare/disabilitare il salvataggio su disco.
        
    Returns:
        Dict[str, np.ndarray]: Dizionario in memoria {nome_immagine: denoised_gray_img}.
    """
    if params is None:
        params = load_or_create_denoise_params()

    out_path = Path(output_dir) if output_dir is not None else get_default_artifacts_dir()
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()

    if save_artifacts:
        out_path.mkdir(parents=True, exist_ok=True)
        # Salva anche una copia dei parametri utilizzati nella cartella artifacts
        with open(out_path / "denoise_params_used.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    denoised_images: Dict[str, np.ndarray] = {}
    total = len(target_images)

    print(f"Avvio denoising su {total} immagini target...")
    for idx, (name, img_bgr) in enumerate(target_images.items(), start=1):
        denoised = denoise_single_image(img_bgr, params)
        denoised_images[name] = denoised

        if save_artifacts:
            file_path = out_path / f"{name}.png"
            cv2.imwrite(str(file_path), denoised)

        if idx % 20 == 0 or idx == total:
            print(f"  Elaborate {idx}/{total} immagini...")

    if save_artifacts:
        print(f"Denoising completato! Salvate {len(denoised_images)} immagini in: {out_path}")
    else:
        print(f"Denoising completato in memoria per {len(denoised_images)} immagini.")

    return denoised_images


def plot_denoise_comparison(
    target_images: Dict[str, np.ndarray],
    denoised_images: Dict[str, np.ndarray],
    n_samples: int = 4,
    seed: Optional[int] = 42,
    figsize: tuple = (16, 8),
    save_plot: bool = True
):
    """
    Visualizza un confronto side-by-side (Originale BGR vs Denoised Grayscale)
    per un sottocampione di immagini target.
    """
    common_keys = sorted(
        list(set(target_images.keys()) & set(denoised_images.keys())),
        key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k
    )
    if not common_keys:
        print("Nessuna immagine in comune da confrontare.")
        return

    if seed is not None:
        random.seed(seed)

    sample_size = min(n_samples, len(common_keys))
    sampled_keys = sorted(
        random.sample(common_keys, sample_size),
        key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k
    )

    fig, axes = plt.subplots(sample_size, 2, figsize=figsize)
    if sample_size == 1:
        axes = np.array([axes])

    for row_idx, key in enumerate(sampled_keys):
        orig_bgr = target_images[key]
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        denoised = denoised_images[key]

        # Colonna 1: Originale
        axes[row_idx, 0].imshow(orig_rgb)
        axes[row_idx, 0].set_title(f"Originale: {key}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Colonna 2: Denoised per Detection
        axes[row_idx, 1].imshow(denoised, cmap="gray")
        axes[row_idx, 1].set_title(f"Denoised per Detection (Hough Circles): {key}", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

    plt.suptitle("Modulo 2: Confronto Preprocessing & Denoising per Detection", fontsize=14, y=0.99)
    plt.tight_layout()

    if save_plot:
        artifacts_dir = get_default_artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        plot_path = artifacts_dir / "denoise_summary.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Grafico riassuntivo del denoising salvato in: {plot_path}")

    plt.show()
    plt.close(fig)
