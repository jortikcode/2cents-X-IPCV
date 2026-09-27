"""Modulo 1: Caricamento e visualizzazione delle 8 immagini di riferimento."""

from pathlib import Path
from typing import Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_default_ref_dir() -> Path:
    """Trova la cartella coin_dataset/reference_set risalendo le cartelle."""
    # 1. Prova exploration/coin_dataset/reference_set (parents[3] è exploration/)
    candidate_exploration = Path(__file__).resolve().parents[3] / "coin_dataset" / "reference_set"
    if candidate_exploration.exists():
        return candidate_exploration
    # 2. Prova root/coin_dataset/reference_set (parents[4] è root del repo)
    candidate_root = Path(__file__).resolve().parents[4] / "coin_dataset" / "reference_set"
    if candidate_root.exists():
        return candidate_root
    return candidate_exploration


def load_reference_images(ref_dir: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """
    Carica le 8 immagini di riferimento da disco in un dizionario {nome_moneta: immagine_bgr}.
    
    Args:
        ref_dir: Cartella contenente le immagini (opzionale, default: exploration/coin_dataset/reference_set).
        
    Returns:
        Dict[str, np.ndarray]: Dizionario con le 8 immagini BGR.
    """
    if ref_dir is not None:
        path = Path(ref_dir)
        if not path.is_absolute() and not path.exists():
            path = (Path(__file__).resolve().parent / path).resolve()
    else:
        path = get_default_ref_dir()

    if not path.exists():
        raise FileNotFoundError(f"Cartella reference non trovata: {path}")

    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(path.glob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"Nessuna immagine trovata in: {path}")

    reference_images = {}
    for p in image_paths:
        name = p.stem  # es. '10cent', '1euro', etc.
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"[Warning] Impossibile caricare l'immagine: {p}")
            continue
        reference_images[name] = img_bgr

    return reference_images


def plot_reference_images(images: Dict[str, np.ndarray], figsize=(14, 7)):
    """
    Visualizza a schermo le immagini di riferimento in una griglia ordinata.
    
    Args:
        images: Dizionario {nome_moneta: immagine_bgr}.
        figsize: Dimensioni della figura matplotlib.
    """
    names = sorted(images.keys())
    n = len(names)
    if n == 0:
        print("Nessuna immagine da visualizzare.")
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, name in enumerate(names):
        ax = axes[idx]
        img_bgr = images[name]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        
        ax.imshow(img_rgb)
        ax.set_title(f"{name}\n({w}x{h} px)", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Modulo 1: Immagini di Riferimento Caricate (8 Denominazioni)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
