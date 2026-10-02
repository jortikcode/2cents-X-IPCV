"""Modulo 1 (Flow B): Caricamento delle immagini target e visualizzazione di un subsample randomico."""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_default_target_dir() -> Path:
    """Restituisce il percorso della cartella coin_dataset/target_set risalendo dal modulo."""
    # __file__ è in pipeline/src/Flow_B_Target/module1_load_target.py
    # parents[2] è pipeline/, parents[3] è la root del progetto
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / "coin_dataset" / "target_set"
    if candidate.exists():
        return candidate

    # Fallback per esecuzione da cartella corrente
    cur_candidate = Path.cwd().resolve() / "coin_dataset" / "target_set"
    if cur_candidate.exists():
        return cur_candidate

    return candidate


def natural_sort_key(path: Path):
    """Chiave di ordinamento naturale (es. image_1, image_2 ... image_10, ...)."""
    match = re.search(r"\d+", path.stem)
    return int(match.group()) if match else path.stem


def load_target_images(
    target_dir: Optional[Union[str, Path]] = None,
    max_images: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Carica le immagini target da disco in un dizionario {nome_immagine: immagine_bgr}.
    
    Args:
        target_dir: Percorso opzionale della cartella target_set.
        max_images: Limite massimo di immagini da caricare (utile per test rapidi, default: None = tutte).
        
    Returns:
        Dict[str, np.ndarray]: Dizionario ordinato {nome_immagine: img_bgr}.
    """
    if target_dir is not None:
        path = Path(target_dir)
        if not path.is_absolute() and not path.exists():
            path = (Path(__file__).resolve().parent / path).resolve()
    else:
        path = get_default_target_dir()

    if not path.exists():
        raise FileNotFoundError(f"Cartella target non trovata: {path}")

    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths: List[Path] = []
    for ext in extensions:
        image_paths.extend(path.glob(ext))

    image_paths = sorted(image_paths, key=natural_sort_key)

    if not image_paths:
        raise ValueError(f"Nessuna immagine trovata in: {path}")

    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    target_images: Dict[str, np.ndarray] = {}
    for p in image_paths:
        name = p.stem
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"[Warning] Impossibile caricare l'immagine: {p}")
            continue
        target_images[name] = img_bgr

    print(f"Caricate con successo {len(target_images)} immagini target da: {path.name}/")
    return target_images


def plot_random_sample(
    images: Dict[str, np.ndarray],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (15, 8)
):
    """
    Seleziona e visualizza a schermo un subsample casuale delle immagini target caricate.
    
    Args:
        images: Dizionario {nome_immagine: immagine_bgr}.
        n_samples: Numero di immagini casuali da mostrare (default: 8).
        seed: Seed opzionale per riproducibilità.
        figsize: Dimensioni della figura matplotlib.
    """
    if not images:
        print("Nessuna immagine disponibile per la visualizzazione.")
        return

    if seed is not None:
        random.seed(seed)

    available_keys = list(images.keys())
    sample_size = min(n_samples, len(available_keys))
    sampled_keys = sorted(random.sample(available_keys, sample_size), key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k)

    ncols = 4
    nrows = int(np.ceil(sample_size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, key in enumerate(sampled_keys):
        ax = axes[idx]
        img_bgr = images[key]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        ax.imshow(img_rgb)
        ax.set_title(f"{key}\n({w}x{h} px)", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(sample_size, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Modulo 1: Subsample Randomico ({sample_size} su {len(images)} Immagini Target)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
