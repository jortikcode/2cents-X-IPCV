"""Modulo 5: Estrazione del Polar Unwarp (Angular Projection) delle mappe di Edge.

Questo modulo:
1. Prende in input le 8 mappe di edge binarie/pesate normalizzate (250x250 px, R=100 px) del Modulo 4.
2. Applica un leggero blur gaussiano (5x5) per rendere continue le transizioni dei bordi (ispirato a angular_projection_v4).
3. Esegue lo srotolamento in coordinate polari (theta, r) producendo per ciascuna moneta
   una matrice polare (360 bins angolari x 50 bins radiali).
4. Salva negli artifacts:
   - Matrice NumPy float32 lossless (.npy per ogni moneta e archivio unificato .npz).
   - Immagini visive PNG per ispezione (.png).
   - Metadati di calcolo (.json).
   - Grafico riassuntivo con le matrici di tutte le 8 monete.
5. Restituisce in memoria il dizionario {nome_moneta: polar_matrix} pronto per i moduli successivi.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_A_Reference/module5_polar_unwarp."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module5_polar_unwarp"


def compute_polar_unwarp(
    edge_img: np.ndarray,
    center: Tuple[float, float] = (125.0, 125.0),
    max_radius: float = 100.0,
    dims: Tuple[int, int] = (50, 360),
    blur_ksize: Optional[Tuple[int, int]] = (5, 5),
    normalize_energy: bool = True
) -> np.ndarray:
    """
    Trasforma una mappa di edge (250x250) in una matrice in coordinate polari (r, theta) orizzontale.
    
    Args:
        edge_img: Immagine 2D grayscale/binaria degli edge (250x250 px).
        center: Centro della moneta normalizzata (cx, cy), default (125.0, 125.0).
        max_radius: Raggio massimo di campionamento in pixel, default 100.0.
        dims: Dimensioni della matrice polare (rad_bins, ang_bins) o viceversa, default (50, 360).
              - 50 bins radiali (righe, campionamento lungo il raggio da centro a bordo).
              - 360 bins angolari (colonne, 1 grado per colonna da 0° a 359°).
        blur_ksize: Dimensione del kernel gaussiano per ammorbidire i contorni binari, default (5, 5).
        normalize_energy: Se True, normalizza la matrice rispetto alla somma totale delle intensità.
        
    Returns:
        np.ndarray: Matrice float32 orizzontale di forma (50, 360) dove:
                    - Asse 0 (righe): Raggio r (0..rad_bins-1, da centro a bordo).
                    - Asse 1 (colonne): Angolo theta (0..359 gradi).
    """
    d1, d2 = dims
    if d1 < d2:
        rad_bins, ang_bins = d1, d2
    else:
        ang_bins, rad_bins = d1, d2

    img = edge_img.copy()

    # 1. Applicazione facoltativa del Gaussian Blur per rendere i bordi continui
    if blur_ksize is not None and blur_ksize[0] > 0 and blur_ksize[1] > 0:
        img = cv2.GaussianBlur(img, blur_ksize, 0)

    # 2. Warp polare tramite OpenCV (WARP_POLAR_LINEAR)
    # cv2.warpPolar mappa rho su width e phi su height:
    # dsize = (int(max_radius), int(ang_bins)) -> shape (ang_bins, int(max_radius))
    polar_full = cv2.warpPolar(
        img,
        (int(max_radius), int(ang_bins)),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )

    # 3. Ridimensionamento della coordinata radiale a rad_bins (50 bins) con INTER_AREA
    if polar_full.shape[1] != rad_bins:
        polar_matrix = cv2.resize(polar_full, (rad_bins, ang_bins), interpolation=cv2.INTER_AREA)
    else:
        polar_matrix = polar_full

    # 4. Trasposizione per ottenere l'organizzazione orizzontale richiesta:
    # Forma (50, 360) -> 50 righe (raggio) x 360 colonne (angolo theta)
    polar_matrix = polar_matrix.T.astype(np.float32)

    # 5. Normalizzazione facoltativa dell'energia per distribuzioni relative
    if normalize_energy:
        total_energy = float(np.sum(polar_matrix))
        if total_energy > 0:
            polar_matrix /= total_energy

    return polar_matrix


def extract_and_save_all_polar(
    edges_dict: Dict[str, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None,
    center: Tuple[float, float] = (125.0, 125.0),
    max_radius: float = 100.0,
    dims: Tuple[int, int] = (50, 360),
    blur_ksize: Optional[Tuple[int, int]] = (5, 5),
    normalize_energy: bool = True
) -> Dict[str, np.ndarray]:
    """
    Estrae le matrici polari per tutte le monete di riferimento fornite e le serializza su disco.
    
    Serializzazione:
    - Pacchetto completo: 'polar_matrices.npz' contenente tutte le 8 matrici float32 in un unico file compresso.
    - Singoli file binari: '{nome}_polar.npy' per il caricamento modulare o indipendente di ogni matrice (50x360).
    - File grafici: '{nome}_polar.png' (8-bit normalizzato, 360 px larghezza x 50 px altezza) per ispezione visiva immediata.
    - Metadati: 'polar_metadata.json' con parametri, dimensioni e statistiche energetiche.
    
    Args:
        edges_dict: Dizionario {nome_moneta: edge_img} restituito dal Modulo 4.
        output_dir: Cartella di output (default: artifacts/Flow_A_Reference/module5_polar_unwarp).
        center: Centro delle ROI normalizzate.
        max_radius: Raggio normalizzato.
        dims: Dimensioni (r, theta) della matrice polare (50, 360).
        blur_ksize: Kernel blur per i bordi.
        normalize_energy: Flag di normalizzazione dell'energia totale.
        
    Returns:
        Dict[str, np.ndarray]: Dizionario in memoria {nome_moneta: polar_matrix (50x360 float32)}.
    """
    out_path = Path(output_dir) if output_dir is not None else get_default_artifacts_dir()
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    polar_matrices: Dict[str, np.ndarray] = {}
    metadata: Dict[str, dict] = {}

    for name, edge_img in edges_dict.items():
        # Calcolo srotolamento polare
        polar_mat = compute_polar_unwarp(
            edge_img,
            center=center,
            max_radius=max_radius,
            dims=dims,
            blur_ksize=blur_ksize,
            normalize_energy=normalize_energy
        )
        polar_matrices[name] = polar_mat

        # 1. Salvataggio array NumPy singolo (.npy) lossless
        npy_file = out_path / f"{name}_polar.npy"
        np.save(str(npy_file), polar_mat)

        # 2. Salvataggio immagine PNG 8-bit per ispezione visiva
        png_file = out_path / f"{name}_polar.png"
        norm_vis = cv2.normalize(polar_mat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(png_file), norm_vis)

        metadata[name] = {
            "name": name,
            "shape": list(polar_mat.shape),
            "dtype": str(polar_mat.dtype),
            "npy_file": f"{name}_polar.npy",
            "png_file": f"{name}_polar.png",
            "min_val": float(np.min(polar_mat)),
            "max_val": float(np.max(polar_mat)),
            "mean_energy": float(np.mean(polar_mat)),
            "sum_energy": float(np.sum(polar_mat))
        }

    # 3. Salvataggio archivio aggregato compresso .npz con tutti gli 8 array
    npz_file = out_path / "polar_matrices.npz"
    np.savez_compressed(str(npz_file), **polar_matrices)

    # 4. Salvataggio metadati globali in JSON
    metadata_wrapper = {
        "module": "module5_polar_unwarp",
        "description": "Matrici di proiezione polare (Angular Projection) delle mappe di edge delle reference",
        "parameters": {
            "center": list(center),
            "max_radius": max_radius,
            "dims": list(dims),
            "blur_ksize": list(blur_ksize) if blur_ksize else None,
            "normalize_energy": normalize_energy
        },
        "coins": metadata
    }
    with open(out_path / "polar_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_wrapper, f, indent=2)

    print(f"Salvate con successo {len(polar_matrices)} matrici polari in: {out_path}")
    print(f"  - Archivio compresso NumPy: {npz_file.name}")
    print(f"  - File array singoli .npy e visualizzazioni .png per ciascuna moneta")
    print(f"  - Metadati serializzati in: polar_metadata.json")

    return polar_matrices


def load_saved_polar_matrices(
    artifacts_dir: Optional[Union[str, Path]] = None
) -> Dict[str, np.ndarray]:
    """
    Ricarica rapidamente le matrici polari salvate negli artifacts.
    Legge preferenzialmente dall'archivio compresso 'polar_matrices.npz',
    oppure carica i singoli file '{nome}_polar.npy'.
    
    Returns:
        Dict[str, np.ndarray]: Dizionario {nome_moneta: polar_matrix}.
    """
    base_dir = Path(artifacts_dir) if artifacts_dir is not None else get_default_artifacts_dir()
    npz_path = base_dir / "polar_matrices.npz"

    if npz_path.exists():
        with np.load(str(npz_path)) as data:
            return {k: data[k] for k in data.files}

    # Fallback su file .npy individuali
    matrices = {}
    for p in sorted(base_dir.glob("*_polar.npy")):
        coin_name = p.stem.replace("_polar", "")
        matrices[coin_name] = np.load(str(p))

    if not matrices:
        raise FileNotFoundError(f"Nessuna matrice polare trovata nella cartella: {base_dir}")

    return matrices


def plot_all_polar_matrices(
    polar_dict: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (16, 10),
    save_plot: bool = True,
    show_plot: bool = False
):
    """
    Visualizza e salva una griglia 2x4 con le 8 matrici polari estratte.
    L'asse orizzontale rappresenta l'angolo theta (0° - 360°),
    l'asse verticale rappresenta il raggio radiale (0 - max_r).
    """
    names = sorted(polar_dict.keys())
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        mat = polar_dict[name]
        # Se la matrice è già (50, 360), la visualizziamo direttamente:
        # Asse X: Angolo theta (0..359°), Asse Y: Raggio r (0..49 bins)
        mat_disp = mat if mat.shape[0] == 50 else mat.T
        im = ax.imshow(mat_disp, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(f"Reference: {name} (50 bins r x 360° θ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Raggio (r: 0-50)")
        if i >= 6:
            ax.set_xlabel("Angolo θ (Gradi: 0° - 360°)")
        else:
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_xticklabels([])

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Modulo 5: Matrici di Proiezione Polare (Angular Projection degli Edge)", fontsize=14, y=0.99)
    plt.tight_layout()

    if save_plot:
        artifacts_dir = get_default_artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifacts_dir / "all_polar_summary.png"
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"Grafico riassuntivo delle matrici polari salvato in: {summary_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Esecuzione standalone per rigenerare le matrici polari di riferimento in formato orizzontale (50x360)
    pipeline_dir = Path(__file__).resolve().parents[2]
    edges_dir = pipeline_dir / "artifacts" / "Flow_A_Reference" / "module4_edges"
    
    edges_dict = {}
    for p in sorted(edges_dir.glob("*_edges.png")):
        coin_name = p.stem.replace("_edges", "")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            edges_dict[coin_name] = img
            
    print(f"Caricate {len(edges_dict)} mappe di edge da: {edges_dir}")
    polar_dict = extract_and_save_all_polar(edges_dict, dims=(50, 360))
    for name, mat in polar_dict.items():
        print(f"  {name}: shape {mat.shape}, min={mat.min():.4f}, max={mat.max():.4f}")
    plot_all_polar_matrices(polar_dict, save_plot=True, show_plot=False)
