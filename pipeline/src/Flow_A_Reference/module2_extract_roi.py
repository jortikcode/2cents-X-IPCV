"""Modulo 2: Estrazione di ROI normalizzate (250x250 px, R=100 px), mascherate e salvate su disco."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_A_Reference/module2_rois."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module2_rois"


def detect_reference_circle(img_bgr: np.ndarray) -> Tuple[float, float, float]:
    """
    Rileva il cerchio della moneta (cx, cy, r) su un'immagine di riferimento pulita.
    
    Returns:
        Tuple[float, float, float]: (cx, cy, raggio)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=40,
        maxRadius=150
    )
    if circles is not None and len(circles[0]) > 0:
        c = max(circles[0], key=lambda x: x[2])
        return float(c[0]), float(c[1]), float(c[2])

    # Fallback al centro dell'immagine se Hough non trova cerchi
    h, w = gray.shape
    return float(w / 2), float(h / 2), float(min(w, h) * 0.45)


def extract_normalized_roi(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    canvas_size: int = 250,
    target_radius: float = 100.0
) -> np.ndarray:
    """
    Scala e centra la moneta su un canvas quadrato, applicando una maschera circolare nera all'esterno.
    
    Returns:
        np.ndarray: Immagine ROI BGR (canvas_size x canvas_size x 3) mascherata.
    """
    center = canvas_size / 2.0
    scale = target_radius / radius

    # Matrice di trasformazione affine: traslazione al centro del canvas e ridimensionamento
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

    # Maschera circolare: tutto ciò che è oltre target_radius diventa nero (0, 0, 0)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.circle(mask, (int(center), int(center)), int(target_radius), 255, -1)
    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

    return roi_masked


def extract_and_save_rois(
    images: Dict[str, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None,
    canvas_size: int = 250,
    target_radius: float = 100.0
) -> Dict[str, np.ndarray]:
    """
    Estrae le ROI normalizzate e mascherate per tutte le immagini di riferimento,
    salvandole nella cartella specificata e restituendole in memoria.
    
    Args:
        images: Dizionario {nome: immagine_bgr} restituito dal Modulo 1.
        output_dir: Cartella di destinazione per il salvataggio dei file (default: artifacts/.../module2_rois).
        canvas_size: Dimensione del canvas quadrato in pixel (default: 250).
        target_radius: Raggio della moneta normalizzata in pixel (default: 100.0).
        
    Returns:
        Dict[str, np.ndarray]: Dizionario con le 8 ROI BGR mascherate {nome: roi_bgr}.
    """
    out_path = Path(output_dir) if output_dir is not None else get_default_artifacts_dir()
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    rois = {}
    for name, img_bgr in images.items():
        cx, cy, r = detect_reference_circle(img_bgr)
        roi = extract_normalized_roi(
            img_bgr, cx, cy, r,
            canvas_size=canvas_size,
            target_radius=target_radius
        )
        rois[name] = roi

        # Serializzazione su file
        roi_filepath = out_path / f"{name}_roi.png"
        cv2.imwrite(str(roi_filepath), roi)

    print(f"Salvate con successo {len(rois)} ROI in: {out_path}")
    return rois


def plot_rois(rois: Dict[str, np.ndarray], figsize=(14, 7)):
    """
    Visualizza a schermo la griglia delle 8 ROI normalizzate e mascherate a colori.
    """
    names = sorted(rois.keys())
    n = len(names)
    if n == 0:
        print("Nessuna ROI da visualizzare.")
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, name in enumerate(names):
        ax = axes[idx]
        roi_bgr = rois[name]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        h, w = roi_bgr.shape[:2]

        ax.imshow(roi_rgb)
        ax.set_title(f"{name} (ROI)\n({w}x{h} px, R=100px)", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Modulo 2: ROI Normalizzate e Mascherate (250x250 px, Maschera Nera Esterna)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
