"""Bimetal vs single-metal coin classification from colour.

1€ and 2€ are the only coins made of two metals: nickel-brass (yellow) and
cupronickel (grey). Comparing the yellowness of the ring with that of the
centre separates them from single-metal coins, and the sign tells them apart:
1€ has a grey centre and a yellow ring, 2€ the opposite.
"""

from pathlib import Path

import numpy as np
import cv2

from preprocessing_detection import preprocess_gray
from detection_circles import detect_circles, filter_by_edge_support

# The two metals meet at about 0.7r, so both zones stay clear of that boundary.
CENTRE_MAX = 0.6
RING_MIN, RING_MAX = 0.8, 0.9  # stop at 0.9r to avoid the rim shadow

BIMETAL_REFERENCES = ("1euro", "2euro")

# Decide a coin only if it lies at least halfway between the threshold and the
# nearest reference coin. Closer than that, the photo alone (noise, lighting)
# could put it on either side, so the decision is left to the next step.
CONFIDENT = 0.5


def ring_minus_centre(image: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Mean yellowness (LAB b*) of the ring minus that of the centre.

    Close to 0 for single-metal coins, positive for 1€, negative for 2€.
    """
    yellow = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2].astype(np.float32)

    ys, xs = np.ogrid[: image.shape[0], : image.shape[1]]
    dist = np.hypot(xs - cx, ys - cy)
    centre = dist < CENTRE_MAX * r
    ring = (dist >= RING_MIN * r) & (dist < RING_MAX * r)

    return float(yellow[ring].mean() - yellow[centre].mean())


def reference_deltas(reference_dir: Path) -> dict[str, float]:
    """Ring-minus-centre yellowness of every reference coin, by denomination.

    Expects one coin per image, named after its denomination (1euro.jpg, 2cent.jpg, ...).
    """
    deltas = {}
    for path in sorted(Path(reference_dir).glob("*.jpg")):
        image = cv2.imread(str(path))
        gray = preprocess_gray(image, use_nlm=False)
        circles = detect_circles(gray)
        kept, _, _ = filter_by_edge_support(gray, circles, min_score=0.80)
        if not circles:
            raise ValueError(f"no coin detected in reference {path.name}")
        coin = max(kept or circles, key=lambda c: c.radius)
        deltas[path.stem] = ring_minus_centre(image, coin.cx, coin.cy, coin.radius)
    return deltas


def threshold_from_references(reference_dir: Path) -> tuple[float, float]:
    """Threshold halfway between the single-metal and the bimetal reference coins.

    Returns the threshold and half the gap between those two coins, which is the
    unit `confidence` is measured in.
    """
    deltas = reference_deltas(reference_dir)
    largest_single = max(
        abs(d) for name, d in deltas.items() if name not in BIMETAL_REFERENCES
    )
    smallest_bimetal = min(abs(deltas[name]) for name in BIMETAL_REFERENCES)
    return (largest_single + smallest_bimetal) / 2, (smallest_bimetal - largest_single) / 2


def confidence(delta: float, threshold: float, half_gap: float) -> float:
    """Distance from the threshold, in units of the distance to the nearest reference, capped at 1.

    0 means the coin is on the threshold; 1 means it is at least as clear-cut
    as the nearest reference coin was.
    """
    return min(abs(abs(delta) - threshold) / half_gap, 1.0)


def classify(delta: float, threshold: float) -> str:
    """Return "1euro", "2euro" or "single-metal"."""
    if delta > threshold:
        return "1euro"
    if delta < -threshold:
        return "2euro"
    return "single-metal"
