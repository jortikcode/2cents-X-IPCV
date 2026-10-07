"""Copper vs gold classification of single-metal coins from colour.

1c, 2c and 5c are copper-plated steel (red-orange); 10c, 20c and 50c are Nordic
gold (yellow). The two metals differ in hue, so the coin's average hue separates
them. The images must already be white balanced, so the colour of the light
does not shift the hue.
"""

from pathlib import Path

import numpy as np
import cv2

from preprocessing_detection import preprocess_gray
from detection_circles import detect_circles, filter_by_edge_support

COIN_MAX = 0.9  # stop at 0.9r to avoid the rim shadow

COPPER_REFERENCES = ("1cent", "2cent", "5cent")
GOLD_REFERENCES = ("10cent", "20cent", "50cent")

# Decide a coin only if it lies at least halfway between the threshold and the
# nearest reference coin. A coin's hue wobbles by about 1.4 degrees when it is
# photographed like a target image, so anything closer is a coin flip.
CONFIDENT = 0.5


def coin_hue(image: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Average hue, in degrees, of the coin's pixels in a white-balanced image.

    Around 10° for copper coins and 25° or more for gold ones.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (round(cx), round(cy)), round(COIN_MAX * r), 255, -1)

    # float input: hue in 0-360 degrees
    hsv = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][mask > 0]
    # hue is circular: 350° is a red, i.e. -10°. Coin colours all lie between
    # red and yellow, so on this scale a plain mean is a valid average.
    hue = np.where(hue > 180, hue - 360, hue)
    return float(hue.mean())


def reference_hues(reference_dir: Path) -> dict[str, float]:
    """Hue of every single-metal reference coin, by denomination.

    Expects one white-balanced coin image per denomination (1cent.jpg, 10cent.jpg, ...).
    """
    hues = {}
    for name in COPPER_REFERENCES + GOLD_REFERENCES:
        image = cv2.imread(str(Path(reference_dir) / f"{name}.jpg"))
        gray = preprocess_gray(image, use_nlm=False)
        circles = detect_circles(gray)
        kept, _, _ = filter_by_edge_support(gray, circles, min_score=0.80)
        if not circles:
            raise ValueError(f"no coin detected in reference {name}.jpg")
        coin = max(kept or circles, key=lambda c: c.radius)
        hues[name] = coin_hue(image, coin.cx, coin.cy, coin.radius)
    return hues


def threshold_from_references(reference_dir: Path) -> tuple[float, float]:
    """Threshold halfway between the most yellow copper and the least yellow gold reference.

    Returns the threshold and half the gap between those two coins, which is the
    unit `confidence` is measured in.
    """
    hues = reference_hues(reference_dir)
    most_yellow_copper = max(hues[name] for name in COPPER_REFERENCES)
    least_yellow_gold = min(hues[name] for name in GOLD_REFERENCES)
    return (most_yellow_copper + least_yellow_gold) / 2, (least_yellow_gold - most_yellow_copper) / 2


def confidence(hue: float, threshold: float, half_gap: float) -> float:
    """Distance from the threshold, in units of the distance to the nearest reference, capped at 1.

    0 means the coin is on the threshold; 1 means it is at least as clear-cut
    as the nearest reference coin was.
    """
    return min(abs(hue - threshold) / half_gap, 1.0)


def classify(hue: float, threshold: float) -> str:
    """Return "copper" or "gold"."""
    return "gold" if hue > threshold else "copper"
