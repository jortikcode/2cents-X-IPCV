"""Circle detection via Hough Transform with per-coin mask generation."""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    radius: float

    @property
    def area(self) -> float:
        return np.pi * self.radius**2


def detect_circles(
    preprocessed: np.ndarray,
    dp: float = 1.3,
    min_dist: float = 100.0,
    param1: float = 50.0,  # upper Canny threshold inside HoughCircles
    param2: float = 30.0,  # accumulator threshold — lower = more detections
    min_radius: int = 50,  # coarse filter in accumulator space (int)
    max_radius: int = 130,  # coarse filter in accumulator space (int)
) -> list[Circle]:
    """Detect circles via Hough Gradient on a grayscale or edge image."""
    gray = (
        preprocessed
        if preprocessed.ndim == 2
        else cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    )
    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if raw is None:
        return []
    # TODO: filtrare i falsi positivi (es. soglia su mean L-channel, edge support, varianza interna)
    return [Circle(cx=float(cx), cy=float(cy), radius=float(r)) for cx, cy, r in raw[0]]


def circles_mask(image: np.ndarray, circles: list[Circle]) -> np.ndarray:
    """Return a single binary mask with all circles merged (union)."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for c in circles:
        cv2.circle(mask, (int(c.cx), int(c.cy)), int(c.radius), 255, thickness=-1)
    return mask


def per_coin_masks(
    image: np.ndarray,
    circles: list[Circle],
    overlap: str = "exclude",  # "include" | "exclude" | "nearest"
) -> list[np.ndarray]:
    """Return one binary mask per circle, handling overlapping regions.

    include: overlap pixels kept in all overlapping masks
    exclude: overlap pixels removed from all masks
    nearest: overlap pixels assigned to the closest circle center (Voronoi)
    """
    h, w = image.shape[:2]

    # Build individual full masks
    full = []
    for c in circles:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m, (int(c.cx), int(c.cy)), int(c.radius), 255, thickness=-1)
        full.append(m)

    if not circles:
        return []
    # TODO: raffinare le maschere (es. active contours / GrabCut lungo il bordo del cerchio)

    if overlap == "include":
        return [m.copy() for m in full]

    # Count how many circles cover each pixel
    count = np.zeros((h, w), dtype=np.int32)
    for m in full:
        count += (m > 0).astype(np.int32)
    overlap_mask = count > 1  # True where ≥2 circles overlap

    if overlap == "exclude":
        return [
            np.where(overlap_mask, 0, m).astype(np.uint8)
            for m in full
        ]

    # overlap == "nearest": assign each overlap pixel to closest center
    Y, X = np.mgrid[:h, :w]
    masks = [m.copy() for m in full]
    for y, x in zip(*np.where(overlap_mask)):
        dists = [
            (x - c.cx) ** 2 + (y - c.cy) ** 2
            for c in circles
        ]
        winner = int(np.argmin(dists))
        for i, m in enumerate(masks):
            if m[y, x] > 0:
                m[y, x] = 255 if i == winner else 0
    return masks


def draw_circles(image: np.ndarray, circles: list[Circle]) -> np.ndarray:
    """Return a copy of image with detected circles drawn (green outline, red center)."""
    out = image.copy()
    for c in circles:
        cv2.circle(out, (int(c.cx), int(c.cy)), int(c.radius), (0, 255, 0), thickness=2)
        cv2.circle(out, (int(c.cx), int(c.cy)), 3, (0, 0, 255), thickness=-1)
    return out
