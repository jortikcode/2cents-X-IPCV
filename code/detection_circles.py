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
    return [Circle(cx=float(cx), cy=float(cy), radius=float(r)) for cx, cy, r in raw[0]]


def compute_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude and angle (radians) via Sobel."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return mag, angle


def edge_support_score(
    cx: float,
    cy: float,
    radius: float,
    mag: np.ndarray,
    angle: np.ndarray,
    n_samples: int = 72,
) -> float:
    """Score in [0, 1]: alignment of circle circumference with image edges.

    Samples n_samples points on the circumference, checks that gradient
    direction is approximately radial at each point. Weighted by magnitude
    so weak-gradient points don't vote.
    """
    h, w = mag.shape
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    px = cx + radius * np.cos(thetas)
    py = cy + radius * np.sin(thetas)

    px_i = np.clip(np.round(px).astype(int), 0, w - 1)
    py_i = np.clip(np.round(py).astype(int), 0, h - 1)

    mags = mag[py_i, px_i]
    angles = angle[py_i, px_i]
    radial = np.arctan2(py - cy, px - cx)
    diff = np.abs(angles - radial) % np.pi
    alignment = np.cos(diff) ** 2  # 1 = radial, 0 = tangential

    total_mag = mags.sum()
    if total_mag < 1e-8:
        return 0.0
    return float((alignment * mags).sum() / total_mag)


def filter_by_edge_support(
    preprocessed: np.ndarray,
    circles: list[Circle],
    min_score: float = 0.85,
    n_samples: int = 72,
) -> tuple[list[Circle], list[Circle], list[float]]:
    """Filter circles by radial edge support. Return (kept, rejected, scores)."""
    if not circles:
        return [], [], []
    mag, angle = compute_gradients(preprocessed)
    scores = [
        edge_support_score(c.cx, c.cy, c.radius, mag, angle, n_samples) for c in circles
    ]
    kept = [c for c, s in zip(circles, scores) if s >= min_score]
    rejected = [c for c, s in zip(circles, scores) if s < min_score]
    return kept, rejected, scores


def circles_mask(
    image: np.ndarray,
    circles: list[Circle],
    shrink_factor: float = 0.9,
) -> np.ndarray:
    """Return a single binary mask with all circles merged (union).

    shrink_factor: multiplied by each radius before drawing (< 1 shrinks,
                   1.0 keeps original radius).
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for c in circles:
        cv2.circle(mask, (int(c.cx), int(c.cy)), int(c.radius * shrink_factor), 255, thickness=-1)
    return mask


def per_coin_masks(
    image: np.ndarray,
    circles: list[Circle],
    overlap: str = "exclude",  # "include" | "exclude" | "nearest"
    shrink_factor: float = 0.9,
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
        cv2.circle(m, (int(c.cx), int(c.cy)), int(c.radius * shrink_factor), 255, thickness=-1)
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
        return [np.where(overlap_mask, 0, m).astype(np.uint8) for m in full]

    # overlap == "nearest": assign each overlap pixel to closest center
    Y, X = np.mgrid[:h, :w]
    masks = [m.copy() for m in full]
    for y, x in zip(*np.where(overlap_mask)):
        dists = [(x - c.cx) ** 2 + (y - c.cy) ** 2 for c in circles]
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
