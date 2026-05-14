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
        return np.pi * self.radius ** 2


def detect_circles(
    preprocessed: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 30.0,
    param1: float = 50.0,    # upper Canny threshold inside HoughCircles
    param2: float = 30.0,    # accumulator threshold — lower = more detections
    min_radius: int = 10,    # coarse filter in accumulator space (int)
    max_radius: int = 500,   # coarse filter in accumulator space (int)
    min_radius_f: float = 0.0,   # precise post-filter (float, 0 = skip)
    max_radius_f: float = 0.0,   # precise post-filter (float, 0 = skip)
) -> list[Circle]:
    """Detect circles via Hough Gradient on a grayscale or edge image."""
    gray = preprocessed if preprocessed.ndim == 2 else cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
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
    circles = []
    for cx, cy, r in raw[0]:
        if min_radius_f > 0.0 and r < min_radius_f:
            continue
        if max_radius_f > 0.0 and r > max_radius_f:
            continue
        circles.append(Circle(cx=float(cx), cy=float(cy), radius=float(r)))
    return circles


def circles_mask(image: np.ndarray, circles: list[Circle]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for c in circles:
        cv2.circle(mask, (int(c.cx), int(c.cy)), int(c.radius), 255, thickness=-1)
    return mask


def draw_circles(image: np.ndarray, circles: list[Circle]) -> np.ndarray:
    out = image.copy()
    for c in circles:
        cv2.circle(out, (int(c.cx), int(c.cy)), int(c.radius), (0, 255, 0), thickness=2)
        cv2.circle(out, (int(c.cx), int(c.cy)), 3, (0, 0, 255), thickness=-1)
    return out
