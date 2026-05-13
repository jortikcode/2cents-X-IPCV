import numpy as np

# import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class FitMethod(Enum):
    FIT_ELLIPSE = "fitEllipse"
    FIT_ELLIPSE_AMS = "fitEllipseAMS"
    FIT_ELLIPSE_DIRECT = "fitEllipseDirect"


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    major_axis: float
    minor_axis: float
    angle: float  # degrees


def _fit_contours(
    contours: Sequence[np.ndarray],
    method: FitMethod,
    min_points: int,
    min_axis: float,
    max_axis: float,
    min_area: float = 0.0,
    min_circularity: float = 0.0,
) -> list[Ellipse]:
    _cv2_fit = {
        "fitEllipse": cv2.fitEllipse,
        "fitEllipseAMS": cv2.fitEllipseAMS,
        "fitEllipseDirect": cv2.fitEllipseDirect,
    }
    fit_fn = _cv2_fit[method.value]
    ellipses = []
    for contour in contours:
        if len(contour) < min_points:
            continue
        if min_area > 0.0 and cv2.contourArea(contour) < min_area:
            continue
        try:
            (cx, cy), (w, h), angle = fit_fn(contour)
        except cv2.error:
            continue
        # Preserve original w/h order from fitEllipse: w is along `angle`, h is perpendicular.
        # Do NOT reorder — cv2.ellipse expects axes[0] along the angle direction.
        axis1, axis2 = w / 2, h / 2
        if not (min_axis <= min(axis1, axis2) and max(axis1, axis2) <= max_axis):
            continue
        if min_circularity > 0.0 and min(axis1, axis2) / max(axis1, axis2) < min_circularity:
            continue
        ellipses.append(
            Ellipse(cx=cx, cy=cy, major_axis=axis1, minor_axis=axis2, angle=angle)
        )
    return ellipses


def detect_ellipses(
    preprocessed: np.ndarray,
    method: FitMethod = FitMethod.FIT_ELLIPSE,
    min_points: int = 1,
    min_axis: float = 10.0,
    max_axis: float = 500.0,
    min_area: float = 0.0,
    min_circularity: float = 0.0,
) -> list[Ellipse]:
    """Detect ellipses in a preprocessed (binary/edge) image."""
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return _fit_contours(contours, method, min_points, min_axis, max_axis, min_area, min_circularity)


def ellipses_mask(image: np.ndarray, ellipses: list[Ellipse]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for e in ellipses:
        cv2.ellipse(
            mask,
            center=(int(e.cx), int(e.cy)),
            axes=(int(e.major_axis), int(e.minor_axis)),
            angle=e.angle,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,  # filled
        )
    return mask


def draw_ellipses(image: np.ndarray, ellipses: list[Ellipse]) -> np.ndarray:
    out = image.copy()
    for e in ellipses:
        cv2.ellipse(
            out,
            center=(int(e.cx), int(e.cy)),
            axes=(int(e.major_axis), int(e.minor_axis)),
            angle=e.angle,
            startAngle=0,
            endAngle=360,
            color=(0, 255, 0),
            thickness=2,
        )
    return out
