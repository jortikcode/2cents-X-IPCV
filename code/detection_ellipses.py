"""Ellipse detection via contour fitting with axis, area, circularity, and tilt filters."""

import numpy as np
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
    max_tilt_deg: float = 90.0,
) -> list[Ellipse]:
    """Fit ellipses to contours and filter by axis size, area, circularity, and tilt."""
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
        try:
            (cx, cy), (w, h), angle = fit_fn(contour)
        except cv2.error:
            continue
        # Preserve original w/h order from fitEllipse: w is along `angle`, h is perpendicular.
        # Do NOT reorder — cv2.ellipse expects axes[0] along the angle direction.
        axis1, axis2 = w / 2, h / 2
        if not (min_axis <= min(axis1, axis2) and max(axis1, axis2) <= max_axis):
            continue
        if min_area > 0.0 and np.pi * axis1 * axis2 < min_area:
            continue
        if (
            min_circularity > 0.0
            and min(axis1, axis2) / max(axis1, axis2) < min_circularity
        ):
            continue
        # fitEllipse returns angle of w-axis, but w < h is possible.
        # If w < h, the actual major axis is perpendicular → shift by 90°.
        major_angle = angle if axis1 >= axis2 else (angle + 90.0) % 180.0
        tilt = min(major_angle, 180.0 - major_angle)
        if tilt > max_tilt_deg:
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
    max_tilt_deg: float = 90.0,  # 90 = no filter; lower = stricter horizontal constraint
) -> list[Ellipse]:
    """Detect ellipses in a preprocessed (binary/edge) image."""
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return _fit_contours(
        contours,
        method,
        min_points,
        min_axis,
        max_axis,
        min_area,
        min_circularity,
        max_tilt_deg,
    )


def ellipses_mask(image: np.ndarray, ellipses: list[Ellipse]) -> np.ndarray:
    """Return a single binary mask with all ellipses merged (union)."""
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
    """Return a copy of image with detected ellipses drawn (green outline)."""
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
