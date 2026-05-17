"""Preprocessing pipeline for coin detection: denoising, normalization, and edge extraction."""

import numpy as np
import cv2


def _percentile_stretch(
    gray: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Stretch intensity range to [0, 255] clipping at given percentiles."""
    p_low, p_high = np.percentile(gray, (low, high))
    if p_high == p_low:
        return gray
    stretched = (gray.astype(np.float32) - p_low) / (p_high - p_low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _clahe_light(
    gray: np.ndarray, clip_limit: float = 1.2, tile: int = 16
) -> np.ndarray:
    # Low clipLimit + large tileGridSize = boost detail, leave coarse structure alone.
    # Raise clipLimit or shrink tile to boost more.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    # Threshold derived from image median — adapts to overall brightness.
    median = float(np.median(gray))
    t1 = max(0.0, (1.0 - sigma) * median)
    t2 = min(255.0, (1.0 + sigma) * median)
    return cv2.Canny(gray, t1, t2)


def preprocess_gray(
    image: np.ndarray,
    median_ksize: int = 3,
    use_nlm: bool = True,
    nlm_h: float = 45.0,
    gaussian_ksize: int = 11,
    normalize: str = "clahe",  # "percentile" | "clahe" | "none"
) -> np.ndarray:
    """Denoise + normalize only — no Canny. Use as input to HoughCircles."""
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        if image.ndim == 3
        else image.copy()
    )

    if median_ksize > 1:
        gray = cv2.medianBlur(gray, median_ksize)
    if use_nlm:
        gray = cv2.fastNlMeansDenoising(
            gray, h=nlm_h, templateWindowSize=7, searchWindowSize=21
        )
    if gaussian_ksize > 1:
        ksize = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    if normalize == "percentile":
        gray = _percentile_stretch(gray)
    elif normalize == "clahe":
        gray = _clahe_light(gray)

    return gray


def preprocess(
    image: np.ndarray,
    median_ksize: int = 3,
    use_nlm: bool = True,
    nlm_h: float = 45.0,
    gaussian_ksize: int = 11,
    normalize: str = "clahe",  # "percentile" | "clahe" | "none"
    canny_sigma: float = 0.33,  # used only when canny_t1/t2 are None
    canny_t1: float | None = None,  # manual lower threshold; None = auto
    canny_t2: float | None = None,  # manual upper threshold; None = auto
    morph_open_ksize: int = 0,  # 0 = skip
    morph_close_ksize: int = 0,  # 0 = skip
) -> np.ndarray:
    """Full preprocessing pipeline: denoise + normalize + Canny edges. Returns binary edge map."""
    gray = preprocess_gray(
        image, median_ksize, use_nlm, nlm_h, gaussian_ksize, normalize
    )

    if canny_t1 is not None and canny_t2 is not None:
        edges = cv2.Canny(gray, canny_t1, canny_t2)
    else:
        edges = _auto_canny(gray, sigma=canny_sigma)

    if morph_open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize,) * 2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k)
    if morph_close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize,) * 2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    return edges
