"""Coin classification pipeline: extract, preprocess, color-group, and classify."""

import numpy as np
import cv2
from pathlib import Path
from typing import Literal

from preprocessing_detection import preprocess_gray

try:  # optional acceleration, not in the dependency set
    import phasepack  # noqa: F401

    _HAS_PHASEPACK = True
except ImportError:
    _HAS_PHASEPACK = False


# ---------------------------------------------------------------------------
# Stage 1 — Disc extraction
# ---------------------------------------------------------------------------


def extract_disc(
    img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    target_radius: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a circular disc centered at (cx, cy) and resize to target_radius.

    Parameters
    ----------
    img : np.ndarray
        BGR input image.
    cx, cy : float
        Circle center in pixel coordinates.
    r : float
        Circle radius in pixels.
    target_radius : int
        Half-side-length of the output square.

    Returns
    -------
    disc_bgr : np.ndarray
        Resized BGR disc, shape (2*target_radius, 2*target_radius, 3).
    mask : np.ndarray
        Boolean circular mask, same spatial shape.
    """
    size = int(2 * r)
    x0 = int(round(cx - r))
    y0 = int(round(cy - r))

    h, w = img.shape[:2]
    # Clamp to image bounds
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(w, x0 + size)
    y2 = min(h, y0 + size)

    crop = img[y1:y2, x1:x2]

    out_size = 2 * target_radius
    disc_bgr = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Boolean circular mask
    yy, xx = np.ogrid[:out_size, :out_size]
    center = out_size / 2.0
    mask = ((xx - center) ** 2 + (yy - center) ** 2) <= (target_radius**2)

    return disc_bgr, mask


# ---------------------------------------------------------------------------
# Stage 2 — Illumination preprocessing
# ---------------------------------------------------------------------------


def preprocess_disc(
    disc: np.ndarray,
    mask: np.ndarray,
    use_nlm_colored: bool = False,
    nlm_h_color: float = 10.0,
    median_ksize: int = 3,
    use_nlm: bool = True,
    nlm_h: float = 25.0,
    gaussian_ksize: int = 7,
    normalize: str = "clahe",
) -> np.ndarray:
    """Denoise and normalize the disc illumination, respecting the circular mask.

    Parameters
    ----------
    disc : np.ndarray
        BGR disc from extract_disc.
    mask : np.ndarray
        Boolean mask from extract_disc.
    use_nlm_colored : bool
        Apply color NLM before grayscale conversion.
    nlm_h_color : float
        Strength of color NLM (max 15).
    median_ksize, use_nlm, nlm_h, gaussian_ksize, normalize
        Passed to preprocess_gray.

    Returns
    -------
    gray : np.ndarray
        Preprocessed grayscale disc, background zeroed outside mask.
    """
    if use_nlm_colored:
        nlm_h_c = min(float(nlm_h_color), 15.0)
        disc = cv2.fastNlMeansDenoisingColored(
            disc, None, int(nlm_h_c), int(nlm_h_c), 7, 21
        )

    gray = preprocess_gray(
        disc,
        median_ksize=median_ksize,
        use_nlm=use_nlm,
        nlm_h=nlm_h,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
    )
    gray[~mask] = 0
    return gray


# ---------------------------------------------------------------------------
# Stage 3 — Color group classification
# ---------------------------------------------------------------------------


def color_group(
    disc: np.ndarray,
    mask: np.ndarray,
    copper_hue_range: tuple[float, float] = (5.0, 25.0),
    gold_hue_range: tuple[float, float] = (20.0, 40.0),
    bimetallic_discontinuity_thresh: float = 25.0,
) -> Literal["copper", "gold", "bimetallic"]:
    """Classify the disc into a color group based on hue analysis.

    Parameters
    ----------
    disc : np.ndarray
        BGR disc (original, before preprocessing).
    mask : np.ndarray
        Boolean circular mask.
    copper_hue_range, gold_hue_range : tuple[float, float]
        OpenCV hue ranges (0-180 scale) for each group.
    bimetallic_discontinuity_thresh : float
        Threshold on max radial hue jump to detect bimetallic coins.

    Returns
    -------
    group : Literal["copper", "gold", "bimetallic"]
    """
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    # Median hue inside mask
    masked_hue = hue[mask]
    if len(masked_hue) == 0:
        return "gold"
    median_hue = float(np.median(masked_hue))

    # Radial profile for bimetallic detection.
    # Bimetallic coins (1/2 euro) show a sharp saturation step between the
    # silver center and the gold rim; copper/gold monometallic coins stay flat.
    # Saturation is used instead of hue because it is invariant to the
    # illumination cast separating target_set from reference_set.
    # Metric: |median sat(inner third) - median sat(outer third)|; measured
    # |dsat| is <= 20 for mono coins and >= 27 for bimetallic ones.
    h, w = sat.shape
    yy, xx = np.ogrid[:h, :w]
    center_r = h / 2.0
    radial_dist = np.sqrt((xx - center_r) ** 2 + (yy - center_r) ** 2)
    max_r = h / 2.0

    n_rings = 12
    ring_edges = np.linspace(0, max_r, n_rings + 1)
    ring_sats = []
    for i in range(n_rings):
        ring_mask = (
            mask & (radial_dist >= ring_edges[i]) & (radial_dist < ring_edges[i + 1])
        )
        ring_s = sat[ring_mask]
        if len(ring_s) > 10:
            ring_sats.append(float(np.median(ring_s)))
        else:
            ring_sats.append(np.nan)

    ring_sats = np.array(ring_sats)
    valid = ~np.isnan(ring_sats)
    if valid.sum() >= 3:
        inner = np.median(ring_sats[valid][: valid.sum() // 3])
        outer = np.median(ring_sats[valid][-valid.sum() // 3 :])
        max_jump = float(abs(inner - outer))
    else:
        max_jump = 0.0

    if max_jump > bimetallic_discontinuity_thresh:
        return "bimetallic"

    # Check ranges (OpenCV hue: 0-180)
    copper_min, copper_max = copper_hue_range
    gold_min, gold_max = gold_hue_range
    if copper_min <= median_hue <= copper_max:
        return "copper"
    elif gold_min <= median_hue <= gold_max:
        return "gold"
    else:
        # Default: use proximity to ranges
        d_copper = min(abs(median_hue - copper_min), abs(median_hue - copper_max))
        d_gold = min(abs(median_hue - gold_min), abs(median_hue - gold_max))
        return "copper" if d_copper <= d_gold else "gold"


# ---------------------------------------------------------------------------
# Stage 4 — Bimetallic disambiguation
# ---------------------------------------------------------------------------


def classify_bimetallic(
    disc: np.ndarray,
    mask: np.ndarray,
    ring_inner_ratio: float = 0.70,
) -> Literal["1euro", "2euro"]:
    """Distinguish 1 euro vs 2 euro based on center/rim hue and saturation.

    Parameters
    ----------
    disc : np.ndarray
        BGR disc (original).
    mask : np.ndarray
        Boolean circular mask.
    ring_inner_ratio : float
        Radius ratio separating center from outer ring.

    Returns
    -------
    denomination : Literal["1euro", "2euro"]
    """
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    center_r = h / 2.0
    yy, xx = np.ogrid[:h, :w]
    radial_dist = np.sqrt((xx - center_r) ** 2 + (yy - center_r) ** 2)

    inner_mask = mask & (radial_dist < ring_inner_ratio * center_r)
    outer_mask = (
        mask & (radial_dist >= ring_inner_ratio * center_r) & (radial_dist < center_r)
    )

    inner_sat = float(np.median(hsv[:, :, 1][inner_mask])) if inner_mask.any() else 0.0
    outer_sat = float(np.median(hsv[:, :, 1][outer_mask])) if outer_mask.any() else 0.0

    # 1 euro: silver center (low saturation) + gold rim
    # 2 euro: gold center + silver rim (low saturation)
    if inner_sat < outer_sat:
        return "1euro"
    else:
        return "2euro"


# ---------------------------------------------------------------------------
# Stage 5 — Polar unwrap
# ---------------------------------------------------------------------------


def to_polar(
    disc: np.ndarray,
    n_theta: int = 720,
    n_r: int = 128,
) -> np.ndarray:
    """Unwrap a grayscale disc into polar coordinates.

    Parameters
    ----------
    disc : np.ndarray
        Grayscale disc (preprocessed). Square, side 2*target_radius.
    n_theta : int
        Angular resolution (columns).
    n_r : int
        Radial resolution (rows; row 0 = center).

    Returns
    -------
    polar : np.ndarray
        Shape (n_r, n_theta); row 0 = center, row n_r-1 = edge.
    """
    out_size = disc.shape[0]
    center = (out_size / 2.0, out_size / 2.0)
    max_radius = out_size / 2.0
    # cv2.warpPolar maps rho onto dsize.width and phi onto dsize.height, so we
    # request (n_r, n_theta) and transpose to get rows=r, cols=theta as documented.
    flags = cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    polar = cv2.warpPolar(
        disc,
        (n_r, n_theta),
        center,
        max_radius,
        flags,
    )
    return np.ascontiguousarray(polar.T)


# ---------------------------------------------------------------------------
# Stage 6 — Rotation alignment via 1D phase correlation
# ---------------------------------------------------------------------------


def rotation_correlation_profile(
    polar_target: np.ndarray,
    polar_ref: np.ndarray,
) -> np.ndarray:
    """Phase-correlation profile over theta shifts between two polar maps.

    Uses the full 2D structure of the polar map (not just the radial mean)
    to find the best theta alignment.  The correlation surface is computed
    via 2D FFT cross-correlation; the peak along the theta axis at r_shift=0
    gives the rotation shift.

    Parameters
    ----------
    polar_target, polar_ref : np.ndarray
        Polar maps (n_r, n_theta), uint8 or float.

    Returns
    -------
    corr_1d : np.ndarray
        Shape (n_theta,); correlation strength per theta shift (index = shift
        in theta pixels, positive = target rolled left relative to ref).
    """
    t = polar_target.astype(np.float32)
    r = polar_ref.astype(np.float32)
    # Remove per-row mean: radially-symmetric brightness profile is rotation-invariant
    # and dominates phase correlation, creating false peaks at shift=0.
    t -= t.mean(axis=1, keepdims=True)
    r -= r.mean(axis=1, keepdims=True)

    fft_t = np.fft.fft2(t)
    fft_r = np.fft.fft2(r)
    cross_power = (fft_t * np.conj(fft_r)) / (np.abs(fft_t) * np.abs(fft_r) + 1e-10)
    corr = np.fft.ifft2(cross_power).real  # shape (n_r, n_theta)

    # For rotation we only need the theta shift at r_shift = 0.
    # To be robust to small radial misalignment, take the max over the
    # first few rows (r_shift near 0).
    n_r = corr.shape[0]
    band = corr[: max(1, n_r // 8), :]  # top 12.5% of rows
    return band.max(axis=0)  # best r_shift per theta_shift


def estimate_rotation(
    polar_target: np.ndarray,
    polar_ref: np.ndarray,
) -> tuple[int, float]:
    """Estimate angular shift between two polar maps via 2D phase correlation.

    Parameters
    ----------
    polar_target, polar_ref : np.ndarray
        Polar maps (n_r, n_theta), uint8 or float.

    Returns
    -------
    shift : int
        Angular shift in theta pixels (positive = target rotated relative to ref).
    peak_height : float
        Normalized correlation peak of the phase-correlation surface.
    """
    corr_1d = rotation_correlation_profile(polar_target, polar_ref)
    shift = int(np.argmax(corr_1d))
    peak_height = float(np.clip(np.max(corr_1d), 0.0, 1.0))
    return shift, peak_height


def align_rotation(polar: np.ndarray, shift: int) -> np.ndarray:
    """Roll a polar map to undo a rotation detected by estimate_rotation.

    estimate_rotation returns the shift of the target relative to the
    reference, so the target is rolled by ``-shift`` to match the reference.

    Parameters
    ----------
    polar : np.ndarray
        Polar map (n_r, n_theta).
    shift : int
        Shift in theta pixels from estimate_rotation.

    Returns
    -------
    aligned : np.ndarray
        np.roll(polar, -shift, axis=1).
    """
    return np.roll(polar, -shift, axis=1)


# ---------------------------------------------------------------------------
# Stage 7 — Phase congruency
# ---------------------------------------------------------------------------


def _log_gabor_filterbank(
    shape: tuple[int, int],
    nscale: int,
    norient: int,
    min_wavelength: int = 3,
    sigma_onf: float = 0.55,
    k: float = 2.0,
) -> list[list[np.ndarray]]:
    """Build a Log-Gabor filter bank in the frequency domain.

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols) of the input image.
    nscale, norient : int
        Number of scales and orientations.
    min_wavelength, sigma_onf, k : float
        Log-Gabor parameters.

    Returns
    -------
    banks : list[list[np.ndarray]]
        banks[scale][orientation] = complex filter, same shape as input.
    """
    rows, cols = shape
    rows_half, cols_half = rows // 2, cols // 2

    # Frequency-domain coordinates (normalized); DC sits at (rows_half, cols_half)
    u = (np.arange(cols) - cols_half) / cols_half
    v = (np.arange(rows) - rows_half) / rows_half
    v, u = np.meshgrid(v, u, indexing="ij")
    radius = np.sqrt(u**2 + v**2)
    radius[rows_half, cols_half] = 1.0  # avoid log(0) at DC
    angle = np.arctan2(v, u)

    # Log-Gabor radial filter
    fo = 1.0 / (min_wavelength * (k ** np.arange(nscale)))
    scales = [np.exp(-(np.log(radius / f) ** 2) / (2 * sigma_onf**2)) for f in fo]

    # Angular filter
    orient_angles = np.arange(norient) * np.pi / norient
    angular = []
    for a in orient_angles:
        dphi = np.abs(angle - a)
        dphi = np.minimum(dphi, np.pi - dphi)  # wrap to [-pi/2, pi/2]
        angular.append(np.exp(-(dphi**2) / (2 * 0.5**2)))

    banks = []
    for s in range(nscale):
        orient_bank = []
        for o in range(norient):
            f = scales[s] * angular[o]
            f[0, 0] = 0.0  # zero DC
            orient_bank.append(f.astype(np.float32))
        banks.append(orient_bank)
    return banks


def phase_congruency(
    img: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: int = 3,
    sigma_onf: float = 0.55,
    k: float = 2.0,
) -> np.ndarray:
    """Compute phase congruency of an image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (e.g. aligned polar map).
    nscale, norient : int
        Number of scales and orientations.
    min_wavelength : int
        Minimum wavelength of the Log-Gabor filter.
    sigma_onf : float
        Standard deviation of the Log-Gabor radial filter.
    k : float
        Scaling factor between successive wavelengths.

    Returns
    -------
    pc : np.ndarray
        Phase congruency map, float32 in [0, 1], same shape as img.
    """
    if img.ndim == 3:
        img = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
        )
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # Prefer the phasepack implementation when available (spec-compliant).
    if _HAS_PHASEPACK:
        try:
            pc_phasepack, _, _ = phasepack.phasecong(
                img,
                nscale=nscale,
                norient=norient,
                minWaveLength=min_wavelength,
                sigmaOnf=sigma_onf,
                k=k,
            )
            return np.clip(pc_phasepack.astype(np.float32), 0.0, 1.0)
        except Exception:
            pass  # fall back to the internal Log-Gabor bank

    rows, cols = img.shape
    fft_img = np.fft.fft2(img)

    banks = _log_gabor_filterbank(
        (rows, cols), nscale, norient, min_wavelength, sigma_onf, k
    )

    # Local-energy model per orientation:
    #   E_o = |sum_s F_s|   (complex responses summed across scales)
    #   A_o = sum_s |F_s|
    #   PC = sum_o E_o / (sum_o A_o + eps)  in [0, 1]
    energy_total = np.zeros((rows, cols), dtype=np.float32)
    amplitude_total = np.zeros((rows, cols), dtype=np.float32)
    for o in range(norient):
        resp_sum = np.zeros((rows, cols), dtype=np.complex64)
        amp_sum = np.zeros((rows, cols), dtype=np.float32)
        for s in range(nscale):
            fft_filt = np.fft.fftshift(banks[s][o])
            resp = np.fft.ifft2(fft_img * fft_filt)
            resp_sum += resp
            amp_sum += np.abs(resp)
        energy_total += np.abs(resp_sum)
        amplitude_total += amp_sum

    pc = energy_total / (amplitude_total + 1e-10)
    pc = np.clip(pc, 0.0, 1.0)
    return pc.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 7b — Edge map (alternative rotation-alignment feature)
# ---------------------------------------------------------------------------


def edge_map(
    img: np.ndarray,
    method: Literal["canny", "sobel", "scharr"] = "canny",
    blur_ksize: int = 5,
    canny_lo: float = 50.0,
    canny_hi: float = 150.0,
) -> np.ndarray:
    """Blurred edge / gradient-magnitude map, an alternative to phase congruency.

    Cross-correlating two coins' edge maps in polar space is another way to find
    the angular offset between them (same roll-equivariance as phase_congruency:
    ``edge_map(roll(x, s, axis=1)) == roll(edge_map(x), s, axis=1)``, so it plugs
    into the same scan/`align_rotation` machinery). The theta axis (columns) is
    circular, so it is wrap-padded before filtering to avoid a seam artifact at
    the 0/2pi boundary; the radius axis (rows) is not periodic and uses ordinary
    border replication.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (e.g. a polar map). uint8 or float.
    method : Literal["canny", "sobel", "scharr"]
        "canny" -> binary edges via cv2.Canny. "sobel"/"scharr" -> normalized
        gradient magnitude in [0, 1].
    blur_ksize : int
        Odd Gaussian kernel size applied before edge extraction (1 = no blur).
    canny_lo, canny_hi : float
        Canny hysteresis thresholds (only used when method="canny").

    Returns
    -------
    edges : np.ndarray
        float32 in [0, 1], same shape as img.
    """
    gray = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)

    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    pad = max(k // 2, 2)
    gray_p = np.pad(gray, ((0, 0), (pad, pad)), mode="wrap")
    if blur_ksize > 1:
        gray_p = cv2.GaussianBlur(gray_p, (k, k), 0)

    if method == "canny":
        edges_p = cv2.Canny(gray_p, canny_lo, canny_hi).astype(np.float32) / 255.0
    else:
        op = cv2.Scharr if method == "scharr" else cv2.Sobel
        kwargs = {} if method == "scharr" else {"ksize": 3}
        gx = op(gray_p, cv2.CV_32F, 1, 0, **kwargs)
        gy = op(gray_p, cv2.CV_32F, 0, 1, **kwargs)
        mag = np.sqrt(gx**2 + gy**2)
        edges_p = mag / (mag.max() + 1e-6)

    edges = edges_p[:, pad:-pad]
    return edges.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 8 — Similarity
# ---------------------------------------------------------------------------


def radial_weight_mask(
    n_r: int,
    center_ratio: float = 0.20,
    rim_ratio: float = 0.85,
    transition_width: float = 0.10,
) -> np.ndarray:
    """Build a 1D radial weight mask emphasizing the engraving band.

    Parameters
    ----------
    n_r : int
        Number of radial rows in the polar map.
    center_ratio, rim_ratio : float
        Fraction of radius where the high-weight band starts/ends.
    transition_width : float
        Cosine transition half-width at each edge.

    Returns
    -------
    w : np.ndarray, shape (n_r,)
        Weights in [0, 1].
    """
    r = np.linspace(0.0, 1.0, n_r)
    w = np.ones(n_r, dtype=np.float32)
    # taper near center
    mask_in = r < center_ratio
    w[mask_in] = 0.5 * (
        1.0 + np.cos(np.pi * (center_ratio - r[mask_in]) / transition_width)
    )
    w[r < center_ratio - transition_width] = 0.0
    # taper near rim
    mask_out = r > rim_ratio
    w[mask_out] = 0.5 * (
        1.0 + np.cos(np.pi * (r[mask_out] - rim_ratio) / transition_width)
    )
    w[r > rim_ratio + transition_width] = 0.0
    return w


def zncc(
    map_a: np.ndarray,
    map_b: np.ndarray,
    mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> float:
    """Zero-mean normalized cross-correlation between two maps.

    Parameters
    ----------
    map_a, map_b : np.ndarray
        Same-shape arrays (e.g. phase congruency maps).
    mask : np.ndarray | None
        Optional boolean mask to restrict the computation.
    weight : np.ndarray | None
        Optional radial weight array of shape (n_r,), broadcast along theta.
        Weights the contribution of each radial row before ZNCC.

    Returns
    -------
    score : float
        ZNCC in [-1, 1].
    """
    a = map_a.astype(np.float64)
    b = map_b.astype(np.float64)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if w.ndim == 1:
            w = w[:, np.newaxis]
        a = a * w
        b = b * w

    a_m = a[mask]
    b_m = b[mask]
    if a_m.size == 0:
        return 0.0

    a_c = a_m - a_m.mean()
    b_c = b_m - b_m.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-10
    return float((a_c * b_c).sum() / denom)


# ---------------------------------------------------------------------------
# Stage 9 — Reference set builder
# ---------------------------------------------------------------------------


def build_reference_set(
    ref_images: list[np.ndarray],
    ref_circles: list[tuple[float, float, float]],
    labels: list[str],
    params: dict,
    save_path: Path | None = None,
) -> dict[str, dict]:
    """Pre-compute features for every reference coin.

    Parameters
    ----------
    ref_images : list[np.ndarray]
        BGR reference images.
    ref_circles : list[tuple[float, float, float]]
        (cx, cy, r) per reference image.
    labels : list[str]
        Ground-truth denomination per reference.
    params : dict
        Pipeline parameters (target_radius, n_theta, n_r, nscale, ...).
    save_path : Path | None
        If set, save the result as a compressed .npz archive.

    Returns
    -------
    refs : dict[str, dict]
        label -> {"disc", "mask", "color_group", "polar", "pc_map"}.
    """
    refs: dict[str, dict] = {}
    for img, circle, label in zip(ref_images, ref_circles, labels):
        disc, mask = extract_disc(img, *circle, target_radius=params["target_radius"])
        gray = preprocess_disc(
            disc,
            mask,
            **{
                k: v
                for k, v in params.items()
                if k
                in (
                    "use_nlm_colored",
                    "nlm_h_color",
                    "median_ksize",
                    "use_nlm",
                    "nlm_h",
                    "gaussian_ksize",
                    "normalize",
                )
            },
        )
        group = color_group(
            disc,
            mask,
            **{
                k: v
                for k, v in params.items()
                if k
                in (
                    "copper_hue_range",
                    "gold_hue_range",
                    "bimetallic_discontinuity_thresh",
                )
            },
        )
        polar = to_polar(gray, n_theta=params["n_theta"], n_r=params["n_r"])
        pc_map = phase_congruency(
            polar,
            nscale=params["nscale"],
            norient=params["norient"],
            min_wavelength=params["min_wavelength"],
            sigma_onf=params["sigma_onf"],
            k=params["k"],
        )
        refs[label] = {
            "disc": disc,
            "mask": mask,
            "color_group": group,
            "polar": polar,
            "pc_map": pc_map,
        }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for label, ref in refs.items():
            for key, val in ref.items():
                if isinstance(val, np.ndarray):
                    arrays[f"{label}/{key}"] = val
                else:
                    arrays[f"{label}/{key}"] = np.array([val])
        arrays["_labels"] = np.array(labels)
        arrays["_params"] = np.array([params], dtype=object)
        np.savez_compressed(save_path, **arrays)

    return refs


def load_reference_set(path: Path) -> dict[str, dict]:
    """Load a reference set saved by build_reference_set.

    Parameters
    ----------
    path : Path
        Path to the .npz archive.

    Returns
    -------
    refs : dict[str, dict]
        label -> {"disc", "mask", "color_group", "polar", "pc_map"}.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    labels = [str(l) for l in data["_labels"]]
    refs: dict[str, dict] = {}
    for label in labels:
        refs[label] = {
            "disc": data[f"{label}/disc"],
            "mask": data[f"{label}/mask"],
            "color_group": str(data[f"{label}/color_group"][0]),
            "polar": data[f"{label}/polar"],
            "pc_map": data[f"{label}/pc_map"],
        }
    return refs


# ---------------------------------------------------------------------------
# Stage 10 — Full classifier
# ---------------------------------------------------------------------------


def classify(
    target_img: np.ndarray,
    target_circle: tuple[float, float, float],
    references: dict[str, dict],
    params: dict,
    peak_threshold: float = 0.15,
) -> tuple[str, dict[str, float]]:
    """Classify a single coin disc against a pre-computed reference set.

    Parameters
    ----------
    target_img : np.ndarray
        BGR image containing the target coin.
    target_circle : tuple[float, float, float]
        (cx, cy, r) of the target coin.
    references : dict[str, dict]
        Output of build_reference_set / load_reference_set.
    params : dict
        Pipeline parameters (same keys used by build_reference_set).
    peak_threshold : float
        Minimum rotation-correlation peak to keep a candidate.

    Returns
    -------
    label : str
        Predicted denomination.
    scores : dict[str, float]
        label -> ZNCC score (only for candidates that passed the peak filter).
    """
    disc, mask = extract_disc(
        target_img, *target_circle, target_radius=params["target_radius"]
    )
    group = color_group(
        disc,
        mask,
        **{
            k: v
            for k, v in params.items()
            if k
            in ("copper_hue_range", "gold_hue_range", "bimetallic_discontinuity_thresh")
        },
    )

    gray = preprocess_disc(
        disc,
        mask,
        **{
            k: v
            for k, v in params.items()
            if k
            in (
                "use_nlm_colored",
                "nlm_h_color",
                "median_ksize",
                "use_nlm",
                "nlm_h",
                "gaussian_ksize",
                "normalize",
            )
        },
    )
    polar = to_polar(gray, n_theta=params["n_theta"], n_r=params["n_r"])

    if group == "bimetallic":
        label = classify_bimetallic(
            disc, mask, ring_inner_ratio=params["ring_inner_ratio"]
        )
        return label, {}

    # Monometallic target: candidate pool = every non-bimetallic reference.
    # The copper/gold hue split does not transfer across the illumination cast
    # separating target_set from reference_set, so structural matching decides.
    candidates = {
        k: v for k, v in references.items() if v["color_group"] != "bimetallic"
    }
    if not candidates:
        return "unknown", {}

    # Estimate rotation for each candidate, drop low peaks
    aligned_maps: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for label, ref in candidates.items():
        shift, peak = estimate_rotation(polar, ref["polar"])
        if peak >= peak_threshold:
            aligned = align_rotation(polar, shift)
            target_pc = phase_congruency(
                aligned,
                nscale=params["nscale"],
                norient=params["norient"],
                min_wavelength=params["min_wavelength"],
                sigma_onf=params["sigma_onf"],
                k=params["k"],
            )
            aligned_maps[label] = target_pc
            scores[label] = zncc(target_pc, ref["pc_map"])

    if not scores:
        return "unknown", {}

    best = max(scores, key=scores.get)
    return best, scores
