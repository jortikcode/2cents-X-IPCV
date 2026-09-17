"""Batch HoughCircles detection on the full target_set with fixed params."""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
from preprocessing_detection import preprocess_gray
from detection_circles import detect_circles, draw_circles, per_coin_masks, filter_by_edge_support

PARAMS = dict(
    normalize="clahe",
    use_nlm=True,
    nlm_h=45.0,
    gaussian_ksize=11,
    dp=1.3,
    min_dist=100,
    param1=50.0,
    param2=30.0,
    min_radius=50,
    max_radius=130,
    overlap="exclude",  # "include" | "exclude" | "nearest"
    min_score=0.85,
)

PALETTE = [
    (220, 60, 60),
    (60, 180, 60),
    (60, 60, 220),
    (220, 180, 60),
    (180, 60, 220),
    (60, 220, 220),
    (220, 120, 60),
    (120, 220, 60),
    (60, 120, 220),
    (220, 60, 180),
]

ROOT = Path(__file__).resolve().parent.parent
_dataset = ROOT / "coin_dataset" / "target_set"
if not _dataset.exists():
    _dataset = ROOT / "exploration" / "coin_dataset" / "target_set"

DATASET = Path(os.environ.get("COIN_DATASET", _dataset))
EXPLORATION = Path(os.environ.get("COIN_OUT_DIR", ROOT / "exploration"))
OUT_DIR = EXPLORATION / "batch_hough"
MASK_DIR = EXPLORATION / "batch_hough_masks"
LOG = EXPLORATION / "batch_hough_results.jsonl"
MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(DATASET.glob("*.jpg")) + sorted(DATASET.glob("*.png"))
LOG.write_text("")  # reset

for img_path in images:
    img = cv2.imread(str(img_path))
    gray = preprocess_gray(
        img,
        use_nlm=PARAMS["use_nlm"],
        nlm_h=PARAMS["nlm_h"],
        gaussian_ksize=PARAMS["gaussian_ksize"],
        normalize=PARAMS["normalize"],
    )
    circles = detect_circles(
        gray,
        dp=PARAMS["dp"],
        min_dist=PARAMS["min_dist"],
        param1=PARAMS["param1"],
        param2=PARAMS["param2"],
        min_radius=PARAMS["min_radius"],
        max_radius=PARAMS["max_radius"],
    )

    kept, rejected, scores = filter_by_edge_support(gray, circles, min_score=PARAMS["min_score"])
    kept_scores = [s for s in scores if s >= PARAMS["min_score"]]

    radii = [c.radius for c in kept]
    entry = {
        **PARAMS,
        "image": img_path.name,
        "n_circles": len(circles),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "mean_radius": round(float(np.mean(radii)), 1) if radii else 0.0,
        "std_radius": round(float(np.std(radii)), 1) if radii else 0.0,
        "circles": [
            {"cx": round(c.cx, 1), "cy": round(c.cy, 1), "r": round(c.radius, 1), "score": round(s, 3)}
            for c, s in zip(kept, kept_scores)
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    annotated = draw_circles(img, kept)
    cv2.imwrite(str(OUT_DIR / img_path.name), annotated)

    colored_mask = np.zeros_like(img)
    for i, mask in enumerate(per_coin_masks(img, kept, overlap=PARAMS["overlap"])):
        color = PALETTE[i % len(PALETTE)]
        colored_mask[mask > 0] = color
    cv2.imwrite(str(MASK_DIR / img_path.name), colored_mask)

    print(f"{img_path.name}: {len(circles)} detected → {len(kept)} kept, {len(rejected)} rejected")

print(f"\nDone. {len(images)} images → {LOG}")
