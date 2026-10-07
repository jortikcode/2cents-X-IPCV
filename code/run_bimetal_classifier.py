"""Classify every coin found by run_batch_hough.py as 1€, 2€ or single-metal."""

import os
import sys
import json
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
from bimetal_classifier import (CONFIDENT, ring_minus_centre, threshold_from_references,
                                classify, confidence)

# Repo-relative; override with COIN_DATASET_DIR / COIN_OUT_DIR if needed.
ROOT = Path(__file__).resolve().parent.parent
EXPLORATION = Path(os.environ.get("COIN_OUT_DIR", ROOT / "exploration"))
DATASET_DIR = Path(os.environ.get("COIN_DATASET_DIR", EXPLORATION / "coin_dataset"))

REFERENCES = DATASET_DIR / "reference_set"
TARGETS = DATASET_DIR / "target_set_rgb"  # correct BGR order for OpenCV
DETECTIONS = EXPLORATION / "batch_hough_results.jsonl"  # circles from run_batch_hough.py
LOG = EXPLORATION / "bimetal_results.jsonl"

threshold, half_gap = threshold_from_references(REFERENCES)
print(f"Threshold from references: {threshold:.4f}  (half gap {half_gap:.4f})\n")

counts = Counter()
with open(LOG, "w") as log:
    for line in DETECTIONS.read_text().splitlines():
        entry = json.loads(line)
        image = cv2.imread(str(TARGETS / entry["image"]))

        coins = []
        for c in entry["circles"]:
            delta = ring_minus_centre(image, c["cx"], c["cy"], c["r"])
            sure = confidence(delta, threshold, half_gap)
            coins.append({**c, "delta": round(delta, 3), "label": classify(delta, threshold),
                          "bimetal_confidence": round(sure, 2), "bimetal_confident": sure >= CONFIDENT})

        log.write(json.dumps({"image": entry["image"], "coins": coins}) + "\n")
        counts.update(coin["label"] for coin in coins)
        counts["unsure"] += sum(not coin["bimetal_confident"] for coin in coins)
        print(f"{entry['image']}: {', '.join(coin['label'] for coin in coins)}")

print(f"\nDone. {sum(counts.values()) - counts['unsure']} coins → {LOG}")
for label in ("1euro", "2euro", "single-metal"):
    print(f"  {label:13s} {counts[label]}")
print(f"  {counts['unsure']} of those are below the confidence cut-off")
