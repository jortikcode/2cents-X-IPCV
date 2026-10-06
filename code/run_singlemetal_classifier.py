"""Split the single-metal coins found by run_bimetal_classifier.py into copper and gold."""

import os
import sys
import json
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
from singlemetal_classifier import (CONFIDENT, white_patch, coin_hue,
                                    threshold_from_references, classify, confidence)

# Repo-relative; override with COIN_DATASET_DIR / COIN_OUT_DIR if needed.
ROOT = Path(__file__).resolve().parent.parent
EXPLORATION = Path(os.environ.get("COIN_OUT_DIR", ROOT / "exploration"))
DATASET_DIR = Path(os.environ.get("COIN_DATASET_DIR", EXPLORATION / "coin_dataset"))

REFERENCES = DATASET_DIR / "reference_set"
TARGETS = DATASET_DIR / "target_set_rgb"  # correct BGR order for OpenCV
BIMETAL = EXPLORATION / "bimetal_results.jsonl"  # labels from run_bimetal_classifier.py
LOG = EXPLORATION / "singlemetal_results.jsonl"

threshold, half_gap = threshold_from_references(REFERENCES)
print(f"Threshold from references: {threshold:.1f}°  (half gap {half_gap:.1f}°)\n")

counts = Counter()
with open(LOG, "w") as log:
    for line in BIMETAL.read_text().splitlines():
        entry = json.loads(line)
        image = white_patch(cv2.imread(str(TARGETS / entry["image"])))  # once per image

        coins = []
        for c in entry["coins"]:
            if c["label"] == "single-metal":  # 1€ and 2€ keep their label
                hue = coin_hue(image, c["cx"], c["cy"], c["r"])
                sure = confidence(hue, threshold, half_gap)
                c = {**c, "hue": round(hue, 1), "label": classify(hue, threshold),
                     "colour_confidence": round(sure, 2), "colour_confident": sure >= CONFIDENT}
            coins.append(c)

        log.write(json.dumps({"image": entry["image"], "coins": coins}) + "\n")
        counts.update(coin["label"] for coin in coins)
        print(f"{entry['image']}: {', '.join(coin['label'] for coin in coins)}")

unsure = sum(1 for line in LOG.read_text().splitlines() for c in json.loads(line)["coins"]
             if not c.get("bimetal_confident", True) or not c.get("colour_confident", True))
print(f"\nDone. {sum(counts.values())} coins → {LOG}")
for label in ("1euro", "2euro", "copper", "gold"):
    print(f"  {label:7s} {counts[label]}")
print(f"  {unsure} coins are below a confidence cut-off (leave them to the next step)")
