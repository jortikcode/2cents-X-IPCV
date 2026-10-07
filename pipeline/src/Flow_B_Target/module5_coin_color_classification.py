"""Modulo 5 (Flow B): Classificazione cromatica delle monete ('kind') e confidence.

Per ogni moneta rilevata dal Modulo 3 applica le stesse misure usate nel Flusso A sulle reference
(code/bimetal_classifier.py e code/singlemetal_classifier.py), con le soglie calibrate dal Modulo 3
del Flusso A (artifacts/Flow_A_Reference/module3_color/color_profiles.json):

1. Delta b* (anello - centro) -> '1 euro', '2 euro' oppure monometallica.
2. Solo per le monometalliche: Hue sull'immagine dopo white-patch (Modulo 4) -> 'copper' oppure 'gold'.

La confidence e' la distanza dalla soglia in unita' di 'half_gap': 1.0 significa che la moneta e'
netta quanto la reference piu' vicina alla soglia. Sotto il cut-off 'confident' la decisione e' incerta.

Aggiunge a ogni moneta di 'annotated_images.json' i campi 'kind', 'confidence', 'confident'
e 'color_metrics'.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
import bimetal_classifier
import singlemetal_classifier

# Etichette 'kind' attese dai Moduli 7.1 e 7.2
BIMETAL_KINDS = {"1euro": "1 euro", "2euro": "2 euro"}


def get_default_flow_a_color_profiles_path() -> Path:
    """Restituisce il percorso del file color_profiles.json estratto dal Flusso A."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module3_color" / "color_profiles.json"


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json unificato in artifacts/Flow_B_Target."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "annotated_images.json"


def load_reference_color_profiles(profiles_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica le soglie di colore calibrate dal Flusso A (Modulo 3)."""
    profiles_path = Path(profiles_path) if profiles_path is not None else get_default_flow_a_color_profiles_path()
    if not profiles_path.exists():
        raise FileNotFoundError(f"File profili colore del Flusso A non trovato: {profiles_path}. Esegui prima il Flusso A.")

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"Profili colore Flusso A caricati da: {profiles_path.name}")
    return profiles


def classify_coin_color(
    img_bgr: np.ndarray,
    img_white_patch: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    profiles: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classifica una moneta in '1 euro', '2 euro', 'copper' o 'gold'.

    Args:
        img_bgr: Immagine target intera (BGR).
        img_white_patch: La stessa immagine dopo white-patch (Modulo 4).
        cx, cy, r: Cerchio della moneta rilevato dal Modulo 3.
        profiles: Soglie del Flusso A.

    Returns:
        Dict[str, Any]: campi 'kind', 'confidence', 'confident' e 'color_metrics' della moneta.
    """
    bimetal, single = profiles["bimetal"], profiles["single_metal"]

    delta = bimetal_classifier.ring_minus_centre(img_bgr, cx, cy, r)
    label = bimetal_classifier.classify(delta, bimetal["threshold"])
    bimetal_conf = bimetal_classifier.confidence(delta, bimetal["threshold"], bimetal["half_gap"])
    metrics = {"delta_b": round(delta, 2), "bimetal_confidence": round(bimetal_conf, 2)}
    confident = bimetal_conf >= bimetal["confident"]
    confidence = bimetal_conf

    if label == "single-metal":
        hue = singlemetal_classifier.coin_hue(img_white_patch, cx, cy, r)
        label = singlemetal_classifier.classify(hue, single["threshold"])
        colour_conf = singlemetal_classifier.confidence(hue, single["threshold"], single["half_gap"])
        metrics.update({"hue": round(hue, 1), "colour_confidence": round(colour_conf, 2)})
        confident = confident and colour_conf >= single["confident"]
        confidence = min(bimetal_conf, colour_conf)

    return {
        "kind": BIMETAL_KINDS.get(label, label),
        "confidence": round(confidence, 2),
        "confident": bool(confident),
        "color_metrics": metrics
    }


def classify_all_coins(
    target_images: Dict[str, np.ndarray],
    white_patch_images: Dict[str, np.ndarray],
    annotated_images: Dict[str, Any],
    profiles: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Esegue la classificazione cromatica su tutte le monete di tutte le immagini target.

    Args:
        target_images: Immagini target BGR preparate da median_target_images del Modulo 4
            (solo Median Filter, senza NLM).
        white_patch_images: Le stesse immagini dopo white_patch_images del Modulo 4.
        annotated_images: Data structure del Modulo 3 con i cerchi rilevati.
        profiles: Soglie del Flusso A (se None vengono caricate dagli artifacts).
        save_artifacts: Se True aggiorna il manifest unificato 'annotated_images.json'.

    Returns:
        Dict[str, Any]: Data structure con 'kind', 'confidence', 'confident' e 'color_metrics' per ogni coin.
    """
    if profiles is None:
        profiles = load_reference_color_profiles()

    updated_annotated: Dict[str, Any] = json.loads(json.dumps(annotated_images))
    kind_counts = {"1 euro": 0, "2 euro": 0, "gold": 0, "copper": 0}
    unsure = 0

    for img_name, entry in updated_annotated.items():
        img_bgr = target_images[img_name]
        img_white_patch = white_patch_images[img_name]

        for coin in entry.get("coins", []):
            cx, cy = coin["center"]
            coin.pop("class", None)
            coin.update(classify_coin_color(img_bgr, img_white_patch, cx, cy, coin["radius"], profiles))
            kind_counts[coin["kind"]] += 1
            unsure += not coin["confident"]

    if save_artifacts:
        ann_main_path = get_default_annotated_images_path()
        ann_main_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_main_path, "w", encoding="utf-8") as f:
            json.dump(updated_annotated, f, indent=2)
        print(f"Aggiornato manifest unificato 'annotated_images.json' in: {ann_main_path}")

    total_coins = sum(kind_counts.values())
    print("\nRiepilogo Classificazione Cromatica (Kind) delle Monete:")
    for knd_name, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_coins * 100) if total_coins > 0 else 0
        print(f"  - {knd_name:7s}: {count:3d} monete ({pct:5.1f}%)")
    print(f"  - {unsure} monete sotto il cut-off di confidence")

    return updated_annotated


def plot_classified_coins_sample(
    target_images: Dict[str, np.ndarray],
    updated_annotated: Dict[str, Any],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (16, 8)
):
    """Mostra a schermo un campione casuale di monete con tipologia ('kind') e confidence."""
    coins = [(img_name, coin) for img_name, entry in updated_annotated.items() for coin in entry.get("coins", [])]
    if not coins:
        print("Nessuna moneta classificata disponibile per la visualizzazione.")
        return

    if seed is not None:
        random.seed(seed)
    sample = random.sample(coins, min(n_samples, len(coins)))

    ncols = 4
    nrows = int(np.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.axis("off")

    for ax, (img_name, coin) in zip(axes, sample):
        x, y, r = round(coin["center"][0]), round(coin["center"][1]), round(coin["radius"])
        crop = target_images[img_name][max(0, y - r):y + r, max(0, x - r):x + r]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        flag = "" if coin["confident"] else " (incerta)"
        ax.set_title(f"{img_name}_{coin.get('coin_id', 0)}: {coin['kind'].upper()}\nconfidence {coin['confidence']:.2f}{flag}",
                     fontsize=10, fontweight="bold")

    plt.suptitle(f"Modulo 5: Campione Monete con Tipologia (Kind) e Confidence ({len(sample)} monete)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
