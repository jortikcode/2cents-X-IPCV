"""Modulo 3: Calibrazione delle soglie di colore sulle 8 immagini di riferimento.

Le misure e le soglie sono quelle di code/bimetal_classifier.py e code/singlemetal_classifier.py:
- Bimetalliche vs monometalliche: Delta b* (anello - centro), soglia a meta' tra la monometallica
  piu' estrema e la bimetallica meno estrema.
- Rame vs oro: tonalita' (Hue) dopo white-patch sull'intera immagine, soglia a meta' tra il rame
  piu' giallo e l'oro meno giallo. Il white-patch e' quello del Modulo 4 del Flusso B: le reference
  bilanciate vengono salvate negli artifacts e misurate da li'.

Le soglie vengono salvate in artifacts/Flow_A_Reference/module3_color/color_profiles.json
e lette dal Modulo 5 del Flusso B.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt

from module1_load_references import get_default_ref_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
import bimetal_classifier
import singlemetal_classifier

# White-patch condiviso con il Flusso B
sys.path.append(str(Path(__file__).resolve().parents[1] / "Flow_B_Target"))
from module4_denoise_colored import white_patch


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_A_Reference/module3_color."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module3_color"


def save_white_patch_references(ref_dir: Path, out_dir: Path) -> Path:
    """Salva in out_dir una copia con white-patch di ogni immagine di riferimento e restituisce out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(ref_dir.glob("*.jpg")):
        cv2.imwrite(str(out_dir / path.name), white_patch(cv2.imread(str(path))), [cv2.IMWRITE_JPEG_QUALITY, 100])
    return out_dir


def extract_and_save_color_profiles(
    reference_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Calcola le due soglie di colore dalle immagini di riferimento e le salva su file JSON.

    Per ciascuna soglia viene salvato anche 'half_gap' (meta' della distanza tra le due monete
    di riferimento piu' vicine alla soglia), che e' l'unita' di misura della confidence.

    Args:
        reference_dir: Cartella delle immagini di riferimento (default: coin_dataset/reference_set).
        output_dir: Cartella per salvare il file JSON (default: artifacts/.../module3_color).

    Returns:
        Dict[str, Any]: Soglie, half_gap e valori misurati su ogni moneta di riferimento.
    """
    ref_dir = Path(reference_dir) if reference_dir is not None else get_default_ref_dir()
    out_path = Path(output_dir) if output_dir is not None else get_default_artifacts_dir()
    out_path.mkdir(parents=True, exist_ok=True)

    tau, tau_half_gap = bimetal_classifier.threshold_from_references(ref_dir)
    white_patch_ref_dir = save_white_patch_references(ref_dir, out_path / "white_patch_references")
    hue_thr, hue_half_gap = singlemetal_classifier.threshold_from_references(white_patch_ref_dir)

    profiles = {
        "bimetal": {
            "feature": "delta_b (LAB b* anello - centro)",
            "threshold": tau,
            "half_gap": tau_half_gap,
            "confident": bimetal_classifier.CONFIDENT,
            "references": bimetal_classifier.reference_deltas(ref_dir)
        },
        "single_metal": {
            "feature": "hue (HSV, gradi, dopo white-patch)",
            "threshold": hue_thr,
            "half_gap": hue_half_gap,
            "confident": singlemetal_classifier.CONFIDENT,
            "references": singlemetal_classifier.reference_hues(white_patch_ref_dir)
        }
    }

    json_filepath = out_path / "color_profiles.json"
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Soglia bimetallica (Delta b*): {tau:.2f}  (half gap {tau_half_gap:.2f})")
    print(f"Soglia rame/oro (Hue):         {hue_thr:.1f}°  (half gap {hue_half_gap:.1f}°)")
    print(f"Profili colore salvati con successo in: {json_filepath}")
    return profiles


def plot_color_analysis(profiles: Dict[str, Any], figsize=(15, 5)):
    """
    Visualizza i valori misurati sulle monete di riferimento e le soglie che ne derivano:
    1. Delta b* delle 8 monete con le soglie +tau / -tau
    2. Hue delle 6 monete monometalliche con la soglia rame/oro
    """
    bimetal, single = profiles["bimetal"], profiles["single_metal"]
    tau, hue_thr = bimetal["threshold"], single["threshold"]

    def get_color(name):
        if name in bimetal_classifier.BIMETAL_REFERENCES:
            return "#1f77b4"
        return "#d95f02" if name in singlemetal_classifier.COPPER_REFERENCES else "#e6ab02"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, values, unit in ((ax1, bimetal["references"], ""), (ax2, single["references"], "°")):
        names = sorted(values, key=values.get)
        bars = ax.bar(names, [values[n] for n in names], color=[get_color(n) for n in names],
                      edgecolor="black", alpha=0.85, width=0.6)
        ax.bar_label(bars, labels=[f"{values[n]:+.1f}{unit}" for n in names], fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax1.axhline(tau, color="red", linestyle="--", linewidth=1.5, label=f"+tau (+{tau:.2f}) -> 1 euro")
    ax1.axhline(-tau, color="blue", linestyle="--", linewidth=1.5, label=f"-tau (-{tau:.2f}) -> 2 euro")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_title("1. Bimetalliche vs monometalliche (Delta b* = Anello - Centro)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Delta b*")
    ax1.legend(loc="best", fontsize=9)

    ax2.axhline(hue_thr, color="purple", linestyle="--", linewidth=1.5, label=f"Soglia {hue_thr:.1f}° (sotto: rame, sopra: oro)")
    ax2.set_title("2. Rame vs oro (Hue dopo white-patch)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Hue (gradi)")
    ax2.legend(loc="best", fontsize=9)

    plt.suptitle("Modulo 3: Soglie di colore calibrate sulle monete di riferimento", fontsize=13, y=1.02)
    plt.tight_layout()

    artifacts_dir = get_default_artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_file = artifacts_dir / "color_profiles_plot.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Grafico profili colore salvato in: {plot_file}")

    plt.show()
    plt.close(fig)
