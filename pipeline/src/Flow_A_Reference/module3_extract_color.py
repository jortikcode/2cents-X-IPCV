"""Modulo 3: Estrazione delle caratteristiche di colore (Delta b*, a*, b*, R/G) e calibrazione delle 4 classi."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_A_Reference/module3_color."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference" / "module3_color"


def compute_delta_b(roi_bgr: np.ndarray, cx: float = 125.0, cy: float = 125.0, r: float = 100.0) -> float:
    """
    Calcola la differenza di ingiallimento (LAB b*) tra anello esterno e centro:
    Delta b* = mean(b*_ring) - mean(b*_centre)
    
    - Anello: raggio compreso tra 0.80*r e 0.90*r
    - Centro: raggio inferiore a 0.60*r
    
    Valori attesi:
    - 1 euro (anello ottone, centro argento) -> Delta b* >> 0 (positivo, ~ +14)
    - 2 euro (anello argento, centro ottone) -> Delta b* << 0 (negativo, ~ -13)
    - Monometalliche (copper/gold)            -> Delta b* ~= 0 (|Delta b*| <= 2.3)
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_star = lab[:, :, 2]

    ys, xs = np.ogrid[:roi_bgr.shape[0], :roi_bgr.shape[1]]
    dist = np.hypot(xs - cx, ys - cy)

    centre_mask = dist < (0.60 * r)
    ring_mask = (dist >= 0.80 * r) & (dist < 0.90 * r)

    if np.sum(ring_mask) == 0 or np.sum(centre_mask) == 0:
        return 0.0

    return float(b_star[ring_mask].mean() - b_star[centre_mask].mean())


def compute_disc_color_stats(roi_bgr: np.ndarray, cx: float = 125.0, cy: float = 125.0, r: float = 100.0) -> Dict[str, float]:
    """
    Estrae le statistiche medie di colore (L*, a*, b*, R/G ratio) all'interno del disco della moneta (0.85*r).
    """
    ys, xs = np.ogrid[:roi_bgr.shape[0], :roi_bgr.shape[1]]
    dist = np.hypot(xs - cx, ys - cy)
    disc_mask = dist < (0.85 * r)

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[disc_mask]
    bgr = roi_bgr.astype(np.float32)[disc_mask]

    mean_b = float(bgr[:, 0].mean())
    mean_g = float(bgr[:, 1].mean())
    mean_r = float(bgr[:, 2].mean())

    return {
        "L_star": round(float(lab[:, 0].mean()), 2),
        "a_star": round(float(lab[:, 1].mean()), 2),
        "b_star": round(float(lab[:, 2].mean()), 2),
        "rg_ratio": round(float(mean_r / (mean_g + 1e-5)), 2),
        "mean_R": round(mean_r, 2),
        "mean_G": round(mean_g, 2),
        "mean_B": round(mean_b, 2),
    }


def extract_and_save_color_profiles(
    rois: Dict[str, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Estrae i parametri cromatici di ogni ROI, calibra i profili per le 4 classi
    ('1euro', '2euro', 'copper', 'gold') e serializza i risultati su file JSON.
    
    Args:
        rois: Dizionario {nome_moneta: roi_bgr} restituito dal Modulo 2.
        output_dir: Cartella per salvare il file JSON (default: artifacts/.../module3_color).
        
    Returns:
        Dict[str, Any]: Mappa completa dei profili cromatici delle 4 classi e delle soglie di decisione.
    """
    out_path = Path(output_dir) if output_dir is not None else get_default_artifacts_dir()
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Calcolo metriche per ciascuna moneta
    coin_data = {}
    for name, roi in rois.items():
        delta_b = compute_delta_b(roi)
        disc_stats = compute_disc_color_stats(roi)
        coin_data[name] = {
            "delta_b": round(delta_b, 2),
            **disc_stats
        }

    # Definizione delle monete appartenenti a ciascuna classe
    copper_names = ["1cent", "2cent", "5cent"]
    gold_names = ["10cent", "20cent", "50cent"]

    # 2. Statistiche aggregate per la classe 'copper'
    copper_deltas = [coin_data[n]["delta_b"] for n in copper_names if n in coin_data]
    copper_a_stars = [coin_data[n]["a_star"] for n in copper_names if n in coin_data]
    copper_b_stars = [coin_data[n]["b_star"] for n in copper_names if n in coin_data]
    copper_rg = [coin_data[n]["rg_ratio"] for n in copper_names if n in coin_data]

    # 3. Statistiche aggregate per la classe 'gold'
    gold_deltas = [coin_data[n]["delta_b"] for n in gold_names if n in coin_data]
    gold_a_stars = [coin_data[n]["a_star"] for n in gold_names if n in coin_data]
    gold_b_stars = [coin_data[n]["b_star"] for n in gold_names if n in coin_data]
    gold_rg = [coin_data[n]["rg_ratio"] for n in gold_names if n in coin_data]

    # 4. Calibrazione soglie di decisione
    single_deltas_abs = [abs(d) for d in copper_deltas + gold_deltas]
    bimetal_deltas_abs = [
        abs(coin_data[n]["delta_b"]) for n in ["1euro", "2euro"] if n in coin_data
    ]
    max_single_delta = max(single_deltas_abs) if single_deltas_abs else 2.5
    min_bimetal_delta = min(bimetal_deltas_abs) if bimetal_deltas_abs else 13.0
    bimetal_tau = round((max_single_delta + min_bimetal_delta) / 2.0, 2)

    # Centrodi nello spazio (a*, b*) per Copper e Gold
    copper_mean_a = float(np.mean(copper_a_stars))
    copper_mean_b = float(np.mean(copper_b_stars))
    gold_mean_a = float(np.mean(gold_a_stars))
    gold_mean_b = float(np.mean(gold_b_stars))

    # Punto medio e vettore normale (Copper - Gold)
    mid_a = (copper_mean_a + gold_mean_a) / 2.0
    mid_b = (copper_mean_b + gold_mean_b) / 2.0
    w_a = copper_mean_a - gold_mean_a
    w_b = copper_mean_b - gold_mean_b

    # Retta bisecante perpendicolare: w_a * (a* - mid_a) + w_b * (b* - mid_b) = 0
    # => b* = k * a* + q  con  k = -w_a / w_b  e  q = mid_b - k * mid_a
    slope_k = -w_a / w_b if abs(w_b) > 1e-6 else 0.0
    intercept_q = mid_b - (slope_k * mid_a)

    profiles = {
        "classes": {
            "1euro": {
                "description": "Bimetallica: anello esterno ottone/oro (giallo), centro nickel (argento)",
                "coin": "1euro",
                "delta_b": coin_data.get("1euro", {}).get("delta_b", 14.08),
                "a_star": coin_data.get("1euro", {}).get("a_star", 137.87),
                "b_star": coin_data.get("1euro", {}).get("b_star", 136.90),
                "rg_ratio": coin_data.get("1euro", {}).get("rg_ratio", 1.37)
            },
            "2euro": {
                "description": "Bimetallica: anello esterno nickel (argento), centro ottone/oro (giallo)",
                "coin": "2euro",
                "delta_b": coin_data.get("2euro", {}).get("delta_b", -13.13),
                "a_star": coin_data.get("2euro", {}).get("a_star", 131.74),
                "b_star": coin_data.get("2euro", {}).get("b_star", 140.38),
                "rg_ratio": coin_data.get("2euro", {}).get("rg_ratio", 1.19)
            },
            "copper": {
                "description": "Monometallica rame (1cent, 2cent, 5cent) - elevata componente rossa a*",
                "coins": copper_names,
                "centroid": {
                    "a_star": round(copper_mean_a, 2),
                    "b_star": round(copper_mean_b, 2)
                },
                "delta_b": {
                    "mean": round(float(np.mean(copper_deltas)), 2),
                    "min": min(copper_deltas),
                    "max": max(copper_deltas)
                },
                "a_star": {
                    "mean": round(copper_mean_a, 2),
                    "min": min(copper_a_stars),
                    "max": max(copper_a_stars)
                },
                "b_star": {
                    "mean": round(copper_mean_b, 2),
                    "min": min(copper_b_stars),
                    "max": max(copper_b_stars)
                },
                "rg_ratio": {
                    "mean": round(float(np.mean(copper_rg)), 2),
                    "min": min(copper_rg),
                    "max": max(copper_rg)
                }
            },
            "gold": {
                "description": "Monometallica oro nordico (10cent, 20cent, 50cent) - ottone dorato",
                "coins": gold_names,
                "centroid": {
                    "a_star": round(gold_mean_a, 2),
                    "b_star": round(gold_mean_b, 2)
                },
                "delta_b": {
                    "mean": round(float(np.mean(gold_deltas)), 2),
                    "min": min(gold_deltas),
                    "max": max(gold_deltas)
                },
                "a_star": {
                    "mean": round(gold_mean_a, 2),
                    "min": min(gold_a_stars),
                    "max": max(gold_a_stars)
                },
                "b_star": {
                    "mean": round(gold_mean_b, 2),
                    "min": min(gold_b_stars),
                    "max": max(gold_b_stars)
                },
                "rg_ratio": {
                    "mean": round(float(np.mean(gold_rg)), 2),
                    "min": min(gold_rg),
                    "max": max(gold_rg)
                }
            }
        },
        "decision_rules": {
            "bimetal_threshold_tau": bimetal_tau,
            "rule_1euro": f"delta_b > +{bimetal_tau}",
            "rule_2euro": f"delta_b < -{bimetal_tau}",
            "rule_single_metal": f"abs(delta_b) <= {bimetal_tau}",
            "copper_gold_boundary": {
                "type": "nearest_centroid_bisector",
                "slope_k": round(slope_k, 3),
                "intercept_q": round(intercept_q, 2),
                "formula": f"b* = {round(slope_k, 3)} * a* + {round(intercept_q, 2)}"
            },
            "rule_copper": f"abs(delta_b) <= {bimetal_tau} AND b_star < ({round(slope_k, 3)} * a_star + {round(intercept_q, 2)})",
            "rule_gold": f"abs(delta_b) <= {bimetal_tau} AND b_star >= ({round(slope_k, 3)} * a_star + {round(intercept_q, 2)})"
        },
        "coins_detail": coin_data
    }

    # Serializzazione su file JSON
    json_filepath = out_path / "color_profiles.json"
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Profili colore salvati con successo in: {json_filepath}")
    return profiles


def plot_color_analysis(profiles: Dict[str, Any], figsize=(15, 5)):
    """
    Visualizza a schermo i grafici analitici delle 4 classi cromatiche:
    1. Grafico a barre di Delta b* (discriminatore 1 euro vs 2 euro vs monometalliche)
    2. Scatter plot nello spazio colore CIELAB (a* vs b*)
    """
    coins_detail = profiles["coins_detail"]
    names = sorted(coins_detail.keys())
    deltas = [coins_detail[n]["delta_b"] for n in names]
    a_stars = [coins_detail[n]["a_star"] for n in names]
    b_stars = [coins_detail[n]["b_star"] for n in names]

    tau = profiles["decision_rules"]["bimetal_threshold_tau"]

    # Assegnazione colori grafici alle 4 classi
    def get_color(name):
        if name == "1euro":
            return "#1f77b4"  # Blu
        elif name == "2euro":
            return "#17becf"  # Ciano
        elif name in ["1cent", "2cent", "5cent"]:
            return "#d95f02"  # Rame / Arancione scuro
        else:
            return "#e6ab02"  # Oro / Giallo scuro

    bar_colors = [get_color(n) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- 1. Grafico a barre: Delta b* ---
    bars = ax1.bar(names, deltas, color=bar_colors, edgecolor="black", alpha=0.85, width=0.6)
    ax1.axhline(tau, color="red", linestyle="--", linewidth=1.5, label=f"Soglia +tau (+{tau}) -> 1 euro")
    ax1.axhline(-tau, color="blue", linestyle="--", linewidth=1.5, label=f"Soglia -tau (-{tau}) -> 2 euro")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_title("1. Discriminatore Bimetallico (Delta b* = Anello - Centro)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Delta b* (Ingiallimento)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    for bar, d in zip(bars, deltas):
        ypos = d + (0.5 if d >= 0 else -1.2)
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos, f"{d:+.2f}", ha="center", fontsize=9, fontweight="bold")

    # --- 2. Scatter plot: a* (Rosso) vs b* (Giallo) ---
    for n, a, b in zip(names, a_stars, b_stars):
        c = get_color(n)
        ax2.scatter(a, b, color=c, s=140, edgecolors="black", linewidth=1.5, zorder=3)
        ax2.annotate(f" {n}", (a + 0.3, b + 0.3), fontsize=9, fontweight="bold")

    # Centrodi per Copper e Gold
    cu_cent = profiles["classes"]["copper"]["centroid"]
    au_cent = profiles["classes"]["gold"]["centroid"]
    ax2.scatter(cu_cent["a_star"], cu_cent["b_star"], marker="P", s=160, color="#b30000", edgecolors="black", zorder=4, label=f"Centroide Copper ({cu_cent['a_star']}, {cu_cent['b_star']})")
    ax2.scatter(au_cent["a_star"], au_cent["b_star"], marker="P", s=160, color="#d99b00", edgecolors="black", zorder=4, label=f"Centroide Gold ({au_cent['a_star']}, {au_cent['b_star']})")

    # Retta di decisione diagonale (bisecante perpendicolare dei centroidi)
    boundary_info = profiles["decision_rules"]["copper_gold_boundary"]
    k = boundary_info["slope_k"]
    q = boundary_info["intercept_q"]
    a_vals = np.linspace(139.0, 148.5, 100)
    b_vals = k * a_vals + q
    ax2.plot(a_vals, b_vals, color="purple", linestyle="--", linewidth=2.0, zorder=2,
             label=f"Split Diagonale:\nb* = {k:.2f} a* + ({q:.1f})")

    ax2.set_title("2. Separazione Cromatica CIELAB (a* vs b*)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("a* (Verde <-> Rosso)")
    ax2.set_ylabel("b* (Blu <-> Giallo)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8)

    plt.suptitle("Modulo 3: Estrazione del Colore e Profiling delle 4 Classi (1euro, 2euro, copper, gold)", fontsize=13, y=1.02)
    plt.tight_layout()

    # Salvataggio grafico nella cartella artifacts
    artifacts_dir = get_default_artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_file = artifacts_dir / "color_profiles_plot.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Grafico profili colore salvato in: {plot_file}")

    plt.show()
    plt.close(fig)
