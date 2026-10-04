"""Modulo 9 (Flow B): Valutazione della Pipeline con Ground Truth e Metriche di Accuratezza.

Questo modulo:
1. Carica le annotazioni di Ground Truth da 'pipeline/conf/annotations_with_position.csv'
   (o 'annotations_with_positions.csv').
2. Carica i risultati finali della pipeline da 'annotated_images.json'.
3. Esegue il matching spaziale biunivoco (Bipartite Matching con algoritmo ungherese)
   tra le monete rilevate e le monete di Ground Truth.
4. Calcola metriche complete di valutazione:
   - Metriche di Detection (TP, FP, FN, Precision, Recall, F1).
   - Metriche di Classificazione per denominazione (Precision, Recall, F1-Score, Support).
   - Accuratezza globale (sulle monete rilevate e su tutto il dataset GT).
   - Matrice di confusione 8x8 (assoluta e normalizzata per riga).
   - Metriche di conteggio monetario (Valore totale GT vs Predetto, errore assoluto e relativo,
     Mean Absolute Error per immagine, percentuale di immagini con conteggio esatto).
   - Suddivisione delle performance: Immagini a moneta singola (Modulo 7.1) vs multiple (Modulo 7.2).
   - Identificazione e ranking delle monete più facili e più difficili.
5. Genera e salva visualizzazioni professionali in artifacts:
   - 'easiest_hardest_coins.png': Grafico a barre con ranking di difficoltà per denominazione.
   - 'confusion_matrix.png': Heatmap della matrice di confusione 8x8 con conteggi e percentuali.
   - 'performance_overview.png': Dashboard a 4 riquadri (Precision/Recall/F1, Valore monetario,
     Single vs Multi coin, Distribuzione errori monetari).
   - 'evaluation_summary.json': Report strutturato completo in formato JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, confusion_matrix


# Mappatura standard dei centesimi alla classe testuale
DENOM_MAP: Dict[int, str] = {
    1: "1cent",
    2: "2cent",
    5: "5cent",
    10: "10cent",
    20: "20cent",
    50: "50cent",
    100: "1euro",
    200: "2euro"
}

# Mappatura inversa dalla classe ai centesimi
CLASS_TO_CENT: Dict[str, int] = {v: k for k, v in DENOM_MAP.items()}

# Valori monetari nominali in Euro
CLASS_VALUES: Dict[str, float] = {
    "1cent": 0.01,
    "2cent": 0.02,
    "5cent": 0.05,
    "10cent": 0.10,
    "20cent": 0.20,
    "50cent": 0.50,
    "1euro": 1.00,
    "2euro": 2.00
}

# Mappa delle classi ai gruppi di colore attesi ("kind")
CLASS_TO_KIND: Dict[str, str] = {
    "1cent": "copper",
    "2cent": "copper",
    "5cent": "copper",
    "10cent": "gold",
    "20cent": "gold",
    "50cent": "gold",
    "1euro": "1 euro",
    "2euro": "2 euro"
}

ORDERED_CLASSES: List[str] = [
    "1cent", "2cent", "5cent", "10cent", "20cent", "50cent", "1euro", "2euro"
]


def get_default_config_path() -> Path:
    """Restituisce il percorso del file di configurazione per il Modulo 9."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "conf" / "Flow_B_Target" / "evaluation_params.json"


def get_default_ground_truth_path() -> Path:
    """Restituisce il percorso del file CSV di Ground Truth."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    candidate1 = pipeline_dir / "conf" / "annotations_with_position.csv"
    if candidate1.exists():
        return candidate1
    candidate2 = pipeline_dir / "conf" / "annotations_with_positions.csv"
    if candidate2.exists():
        return candidate2
    return candidate1


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json principale."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module3_coin_detection" / "annotated_images.json"


def get_default_artifacts_dir() -> Path:
    """Restituisce la cartella di output per i risultati del Modulo 9."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module9_evaluation"


def resolve_project_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    """Risolve un percorso relativo rispetto alla root del progetto o CWD."""
    if p is None:
        return None
    path_obj = Path(p)
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj
    if path_obj.exists():
        return path_obj.resolve()
    # Controllo rispetto alla root del progetto (4 livelli sopra module9_evaluation.py)
    project_root = Path(__file__).resolve().parents[3]
    cand_root = project_root / path_obj
    if cand_root.exists():
        return cand_root.resolve()
    # Controllo rispetto alla cartella pipeline (3 livelli sopra)
    pipeline_dir = Path(__file__).resolve().parents[2]
    cand_pipe = pipeline_dir / path_obj
    if cand_pipe.exists():
        return cand_pipe.resolve()
    return cand_root.resolve()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica la configurazione del modulo dal file JSON."""
    resolved = resolve_project_path(config_path) if config_path else get_default_config_path()
    if resolved and resolved.exists():
        with open(resolved, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_ground_truth(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Carica il dataframe delle annotazioni di Ground Truth."""
    p = resolve_project_path(csv_path) if csv_path else get_default_ground_truth_path()
    if p is None or not p.exists():
        # Fallback cercato direttamente nella cartella conf di pipeline
        pipeline_dir = Path(__file__).resolve().parents[2]
        for name in ["annotations_with_position.csv", "annotations_with_positions.csv"]:
            cand = pipeline_dir / "conf" / name
            if cand.exists():
                p = cand
                break
        else:
            raise FileNotFoundError(f"File Ground Truth CSV non trovato in: {p}")

    df = pd.read_csv(p)
    # Assicura il formato corretto del nome immagine
    df["image_stem"] = df["image"].apply(lambda x: str(x).replace(".jpg", "").replace(".png", "").strip())
    df["gt_class"] = df["denomination_cent"].map(DENOM_MAP)
    df["gt_kind"] = df["gt_class"].map(CLASS_TO_KIND)
    df["gt_value_eur"] = df["gt_class"].map(CLASS_VALUES)
    return df


def load_detections(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carica le annotazioni predette dalla pipeline da annotated_images.json."""
    p = resolve_project_path(json_path) if json_path else get_default_annotated_images_path()
    if p is None or not p.exists():
        p = get_default_annotated_images_path()
    if not p.exists():
        raise FileNotFoundError(f"File annotated_images.json non trovato in: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def match_detections_with_ground_truth(
    df_gt: pd.DataFrame,
    annotated_images: Dict[str, Any],
    max_distance_px: float = 50.0,
    radius_factor: float = 1.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Esegue il bipartite matching ungherese tra monete rilevate e ground truth.

    Returns:
        matched_pairs: Lista di dizionari con monete abbinate (TP detection).
        unmatched_detections: Lista di cerchi rilevati senza corrispondente GT (FP detection).
        unmatched_ground_truths: Lista di monete GT non rilevate dalla Hough (FN detection).
    """
    matched_pairs: List[Dict[str, Any]] = []
    unmatched_detections: List[Dict[str, Any]] = []
    unmatched_ground_truths: List[Dict[str, Any]] = []

    # Raggruppa GT per immagine
    gt_by_img = {k: v for k, v in df_gt.groupby("image_stem")}

    # Insieme di tutte le immagini uniche
    all_image_keys = sorted(set(list(annotated_images.keys()) + list(gt_by_img.keys())))

    for img_key in all_image_keys:
        det_data = annotated_images.get(img_key, {})
        det_coins = det_data.get("coins", [])
        
        gt_subset = gt_by_img.get(img_key)
        gt_coins = gt_subset.to_dict("records") if gt_subset is not None else []

        n_det = len(det_coins)
        n_gt = len(gt_coins)

        if n_det == 0 and n_gt == 0:
            continue

        if n_det == 0:
            for g in gt_coins:
                unmatched_ground_truths.append({
                    "image": img_key,
                    "gt_coin": g
                })
            continue

        if n_gt == 0:
            for d in det_coins:
                unmatched_detections.append({
                    "image": img_key,
                    "det_coin": d
                })
            continue

        # Matrice di costo con distanze Euclidee
        cost_matrix = np.zeros((n_det, n_gt), dtype=float)
        for i, d in enumerate(det_coins):
            cx_d, cy_d = d["center"]
            for j, g in enumerate(gt_coins):
                cost_matrix[i, j] = float(np.hypot(cx_d - g["cx"], cy_d - g["cy"]))

        # Risoluzione ottima di assegnamento globale
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_det = set()
        assigned_gt = set()

        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            # La moneta è accoppiata se il centro rilevato cade entro la moneta reale
            # o entro la soglia minima di tolleranza
            r_gt = float(gt_coins[c]["radius"])
            gating_thresh = max(max_distance_px, radius_factor * r_gt)

            if dist <= gating_thresh:
                matched_pairs.append({
                    "image": img_key,
                    "distance_px": float(dist),
                    "n_coins_image": det_data.get("n_coins", len(det_coins)),
                    "det_coin": det_coins[r],
                    "gt_coin": gt_coins[c],
                    "pred_class": det_coins[r].get("class"),
                    "gt_class": gt_coins[c]["gt_class"],
                    "pred_kind": det_coins[r].get("kind"),
                    "gt_kind": gt_coins[c]["gt_kind"],
                    "kind_confidence": det_coins[r].get("confidence", 0.0),
                    "classification_method": det_coins[r].get("classification_method", "unknown")
                })
                assigned_det.add(r)
                assigned_gt.add(c)

        for i in range(n_det):
            if i not in assigned_det:
                unmatched_detections.append({
                    "image": img_key,
                    "det_coin": det_coins[i]
                })

        for j in range(n_gt):
            if j not in assigned_gt:
                unmatched_ground_truths.append({
                    "image": img_key,
                    "gt_coin": gt_coins[j]
                })

    return matched_pairs, unmatched_detections, unmatched_ground_truths


def compute_metrics(
    df_gt: pd.DataFrame,
    annotated_images: Dict[str, Any],
    matched_pairs: List[Dict[str, Any]],
    unmatched_detections: List[Dict[str, Any]],
    unmatched_ground_truths: List[Dict[str, Any]],
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calcola tutte le metriche quantitative di detection, classificazione e conteggio."""
    if classes is None:
        classes = ORDERED_CLASSES

    # 1. Metriche di Detection
    total_gt_coins = len(df_gt)
    total_detected_circles = sum(len(img.get("coins", [])) for img in annotated_images.values())
    tp_detection = len(matched_pairs)
    fp_detection = len(unmatched_detections)
    fn_detection = len(unmatched_ground_truths)

    det_precision = tp_detection / (tp_detection + fp_detection) if (tp_detection + fp_detection) > 0 else 0.0
    det_recall = tp_detection / (tp_detection + fn_detection) if (tp_detection + fn_detection) > 0 else 0.0
    det_f1 = (2 * det_precision * det_recall / (det_precision + det_recall)) if (det_precision + det_recall) > 0 else 0.0

    # 2. Metriche di Classificazione sui cerchi correttamente individuati
    y_true = [m["gt_class"] for m in matched_pairs]
    y_pred = [m["pred_class"] for m in matched_pairs]

    correct_matched = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy_matched = correct_matched / len(matched_pairs) if matched_pairs else 0.0
    accuracy_pipeline_total = correct_matched / total_gt_coins if total_gt_coins > 0 else 0.0

    # Report dettagliato scikit-learn
    clf_report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    
    # Matrice di Confusione 8x8
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Matrice normalizzata per riga (Recall percentages)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    # Per-class summary
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for c in classes:
        metrics = clf_report.get(c, {})
        support = int(metrics.get("support", 0))
        prec = float(metrics.get("precision", 0.0))
        rec = float(metrics.get("recall", 0.0))
        f1 = float(metrics.get("f1-score", 0.0))
        per_class_metrics[c] = {
            "denomination": c,
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "nominal_value_eur": CLASS_VALUES[c]
        }

    # Ranking: Monete più facili e più difficili
    # Ordiniamo per F1-score e Recall
    ranked_classes = sorted(
        per_class_metrics.values(),
        key=lambda x: (x["f1_score"], x["recall"]),
        reverse=True
    )
    easiest_coins = [x["denomination"] for x in ranked_classes[:3]]
    hardest_coins = [x["denomination"] for x in reversed(ranked_classes[-3:])]

    # 3. Analisi Sub-pipeline: Singola Moneta vs Monete Multiple
    single_pairs = [m for m in matched_pairs if m["n_coins_image"] == 1]
    multi_pairs = [m for m in matched_pairs if m["n_coins_image"] > 1]

    single_correct = sum(1 for m in single_pairs if m["pred_class"] == m["gt_class"])
    single_accuracy = (single_correct / len(single_pairs)) if single_pairs else 0.0

    multi_correct = sum(1 for m in multi_pairs if m["pred_class"] == m["gt_class"])
    multi_accuracy = (multi_correct / len(multi_pairs)) if multi_pairs else 0.0

    # 4. Accuratezza al livello Macro-Colore ("kind")
    kind_correct = sum(1 for m in matched_pairs if m["pred_kind"] == m["gt_kind"])
    kind_accuracy = kind_correct / len(matched_pairs) if matched_pairs else 0.0

    # 5. Metriche di Conteggio Monetario
    gt_totals_per_img: Dict[str, float] = {}
    for img_stem, grp in df_gt.groupby("image_stem"):
        gt_totals_per_img[img_stem] = round(grp["denomination_cent"].sum() / 100.0, 2)

    pred_totals_per_img: Dict[str, float] = {}
    for img_key, det_item in annotated_images.items():
        coins = det_item.get("coins", [])
        pred_val = sum(CLASS_VALUES.get(c.get("class", ""), 0.0) for c in coins)
        pred_totals_per_img[img_key] = round(pred_val, 2)

    all_keys = sorted(set(list(gt_totals_per_img.keys()) + list(pred_totals_per_img.keys())))
    img_errors: List[float] = []
    exact_match_count = 0
    diff_under_10cent = 0
    diff_under_20cent = 0
    diff_under_50cent = 0

    per_image_money: Dict[str, Dict[str, float]] = {}
    for k in all_keys:
        gt_v = gt_totals_per_img.get(k, 0.0)
        pr_v = pred_totals_per_img.get(k, 0.0)
        diff = round(pr_v - gt_v, 2)
        abs_diff = abs(diff)
        img_errors.append(abs_diff)

        if abs_diff < 0.001:
            exact_match_count += 1
        if abs_diff <= 0.10:
            diff_under_10cent += 1
        if abs_diff <= 0.20:
            diff_under_20cent += 1
        if abs_diff <= 0.50:
            diff_under_50cent += 1

        per_image_money[k] = {
            "gt_eur": gt_v,
            "pred_eur": pr_v,
            "diff_eur": diff
        }

    total_gt_eur = round(sum(gt_totals_per_img.values()), 2)
    total_pred_eur = round(sum(pred_totals_per_img.values()), 2)
    total_diff_eur = round(total_pred_eur - total_gt_eur, 2)
    total_relative_error_pct = round(abs(total_diff_eur) / total_gt_eur * 100.0, 2) if total_gt_eur > 0 else 0.0
    mae_eur_per_image = round(float(np.mean(img_errors)), 4) if img_errors else 0.0

    return {
        "detection_metrics": {
            "total_gt_coins": total_gt_coins,
            "total_detected_circles": total_detected_circles,
            "true_positives": tp_detection,
            "false_positives": fp_detection,
            "false_negatives": fn_detection,
            "precision": round(det_precision, 4),
            "recall": round(det_recall, 4),
            "f1_score": round(det_f1, 4)
        },
        "classification_metrics": {
            "overall_accuracy_on_detected": round(accuracy_matched, 4),
            "overall_accuracy_on_all_gt": round(accuracy_pipeline_total, 4),
            "macro_precision": round(clf_report["macro avg"]["precision"], 4),
            "macro_recall": round(clf_report["macro avg"]["recall"], 4),
            "macro_f1": round(clf_report["macro avg"]["f1-score"], 4),
            "weighted_precision": round(clf_report["weighted avg"]["precision"], 4),
            "weighted_recall": round(clf_report["weighted avg"]["recall"], 4),
            "weighted_f1": round(clf_report["weighted avg"]["f1-score"], 4),
            "color_kind_accuracy": round(kind_accuracy, 4),
            "per_class": per_class_metrics,
            "ranked_easiest_to_hardest": [x["denomination"] for x in ranked_classes],
            "easiest_coins": easiest_coins,
            "hardest_coins": hardest_coins
        },
        "subpipeline_breakdown": {
            "single_coin": {
                "total_matched": len(single_pairs),
                "correct": single_correct,
                "accuracy": round(single_accuracy, 4),
                "method": "Module 7.1 (Kind restriction + Circular ZNCC)"
            },
            "multi_coin": {
                "total_matched": len(multi_pairs),
                "correct": multi_correct,
                "accuracy": round(multi_accuracy, 4),
                "method": "Module 7.2 (Relative scale consistency + CSP)"
            }
        },
        "monetary_metrics": {
            "total_ground_truth_eur": total_gt_eur,
            "total_predicted_eur": total_pred_eur,
            "total_difference_eur": total_diff_eur,
            "relative_error_pct": total_relative_error_pct,
            "mean_absolute_error_eur_per_image": mae_eur_per_image,
            "exact_matches_count": exact_match_count,
            "exact_matches_pct": round(exact_match_count / len(all_keys) * 100.0, 2),
            "within_10cent_pct": round(diff_under_10cent / len(all_keys) * 100.0, 2),
            "within_20cent_pct": round(diff_under_20cent / len(all_keys) * 100.0, 2),
            "within_50cent_pct": round(diff_under_50cent / len(all_keys) * 100.0, 2),
            "total_images": len(all_keys)
        },
        "confusion_matrix": {
            "classes": classes,
            "raw_matrix": cm.tolist(),
            "normalized_matrix": [[round(val, 4) for val in row] for row in cm_norm.tolist()]
        }
    }


def plot_easiest_hardest_coins(
    metrics: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """Crea e salva il grafico a barre che classifica le monete da più facili a più difficili."""
    per_class = metrics["classification_metrics"]["per_class"]
    ranked_names = metrics["classification_metrics"]["ranked_easiest_to_hardest"]

    # Estrai dati ordinati
    recalls = [per_class[c]["recall"] * 100.0 for c in ranked_names]
    f1_scores = [per_class[c]["f1_score"] * 100.0 for c in ranked_names]
    supports = [per_class[c]["support"] for c in ranked_names]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Palette gradiente da verde (facile) ad arancio/rosso (difficile)
    n = len(ranked_names)
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.20, n))

    bars = ax.barh(ranked_names[::-1], recalls[::-1], color=colors[::-1], edgecolor="black", alpha=0.9, height=0.6)

    # Annotazioni sulle barre
    for bar, c_name, rec, f1, sup in zip(bars, ranked_names[::-1], recalls[::-1], f1_scores[::-1], supports[::-1]):
        width = bar.get_width()
        label_text = f" Recall: {rec:.1f}% | F1: {f1:.1f}% (N={sup})"
        ax.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#222222"
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Recall / Accuratezza di Recupero (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Denominazione Moneta", fontsize=11, fontweight="bold")
    ax.set_title("Ranking delle Monete: Dalle Più Facili (in alto) alle Più Difficili (in basso)", fontsize=13, fontweight="bold", pad=14)

    # Aggiungi badge interpretativi
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
    ax.text(2, n - 0.7, "[+] Piu' Facili (Alta Accuratezza)", color="green", fontweight="bold", fontsize=10)
    ax.text(2, 0.2, "[!] Piu' Difficili (Confusione / Ombra)", color="#c0392b", fontweight="bold", fontsize=10)

    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_confusion_matrix_heatmap(
    metrics: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """Crea e salva una heatmap elegante della matrice di confusione 8x8."""
    classes = metrics["confusion_matrix"]["classes"]
    cm = np.array(metrics["confusion_matrix"]["raw_matrix"])
    cm_norm = np.array(metrics["confusion_matrix"]["normalized_matrix"]) * 100.0

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)

    # Costruisci etichette testuali (conteggio assoluto + % di riga)
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            pct = cm_norm[i, j]
            if count > 0:
                annot[i, j] = f"{count}\n({pct:.0f}%)"
            else:
                annot[i, j] = "0"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Recall per Classe (% riga)"},
        ax=ax,
        linewidths=0.5,
        linecolor="#dddddd"
    )

    ax.set_title("Matrice di Confusione Completa 8x8 (Ground Truth vs Predetto)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Denominazione Predetta dalla Pipeline", fontsize=11, fontweight="bold")
    ax.set_ylabel("Denominazione Reale (Ground Truth)", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_performance_overview_dashboard(
    metrics: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """Genera una dashboard con 4 grafici sintetici riassuntivi."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    classes = ORDERED_CLASSES
    per_class = metrics["classification_metrics"]["per_class"]

    # 1. Top-Left: Precision, Recall e F1-score per classe
    ax1 = axes[0, 0]
    x = np.arange(len(classes))
    width = 0.27

    precs = [per_class[c]["precision"] * 100 for c in classes]
    recs = [per_class[c]["recall"] * 100 for c in classes]
    f1s = [per_class[c]["f1_score"] * 100 for c in classes]

    ax1.bar(x - width, precs, width, label="Precision", color="#3498db")
    ax1.bar(x, recs, width, label="Recall", color="#2ecc71")
    ax1.bar(x + width, f1s, width, label="F1-Score", color="#f39c12")

    ax1.set_title("Metriche per Denominazione: Precision, Recall, F1", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Punteggio (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Top-Right: Moneta Singola vs Monete Multiple
    ax2 = axes[0, 1]
    sub = metrics["subpipeline_breakdown"]
    categories = [
        f"Moneta Singola\n(Mod 7.1 ZNCC)\nN={sub['single_coin']['total_matched']}",
        f"Monete Multiple\n(Mod 7.2 CSP)\nN={sub['multi_coin']['total_matched']}",
        f"Pipeline Globale\n(Totale)\nN={metrics['detection_metrics']['true_positives']}"
    ]
    accuracies = [
        sub["single_coin"]["accuracy"] * 100,
        sub["multi_coin"]["accuracy"] * 100,
        metrics["classification_metrics"]["overall_accuracy_on_detected"] * 100
    ]
    bars_sub = ax2.bar(categories, accuracies, color=["#9b59b6", "#34495e", "#16a085"], width=0.55, edgecolor="black")
    for b, acc in zip(bars_sub, accuracies):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=10.5)

    ax2.set_title("Confronto Accuratezza: Singola Moneta vs Multiple", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuratezza di Classificazione (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # 3. Bottom-Left: Conteggio Monetario Complessivo
    ax3 = axes[1, 0]
    money = metrics["monetary_metrics"]
    comp_labels = ["Valore Reale (GT)", "Valore Predetto (Pipeline)"]
    comp_values = [money["total_ground_truth_eur"], money["total_predicted_eur"]]
    bars_money = ax3.bar(comp_labels, comp_values, color=["#27ae60", "#2980b9"], width=0.45, edgecolor="black")

    for b, v in zip(bars_money, comp_values):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{v:.2f} €", ha="center", fontweight="bold", fontsize=11)

    diff_str = f"Differenza: {money['total_difference_eur']:+.2f} € (Errore: {money['relative_error_pct']}%)"
    ax3.set_title(f"Conteggio Monetario Totale\n{diff_str}", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Valore Totale (€)", fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(comp_values) * 1.2)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Bottom-Right: Tolleranza Errore per Immagine
    ax4 = axes[1, 1]
    tol_labels = ["Esatto (0.00 €)", "Errore ≤ 0.10 €", "Errore ≤ 0.20 €", "Errore ≤ 0.50 €"]
    tol_pcts = [
        money["exact_matches_pct"],
        money["within_10cent_pct"],
        money["within_20cent_pct"],
        money["within_50cent_pct"]
    ]
    bars_tol = ax4.bar(tol_labels, tol_pcts, color="#e67e22", width=0.55, edgecolor="black")

    for b, pct in zip(bars_tol, tol_pcts):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=10.5)

    ax4.set_title(f"Precisione Monetaria per Immagine (Totale: {money['total_images']} img)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Percentuale Immagini (%)", fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 110)
    ax4.set_xticks(range(len(tol_labels)))
    ax4.set_xticklabels(tol_labels, rotation=15, ha="right", fontsize=9)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Panoramica Completa delle Prestazioni della Pipeline (Flow B Target Set)", fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Stampa un resoconto testuale formattato ed esaustivo delle metriche."""
    det = metrics["detection_metrics"]
    clf = metrics["classification_metrics"]
    sub = metrics["subpipeline_breakdown"]
    money = metrics["monetary_metrics"]

    print("=" * 78)
    print("           MODULO 9: REPORT DI VALUTAZIONE E ACCURATEZZA PIPELINE")
    print("=" * 78)

    print("\n1. METRICHE DI DETECTION (HOUGH CIRCLES vs GROUND TRUTH)")
    print(f"   - Monete Reali Totali (GT):        {det['total_gt_coins']}")
    print(f"   - Cerchi Rilevati dalla Pipeline:  {det['total_detected_circles']}")
    print(f"   - True Positives (Corrispondenti): {det['true_positives']} / {det['total_gt_coins']} ({det['recall']*100:.2f}%)")
    print(f"   - False Positives (Cerchi extra):  {det['false_positives']}")
    print(f"   - False Negatives (Monete perse):  {det['false_negatives']}")
    print(f"   - Precision Detection:             {det['precision']*100:.2f}%")
    print(f"   - Recall Detection:                {det['recall']*100:.2f}%")
    print(f"   - F1-Score Detection:              {det['f1_score']*100:.2f}%")

    print("\n2. METRICHE DI CLASSIFICAZIONE GLOBALE")
    print(f"   - Accuratezza sulle Monete Rilevate:   {clf['overall_accuracy_on_detected']*100:.2f}% ({det['true_positives']} monete)")
    print(f"   - Accuratezza Pipeline su tutto il GT: {clf['overall_accuracy_on_all_gt']*100:.2f}% ({det['total_gt_coins']} monete)")
    print(f"   - Accuratezza a livello Macro-Colore:  {clf['color_kind_accuracy']*100:.2f}%")
    print(f"   - Macro Average F1-Score:              {clf['macro_f1']*100:.2f}%")
    print(f"   - Weighted Average F1-Score:           {clf['weighted_f1']*100:.2f}%")

    print("\n3. PRESTAZIONI PER DENOMINAZIONE (PRECISION, RECALL, F1-SCORE)")
    print("   " + "-" * 72)
    print(f"   {'Denominazione':<14} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("   " + "-" * 72)
    for c in ORDERED_CLASSES:
        m = clf["per_class"][c]
        print(f"   {c:<14} | {m['support']:<8} | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | {m['f1_score']*100:>8.1f}%")
    print("   " + "-" * 72)

    print("\n4. RANKING DI DIFFICOLTA' (DALLE PIU' FACILI ALLE PIU' DIFFICILI)")
    print(f"   [+] MONETE PIU' FACILI:    {', '.join(clf['easiest_coins'])}")
    print(f"   [!] MONETE PIU' DIFFICILI: {', '.join(clf['hardest_coins'])}")

    print("\n5. CONFRONTO SOTTORAMI DELLA PIPELINE")
    print(f"   - Moneta Singola (Mod 7.1 Polar ZNCC): {sub['single_coin']['accuracy']*100:.2f}% ({sub['single_coin']['correct']}/{sub['single_coin']['total_matched']})")
    print(f"   - Monete Multiple (Mod 7.2 Scale CSP): {sub['multi_coin']['accuracy']*100:.2f}% ({sub['multi_coin']['correct']}/{sub['multi_coin']['total_matched']})")

    print("\n6. CONTEGGIO MONETARIO TOTALE ED ERRORI")
    print(f"   - Valore Reale Ground Truth:     {money['total_ground_truth_eur']:.2f} EUR")
    print(f"   - Valore Predetto Pipeline:      {money['total_predicted_eur']:.2f} EUR")
    print(f"   - Differenza Complessiva:        {money['total_difference_eur']:+.2f} EUR (Errore relativo: {money['relative_error_pct']}%)")
    print(f"   - Errore Medio Assoluto (MAE):   {money['mean_absolute_error_eur_per_image']:.2f} EUR per immagine")
    print(f"   - Immagini con Conteggio Esatto: {money['exact_matches_count']} / {money['total_images']} ({money['exact_matches_pct']}%)")
    print(f"   - Immagini con Errore <= 0.20 EUR: {money['within_20cent_pct']}%")
    print("=" * 78)


def run_evaluation(
    ground_truth_path: Optional[Union[str, Path]] = None,
    annotated_images: Optional[Dict[str, Any]] = None,
    annotated_images_path: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True,
    show_plots: bool = True
) -> Dict[str, Any]:
    """Funzione principale del Modulo 9.

    Esegue il caricamento, il matching con ground truth, il calcolo delle metriche,
    il salvataggio degli artifacts e il rendering dei grafici.
    """
    config = load_config()

    # Risoluzione percorsi
    gt_cfg = config.get("ground_truth_csv")
    gt_path = resolve_project_path(ground_truth_path) if ground_truth_path else (resolve_project_path(gt_cfg) if gt_cfg else get_default_ground_truth_path())
    df_gt = load_ground_truth(gt_path)

    if annotated_images is None:
        det_cfg = config.get("annotated_images_path")
        det_path = resolve_project_path(annotated_images_path) if annotated_images_path else (resolve_project_path(det_cfg) if det_cfg else get_default_annotated_images_path())
        annotated_images = load_detections(det_path)

    art_cfg = config.get("artifacts_dir")
    out_dir = resolve_project_path(artifacts_dir) if artifacts_dir else (resolve_project_path(art_cfg) if art_cfg else get_default_artifacts_dir())
    if save_artifacts and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Matching Bipartito
    max_dist = float(config.get("max_distance_px", 50.0))
    rad_factor = float(config.get("matching_distance_threshold_factor", 1.0))
    matched_pairs, unmatched_det, unmatched_gt = match_detections_with_ground_truth(
        df_gt=df_gt,
        annotated_images=annotated_images,
        max_distance_px=max_dist,
        radius_factor=rad_factor
    )

    # 2. Calcolo Metriche
    classes = config.get("classes", ORDERED_CLASSES)
    metrics = compute_metrics(
        df_gt=df_gt,
        annotated_images=annotated_images,
        matched_pairs=matched_pairs,
        unmatched_detections=unmatched_det,
        unmatched_ground_truths=unmatched_gt,
        classes=classes
    )

    # 3. Generazione e Salvataggio Grafici
    easiest_hardest_path = out_dir / "easiest_hardest_coins.png" if save_artifacts else None
    cm_path = out_dir / "confusion_matrix.png" if save_artifacts else None
    overview_path = out_dir / "performance_overview.png" if save_artifacts else None

    plot_easiest_hardest_coins(metrics, save_path=easiest_hardest_path, show_plot=show_plots)
    plot_confusion_matrix_heatmap(metrics, save_path=cm_path, show_plot=show_plots)
    plot_performance_overview_dashboard(metrics, save_path=overview_path, show_plot=show_plots)

    # 4. Salvataggio JSON di sintesi
    if save_artifacts:
        summary_path = out_dir / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 5. Stampa resoconto sintetico
    print_evaluation_summary(metrics)

    return metrics


if __name__ == "__main__":
    # Esecuzione standalone di test
    run_evaluation(show_plots=False)
