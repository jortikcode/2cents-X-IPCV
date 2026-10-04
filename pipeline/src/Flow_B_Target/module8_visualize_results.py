"""Modulo 8 (Flow B): Visualizzazione Finale dei Risultati e Conteggio Monetario.

Questo modulo:
1. Carica le immagini target e i dati di classificazione finale da 'annotated_images.json'.
2. Per ciascuna immagine target:
   - Sovrappone un quadrato (bounding box) attorno a ciascuna moneta rilevata.
   - Posiziona sopra il quadrato una label chiara con la denominazione ('class') assegnata:
     (1cent, 2cent, 5cent, 10cent, 20cent, 50cent, 1euro, 2euro).
   - Calcola e visualizza il valore monetario complessivo dell'immagine in Euro (€).
3. Salva tutte le immagini elaborate nella cartella artifacts:
   'pipeline/artifacts/Flow_B_Target/module8_visualize_results/{nome_immagine}.png'.
4. Salva un file JSON di riepilogo del conteggio finale:
   'final_counting_summary.json'.
5. Fornisce la funzione per visualizzare a schermo un subsample di immagini annotate nel notebook.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Mappa dei valori monetari in Euro per ciascuna denominazione
VALUE_MAP: Dict[str, float] = {
    "1cent": 0.01,
    "2cent": 0.02,
    "5cent": 0.05,
    "10cent": 0.10,
    "20cent": 0.20,
    "50cent": 0.50,
    "1euro": 1.00,
    "2euro": 2.00
}

# Mappa colori per denominazione (formato BGR)
COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "1euro": (0, 215, 255),    # Giallo / Oro brillante
    "2euro": (255, 190, 30),   # Argento / Azzurro acceso
    "10cent": (0, 215, 255),   # Giallo Oro
    "20cent": (0, 215, 255),
    "50cent": (0, 215, 255),
    "1cent": (40, 120, 235),   # Rame / Arancio brillante
    "2cent": (40, 120, 235),
    "5cent": (40, 120, 235)
}


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso della cartella artifacts/Flow_B_Target/module8_visualize_results."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module8_visualize_results"


def get_default_annotated_images_path() -> Path:
    """Restituisce il percorso del file annotated_images.json principale."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module3_coin_detection" / "annotated_images.json"


def get_default_denoised_dir() -> Path:
    """Restituisce la cartella con le immagini denoised a colori prodotte dal Modulo 4."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_B_Target" / "module4_denoise_colored"


def get_default_target_dir() -> Path:
    """Restituisce il percorso originale di coin_dataset/target_set."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "coin_dataset" / "target_set"


def draw_coin_box_and_label(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    denomination: str
) -> None:
    """
    Disegna un quadrato (bounding box) attorno alla moneta e un'etichetta
    in sovraimpressione con il nome della denominazione.
    """
    H, W = img.shape[:2]

    # Coordinate del quadrato circoscritto alla moneta
    x1 = max(0, int(round(cx - radius)))
    y1 = max(0, int(round(cy - radius)))
    x2 = min(W - 1, int(round(cx + radius)))
    y2 = min(H - 1, int(round(cy + radius)))

    box_color = COLOR_MAP.get(denomination, (0, 255, 0))

    # 1. Bounding box quadrato attorno alla moneta
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Etichetta con la denominazione
    label_text = denomination.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Posizionamento del badge dell'etichetta sopra il quadrato (o dentro se troppo vicina al bordo superiore)
    if y1 - th - 8 >= 0:
        badge_y1 = y1 - th - 8
        badge_y2 = y1
        text_y = y1 - 4
    else:
        badge_y1 = y1
        badge_y2 = y1 + th + 8
        text_y = y1 + th + 4

    badge_x1 = x1
    badge_x2 = min(W - 1, x1 + tw + 10)
    text_x = x1 + 5

    # Sfondo dell'etichetta (badge semitrasparente/scuro per massima leggibilità)
    overlay = img.copy()
    cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    # Bordo attorno al badge dell'etichetta
    cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 1, lineType=cv2.LINE_AA)

    # Testo dell'etichetta
    cv2.putText(img, label_text, (text_x, text_y), font, font_scale, box_color, thickness, lineType=cv2.LINE_AA)


def draw_image_header(
    img: np.ndarray,
    img_name: str,
    coins: List[Dict[str, Any]]
) -> float:
    """
    Disegna un banner riassuntivo in alto a sinistra con il nome dell'immagine,
    il numero di monete e il totale monetario calcolato in Euro (€).
    
    Returns:
        float: Totale monetario dell'immagine in Euro.
    """
    total_val = sum(VALUE_MAP.get(c.get("class", ""), 0.0) for c in coins)
    num_coins = len(coins)

    header_text = f"{img_name} | {num_coins} monete | Valore: {total_val:.2f} EUR"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.60
    thickness = 2
    (tw, th), _ = cv2.getTextSize(header_text, font, font_scale, thickness)

    # Banner scuro in alto a sinistra
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (tw + 26, th + 22), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.rectangle(img, (8, 8), (tw + 26, th + 22), (100, 100, 100), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, header_text, (16, th + 14), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return round(total_val, 2)


def visualize_and_save_all_results(
    images: Optional[Dict[str, np.ndarray]] = None,
    annotated_images: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True
) -> Dict[str, np.ndarray]:
    """
    Prende le immagini target e sovrappone i quadrati attorno alle monete
    con sopra la loro denominazione, basandosi sui dati del JSON dei risultati.
    Salva tutte le immagini elaborate nella cartella artifacts e genera il report finale.
    
    Returns:
        Dict[str, np.ndarray]: Dizionario {nome_immagine: immagine_annotata_bgr}.
    """
    # 1. Caricamento del JSON dei risultati se non passato
    if annotated_images is None:
        ann_path = get_default_annotated_images_path()
        if not ann_path.exists():
            raise FileNotFoundError(f"File annotazioni non trovato: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            annotated_images = json.load(f)

    # 2. Cartella di destinazione artifacts
    if output_dir is not None:
        out_path = Path(output_dir)
        if not out_path.is_absolute() and not out_path.exists():
            proj_root = Path(__file__).resolve().parents[3]
            if (proj_root / out_path).parent.exists():
                out_path = proj_root / out_path
    else:
        out_path = get_default_artifacts_dir()

    if save_artifacts:
        out_path.mkdir(parents=True, exist_ok=True)

    denoised_dir = get_default_denoised_dir()
    target_dir = get_default_target_dir()

    annotated_results: Dict[str, np.ndarray] = {}
    image_summaries: Dict[str, Any] = {}
    grand_total_value = 0.0
    denomination_counts: Dict[str, int] = {}
    total_coins_count = 0

    total_images = len(annotated_images)
    print(f"Avvio rendering dei risultati finali su {total_images} immagini target...")

    for idx, (img_name, data) in enumerate(annotated_images.items(), start=1):
        # Recupero dell'immagine originale/denoised
        if images is not None and img_name in images:
            img = images[img_name].copy()
        else:
            denoised_file = denoised_dir / f"{img_name}.png"
            if denoised_file.exists():
                img = cv2.imread(str(denoised_file))
            else:
                raw_file = target_dir / f"{img_name}.jpg"
                if not raw_file.exists():
                    raw_file = target_dir / f"{img_name}.png"
                img = cv2.imread(str(raw_file))
                # Se caricata grezza, correggi inversione B/R
                if img is not None and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            print(f"[Warning] Impossibile caricare {img_name}. Skip.")
            continue

        annotated_img = img.copy()
        coins = data.get("coins", [])

        # Disegna i quadrati e le label per ciascuna moneta
        for coin in coins:
            cx, cy = float(coin["center"][0]), float(coin["center"][1])
            r = float(coin["radius"])
            denomination = coin.get("class", "unknown")

            draw_coin_box_and_label(annotated_img, cx, cy, r, denomination)

            denomination_counts[denomination] = denomination_counts.get(denomination, 0) + 1
            total_coins_count += 1

        # Disegna il banner riassuntivo con il conteggio monetario
        img_val = draw_image_header(annotated_img, img_name, coins)
        grand_total_value += img_val

        annotated_results[img_name] = annotated_img

        image_summaries[img_name] = {
            "coins_count": len(coins),
            "coins": [
                {
                    "coin_id": c.get("coin_id"),
                    "class": c.get("class"),
                    "value_eur": VALUE_MAP.get(c.get("class", ""), 0.0),
                    "center": c.get("center"),
                    "radius": c.get("radius")
                }
                for c in coins
            ],
            "total_value_eur": img_val
        }

        # Salvataggio dell'immagine finale negli artifacts
        if save_artifacts:
            out_file = out_path / f"{img_name}.png"
            cv2.imwrite(str(out_file), annotated_img)

        if idx % 30 == 0 or idx == total_images:
            print(f"  [{idx}/{total_images}] Immagini annotate elaborate...")

    # Salvataggio del report di conteggio monetario aggregato
    if save_artifacts:
        report = {
            "total_images": total_images,
            "total_coins": total_coins_count,
            "grand_total_value_eur": round(grand_total_value, 2),
            "denomination_counts": denomination_counts,
            "images": image_summaries
        }
        with open(out_path / "final_counting_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSalvate {len(annotated_results)} immagini finali in: {out_path.name}/")
        print(f"Salvato report monetario in: {out_path.name}/final_counting_summary.json")

    print("\n==============================================")
    print("      RIEPILOGO CONTEGGIO FINALE DATASET      ")
    print("==============================================")
    print(f"Totale Immagini:        {total_images}")
    print(f"Totale Monete Rilevate: {total_coins_count}")
    print(f"VALORE TOTALE DATASET:  {grand_total_value:.2f} EUR\n")
    print("Distribuzione per Denominazione:")
    for denom, count in sorted(denomination_counts.items(), key=lambda x: -x[1]):
        val = VALUE_MAP.get(denom, 0.0) * count
        pct = (count / total_coins_count * 100) if total_coins_count > 0 else 0
        print(f"  - {denom:7s}: {count:3d} monete ({pct:5.1f}%) -> {val:6.2f} EUR")
    print("==============================================")

    return annotated_results


def plot_results_sample(
    annotated_results: Dict[str, np.ndarray],
    n_samples: int = 8,
    seed: Optional[int] = None,
    figsize: tuple = (16, 9)
):
    """
    Mostra a schermo una griglia con un campione casuale di immagini finali
    con quadrati e denominazioni sovrapposte.
    """
    if not annotated_results:
        print("Nessuna immagine annotata disponibile per la visualizzazione.")
        return

    if seed is not None:
        random.seed(seed)

    keys = list(annotated_results.keys())
    sample_keys = random.sample(keys, min(n_samples, len(keys)))

    ncols = 4
    nrows = int(np.ceil(len(sample_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, key in enumerate(sample_keys):
        ax = axes[idx]
        img_bgr = annotated_results[key]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_title(f"{key}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(len(sample_keys), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Modulo 8: Risultati Finali - Rilevamento, Classificazione e Conteggio ({len(sample_keys)} immagini)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    annotated = visualize_and_save_all_results(save_artifacts=True)
