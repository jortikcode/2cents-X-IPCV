"""Modulo di supporto per il Tuning Interattivo della Ground Truth (gt.csv).

Fornisce l'interfaccia a widget (ipywidgets) per:
1. Navigare tra tutte le immagini del target_set.
2. Selezionare le singole monete annotate in ciascuna immagine.
3. Regolare interattivamente cx, cy e raggio con slider e pulsanti di precisione (+/- 1px).
4. Visualizzare in tempo reale sia la panoramica completa che il crop zoomato ad alta definizione.
5. Salvare le correzioni direttamente nel file pipeline/conf/gt.csv sovrascrivendo la riga corrispondente.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
import ipywidgets as widgets


DENOM_MAP = {
    1: "1 cent",
    2: "2 cent",
    5: "5 cent",
    10: "10 cent",
    20: "20 cent",
    50: "50 cent",
    100: "1 euro",
    200: "2 euro"
}

DENOM_OPTIONS = [
    ("1 cent (1c)", 1),
    ("2 cent (2c)", 2),
    ("5 cent (5c)", 5),
    ("10 cent (10c)", 10),
    ("20 cent (20c)", 20),
    ("50 cent (50c)", 50),
    ("1 euro (100c)", 100),
    ("2 euro (200c)", 200)
]


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """Chiave per ordinamento naturale alfanumerico (es. image_2 prima di image_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def find_project_root() -> Path:
    """Risolve dinamicamente la root del progetto."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "coin_dataset" / "target_set").exists():
            return parent
    return Path.cwd().resolve()


def get_default_gt_path() -> Path:
    """Restituisce il percorso standard del file gt.csv."""
    root = find_project_root()
    return root / "pipeline" / "conf" / "gt.csv"


def get_default_target_dir() -> Path:
    """Restituisce la cartella con le immagini target originali."""
    root = find_project_root()
    return root / "coin_dataset" / "target_set"


def create_gt_tuning_widget(
    gt_csv_path: Optional[Union[str, Path]] = None,
    target_dir: Optional[Union[str, Path]] = None
) -> widgets.Widget:
    """
    Crea l'interfaccia interattiva a widget per il tuning fine della Ground Truth.
    """
    csv_path = Path(gt_csv_path) if gt_csv_path else get_default_gt_path()
    img_dir = Path(target_dir) if target_dir else get_default_target_dir()

    if not csv_path.exists():
        raise FileNotFoundError(f"File ground truth non trovato: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Cartella target images non trovata: {img_dir}")

    # Carica la ground truth
    df = pd.read_csv(csv_path)
    required_cols = {"image", "cx", "cy", "radius", "denomination_cent"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Il file {csv_path} deve contenere le colonne: {required_cols}")

    # Cache immagini caricate per rendering reattivo
    image_cache: Dict[str, np.ndarray] = {}

    def get_image_rgb(img_name: str) -> Optional[np.ndarray]:
        if img_name not in image_cache:
            p = img_dir / img_name
            if not p.exists():
                return None
            bgr = cv2.imread(str(p))
            if bgr is None:
                return None
            image_cache[img_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return image_cache[img_name]

    # Lista ordinata univoca delle immagini
    unique_images = sorted(df["image"].unique(), key=natural_sort_key)
    img_counts = df["image"].value_counts().to_dict()

    image_dropdown_options = [
        (f"{img} ({img_counts[img]} {'moneta' if img_counts[img] == 1 else 'monete'})", img)
        for img in unique_images
    ]

    # --- Widget di Controllo ---
    dropdown_image = widgets.Dropdown(
        options=image_dropdown_options,
        value=unique_images[0],
        description="Immagine:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="340px")
    )

    btn_prev_img = widgets.Button(
        description="◀ Prec",
        tooltip="Immagine precedente",
        layout=widgets.Layout(width="70px")
    )
    btn_next_img = widgets.Button(
        description="Succ ▶",
        tooltip="Immagine successiva",
        layout=widgets.Layout(width="70px")
    )

    dropdown_coin = widgets.Dropdown(
        description="Moneta:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="490px")
    )

    slider_cx = widgets.IntSlider(
        description="Centro X (cx):",
        min=0, max=1000, value=200, step=1,
        continuous_update=True,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="420px")
    )

    slider_cy = widgets.IntSlider(
        description="Centro Y (cy):",
        min=0, max=1000, value=200, step=1,
        continuous_update=True,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="420px")
    )

    slider_radius = widgets.IntSlider(
        description="Raggio (r):",
        min=10, max=250, value=70, step=1,
        continuous_update=True,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="420px")
    )

    dropdown_denom = widgets.Dropdown(
        options=DENOM_OPTIONS,
        value=1,
        description="Denominazione:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="300px")
    )

    # Pulsanti di micro-aggiustamento di precisione (+/- 1 px e +/- 5 px)
    def make_step_btn(desc: str, delta: int, target_slider: widgets.IntSlider) -> widgets.Button:
        btn = widgets.Button(description=desc, layout=widgets.Layout(width="55px", height="28px"))
        def on_click(_):
            target_slider.value = max(target_slider.min, min(target_slider.max, target_slider.value + delta))
        btn.on_click(on_click)
        return btn

    btn_cx_m5 = make_step_btn("-5", -5, slider_cx)
    btn_cx_m1 = make_step_btn("-1", -1, slider_cx)
    btn_cx_p1 = make_step_btn("+1", +1, slider_cx)
    btn_cx_p5 = make_step_btn("+5", +5, slider_cx)

    btn_cy_m5 = make_step_btn("-5", -5, slider_cy)
    btn_cy_m1 = make_step_btn("-1", -1, slider_cy)
    btn_cy_p1 = make_step_btn("+1", +1, slider_cy)
    btn_cy_p5 = make_step_btn("+5", +5, slider_cy)

    btn_r_m5 = make_step_btn("-5", -5, slider_radius)
    btn_r_m1 = make_step_btn("-1", -1, slider_radius)
    btn_r_p1 = make_step_btn("+1", +1, slider_radius)
    btn_r_p5 = make_step_btn("+5", +5, slider_radius)

    btn_save = widgets.Button(
        description="💾 Salva Modifiche su gt.csv",
        button_style="success",
        icon="save",
        layout=widgets.Layout(width="240px", height="36px", font_weight="bold")
    )

    btn_reset = widgets.Button(
        description="↺ Ripristina Valori Riga",
        button_style="warning",
        icon="undo",
        layout=widgets.Layout(width="180px", height="36px")
    )

    out_status = widgets.HTML(
        value='<div style="padding: 6px; color: #495057; font-size: 13px;">'
              '<i>Seleziona un\'immagine e regola i cursori. Clicca su <b>Salva Modifiche</b> per scrivere su gt.csv.</i></div>'
    )

    out_plot = widgets.Output()

    # Stato interno per evitare chiamate ridondanti durante gli update
    is_updating_controls = False

    def get_coin_options_for_image(img_name: str) -> List[Tuple[str, int]]:
        sub = df[df["image"] == img_name]
        opts = []
        for idx, row in sub.iterrows():
            d_name = DENOM_MAP.get(int(row["denomination_cent"]), f"{row['denomination_cent']}c")
            label = f"Moneta #{len(opts) + 1}: {d_name} [cx={row['cx']}, cy={row['cy']}, r={row['radius']}] (riga {idx})"
            opts.append((label, int(idx)))
        return opts

    def sync_sliders_to_row(row_idx: int):
        nonlocal is_updating_controls
        is_updating_controls = True
        row = df.loc[row_idx]
        cur_cx = int(row["cx"])
        cur_cy = int(row["cy"])
        cur_r = int(row["radius"])
        cur_denom = int(row["denomination_cent"])

        img_rgb = get_image_rgb(row["image"])
        h, w = (960, 720) if img_rgb is None else img_rgb.shape[:2]

        # Range centrato attorno al valore corrente con margine confortevole
        slider_cx.min = max(0, cur_cx - 150)
        slider_cx.max = min(w, cur_cx + 150)
        slider_cx.value = cur_cx

        slider_cy.min = max(0, cur_cy - 150)
        slider_cy.max = min(h, cur_cy + 150)
        slider_cy.value = cur_cy

        slider_radius.min = max(5, cur_r - 60)
        slider_radius.max = cur_r + 60
        slider_radius.value = cur_r

        if cur_denom in [val for _, val in DENOM_OPTIONS]:
            dropdown_denom.value = cur_denom

        is_updating_controls = False

    def render_plot():
        with out_plot:
            out_plot.clear_output(wait=True)
            img_name = dropdown_image.value
            row_idx = dropdown_coin.value
            if row_idx is None or row_idx not in df.index:
                print("Seleziona una moneta valida.")
                return

            img_rgb = get_image_rgb(img_name)
            if img_rgb is None:
                print(f"Immagine non trovata su disco: {img_name}")
                return

            h, w = img_rgb.shape[:2]
            cx = slider_cx.value
            cy = slider_cy.value
            r = slider_radius.value
            denom_val = dropdown_denom.value
            denom_str = DENOM_MAP.get(denom_val, f"{denom_val}c")

            fig, (ax_full, ax_zoom) = plt.subplots(
                1, 2, figsize=(14, 7),
                gridspec_kw={"width_ratios": [1.2, 1.0]}
            )

            # --- 1. Panoramica Immagine Intera ---
            ax_full.imshow(img_rgb)
            ax_full.set_title(f"Panoramica: {img_name} (W={w}, H={h})", fontsize=12, fontweight="bold")

            # Disegna tutte le altre monete dell'immagine (inattive)
            other_coins = df[(df["image"] == img_name) & (df.index != row_idx)]
            for o_idx, o_row in other_coins.iterrows():
                o_cx, o_cy, o_r = float(o_row["cx"]), float(o_row["cy"]), float(o_row["radius"])
                o_denom = DENOM_MAP.get(int(o_row["denomination_cent"]), f"{o_row['denomination_cent']}c")
                circ_other = patches.Circle(
                    (o_cx, o_cy), o_r,
                    fill=False, color="cyan", linewidth=1.5, linestyle="--"
                )
                ax_full.add_patch(circ_other)
                ax_full.plot(o_cx, o_cy, marker="+", color="cyan", markersize=8)
                ax_full.text(
                    o_cx, o_cy - o_r - 6, o_denom,
                    color="cyan", fontsize=9, fontweight="bold",
                    ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5)
                )

            # Disegna la moneta ATTIVA (in evidenza)
            circ_active = patches.Circle(
                (cx, cy), r,
                fill=False, color="#00FF00", linewidth=2.5
            )
            ax_full.add_patch(circ_active)
            ax_full.plot(cx, cy, marker="+", color="red", markersize=12, markeredgewidth=2)
            ax_full.text(
                cx, cy - r - 8, f"ATTIVA: {denom_str}",
                color="#00FF00", fontsize=11, fontweight="bold",
                ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
            )

            # Bounding box del crop zoomato sulla panoramica
            crop_margin = int(r * 1.5)
            x1 = max(0, cx - crop_margin)
            y1 = max(0, cy - crop_margin)
            x2 = min(w, cx + crop_margin)
            y2 = min(h, cy + crop_margin)
            rect_zoom = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color="yellow", linewidth=1.5, linestyle=":"
            )
            ax_full.add_patch(rect_zoom)
            ax_full.axis("off")

            # --- 2. Zoom Pixel-Perfect sulla Moneta Attiva ---
            crop_img = img_rgb[y1:y2, x1:x2]
            ax_zoom.imshow(crop_img)
            ax_zoom.set_title(f"Zoom Moneta Attiva: cx={cx}, cy={cy}, r={r} px", fontsize=12, fontweight="bold")

            # Coordinate relative nel crop
            local_cx = cx - x1
            local_cy = cy - y1

            # Cerchio raggio corrente
            circ_zoom = patches.Circle(
                (local_cx, local_cy), r,
                fill=False, color="#00FF00", linewidth=2.0
            )
            ax_zoom.add_patch(circ_zoom)

            # Cerchi sottili di guida r +/- 2 px per verificare aderenza perfetta al bordo
            circ_zoom_inner = patches.Circle((local_cx, local_cy), max(1, r - 2), fill=False, color="#00FF00", linewidth=0.8, linestyle=":")
            circ_zoom_outer = patches.Circle((local_cx, local_cy), r + 2, fill=False, color="#00FF00", linewidth=0.8, linestyle=":")
            ax_zoom.add_patch(circ_zoom_inner)
            ax_zoom.add_patch(circ_zoom_outer)

            # Centro croce
            ax_zoom.plot(local_cx, local_cy, marker="+", color="red", markersize=14, markeredgewidth=2)
            ax_zoom.grid(color="white", alpha=0.2, linestyle="--", linewidth=0.5)

            plt.tight_layout()
            display(fig)
            plt.close(fig)

    # --- Callbacks ---
    def on_image_change(change):
        nonlocal is_updating_controls
        img_name = change["new"]
        options = get_coin_options_for_image(img_name)
        is_updating_controls = True
        dropdown_coin.options = options
        if options:
            dropdown_coin.value = options[0][1]
            sync_sliders_to_row(options[0][1])
        is_updating_controls = False
        render_plot()

    def on_coin_change(change):
        if is_updating_controls or change["new"] is None:
            return
        sync_sliders_to_row(change["new"])
        render_plot()

    def on_slider_change(change):
        if is_updating_controls:
            return
        render_plot()

    def on_prev_clicked(_):
        idx = unique_images.index(dropdown_image.value)
        if idx > 0:
            dropdown_image.value = unique_images[idx - 1]

    def on_next_clicked(_):
        idx = unique_images.index(dropdown_image.value)
        if idx < len(unique_images) - 1:
            dropdown_image.value = unique_images[idx + 1]

    def on_save_clicked(_):
        row_idx = dropdown_coin.value
        if row_idx is None:
            return
        new_cx = int(slider_cx.value)
        new_cy = int(slider_cy.value)
        new_r = int(slider_radius.value)
        new_denom = int(dropdown_denom.value)
        img_name = dropdown_image.value

        # Aggiornamento in-place del DataFrame
        df.loc[row_idx, "cx"] = new_cx
        df.loc[row_idx, "cy"] = new_cy
        df.loc[row_idx, "radius"] = new_r
        df.loc[row_idx, "denomination_cent"] = new_denom

        # Scrittura su gt.csv
        try:
            df.to_csv(csv_path, index=False)
            d_name = DENOM_MAP.get(new_denom, f"{new_denom}c")
            out_status.value = (
                f'<div style="background-color: #d4edda; color: #155724; padding: 8px 12px; border-radius: 4px; font-weight: bold; border: 1px solid #c3e6cb;">'
                f'✅ Salvato con successo in {csv_path.name}! [Riga {row_idx}: {img_name} -> cx={new_cx}, cy={new_cy}, r={new_r}, {d_name}]'
                f'</div>'
            )
            # Aggiorna il testo dell'opzione nel dropdown moneta
            curr_opts = get_coin_options_for_image(img_name)
            dropdown_coin.options = curr_opts
            dropdown_coin.value = row_idx
        except Exception as e:
            out_status.value = (
                f'<div style="background-color: #f8d7da; color: #721c24; padding: 8px 12px; border-radius: 4px; font-weight: bold; border: 1px solid #f5c6cb;">'
                f'❌ Errore durante il salvataggio: {e}'
                f'</div>'
            )

    def on_reset_clicked(_):
        row_idx = dropdown_coin.value
        if row_idx is None:
            return
        sync_sliders_to_row(row_idx)
        out_status.value = (
            f'<div style="background-color: #fff3cd; color: #856404; padding: 6px 10px; border-radius: 4px;">'
            f'↺ Cursori ripristinati ai valori attuali di gt.csv per la riga {row_idx}.'
            f'</div>'
        )
        render_plot()

    # Collega gli eventi
    dropdown_image.observe(on_image_change, names="value")
    dropdown_coin.observe(on_coin_change, names="value")
    slider_cx.observe(on_slider_change, names="value")
    slider_cy.observe(on_slider_change, names="value")
    slider_radius.observe(on_slider_change, names="value")
    dropdown_denom.observe(on_slider_change, names="value")

    btn_prev_img.on_click(on_prev_clicked)
    btn_next_img.on_click(on_next_clicked)
    btn_save.on_click(on_save_clicked)
    btn_reset.on_click(on_reset_clicked)

    # Inizializzazione controlli alla prima moneta della prima immagine
    first_img = unique_images[0]
    initial_options = get_coin_options_for_image(first_img)
    dropdown_coin.options = initial_options
    if initial_options:
        dropdown_coin.value = initial_options[0][1]
        sync_sliders_to_row(initial_options[0][1])

    # --- Layout dell'Interfaccia ---
    row_nav = widgets.HBox([
        dropdown_image,
        btn_prev_img,
        btn_next_img,
        dropdown_coin
    ], layout=widgets.Layout(margin="0 0 10px 0"))

    row_cx = widgets.HBox([slider_cx, btn_cx_m5, btn_cx_m1, btn_cx_p1, btn_cx_p5])
    row_cy = widgets.HBox([slider_cy, btn_cy_m5, btn_cy_m1, btn_cy_p1, btn_cy_p5])
    row_r = widgets.HBox([slider_radius, btn_r_m5, btn_r_m1, btn_r_p1, btn_r_p5])

    row_actions = widgets.HBox([
        dropdown_denom,
        btn_save,
        btn_reset
    ], layout=widgets.Layout(margin="10px 0 10px 0", align_items="center"))

    controls_box = widgets.VBox([
        row_nav,
        row_cx,
        row_cy,
        row_r,
        row_actions,
        out_status
    ], layout=widgets.Layout(
        padding="15px",
        border="1px solid #ced4da",
        border_radius="8px",
        background_color="#f8f9fa",
        margin="0 0 15px 0"
    ))

    # Render iniziale
    render_plot()

    return widgets.VBox([controls_box, out_plot])
