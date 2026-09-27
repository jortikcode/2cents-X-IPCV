"""Modulo 4: Estrazione Edge delle Reference con Tuner Interattivo (ipywidgets).

Questo modulo:
1. Converte le 8 ROI BGR in Grayscale e le salva in artifacts/Flow_A_Reference/module4_rois_gray.
2. Fornisce un'interfaccia interattiva ipywidgets per tunare singolarmente per ogni moneta:
   - Flag CLAHE
   - Flag Bilateral Filter (con slider diametro e sigma)
   - Slider Canny T1 e T2
   - Bottone 'Save' per memorizzare i valori in memoria e serializzarli su JSON.
3. Estrae e salva gli 8 edge binari in artifacts/Flow_A_Reference/module4_edges.
4. Restituisce in memoria il dizionario {nome_moneta: edge_img} pronto per il Modulo 5.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import opzionale di ipywidgets e IPython per compatibilità sia interattiva che script
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


def get_default_artifacts_dir() -> Path:
    """Restituisce il percorso base della cartella artifacts/Flow_A_Reference."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "artifacts" / "Flow_A_Reference"


def get_default_conf_dir() -> Path:
    """Restituisce il percorso base della cartella conf/Flow_A_Reference."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    return pipeline_dir / "conf" / "Flow_A_Reference"


def convert_and_save_gray_rois(
    rois_bgr: Dict[str, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, np.ndarray]:
    """
    Converte tutte le 8 ROI BGR in Grayscale e le salva su disco.
    
    Args:
        rois_bgr: Dizionario {nome_moneta: roi_bgr} restituito dal Modulo 2.
        output_dir: Cartella di destinazione (default: artifacts/Flow_A_Reference/module4_rois_gray).
        
    Returns:
        Dict[str, np.ndarray]: Dizionario {nome_moneta: roi_gray}.
    """
    base_dir = get_default_artifacts_dir()
    out_path = Path(output_dir) if output_dir is not None else base_dir / "module4_rois_gray"
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    rois_gray = {}
    for name, roi in rois_bgr.items():
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        rois_gray[name] = gray
        filepath = out_path / f"{name}_gray.png"
        cv2.imwrite(str(filepath), gray)

    print(f"Salvate con successo {len(rois_gray)} ROI Grayscale in: {out_path}")
    return rois_gray


def get_default_edge_params() -> Dict[str, Dict[str, Any]]:
    """
    Restituisce i parametri di default calibrati per ciascuna delle 8 monete.
    Ogni moneta ha un proprio dizionario con parametri specifici.
    """
    return {
        "10cent": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 75,
            "canny_t1": 60,
            "canny_t2": 150
        },
        "1cent": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 80,
            "canny_t1": 70,
            "canny_t2": 160
        },
        "1euro": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 75,
            "canny_t1": 50,
            "canny_t2": 140
        },
        "20cent": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 70,
            "canny_t1": 60,
            "canny_t2": 150
        },
        "2cent": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 85,
            "canny_t1": 65,
            "canny_t2": 155
        },
        "2euro": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 75,
            "canny_t1": 50,
            "canny_t2": 140
        },
        "50cent": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 75,
            "canny_t1": 60,
            "canny_t2": 150
        },
        "5cent": {
            "use_clahe": True,
            "use_bilateral": True,
            "bilateral_d": 7,
            "bilateral_sigma": 80,
            "canny_t1": 65,
            "canny_t2": 155
        }
    }


def load_or_create_edge_params(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Carica i parametri dal file JSON se esiste, altrimenti inizializza con i default e crea il file.
    """
    if json_path is None:
        json_path = get_default_conf_dir() / "edge_params.json"
    else:
        json_path = Path(json_path)

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            return params
        except Exception as e:
            print(f"Avviso: Errore nella lettura di {json_path} ({e}). Caricamento parametri di default.")

    # Se non esiste, crea parametri di default e salva
    params = get_default_edge_params()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return params


def save_edge_params(params: Dict[str, Dict[str, Any]], json_path: Optional[Union[str, Path]] = None) -> Path:
    """Serializza il dizionario dei parametri nel file JSON."""
    if json_path is None:
        json_path = get_default_conf_dir() / "edge_params.json"
    else:
        json_path = Path(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return json_path


def apply_edge_pipeline(
    gray_roi: np.ndarray,
    params: Dict[str, Any],
    mask_radius: float = 100.0,
    cx: float = 125.0,
    cy: float = 125.0
) -> np.ndarray:
    """
    Applica la pipeline di estrazione edge ad una singola ROI in scala di grigi:
    1. CLAHE (opzionale se use_clahe=True)
    2. Bilateral Filter (opzionale se use_bilateral=True con diametro e sigma)
    3. Canny Edge Detector (thresholds canny_t1, canny_t2)
    4. Maschera circolare (azzera eventuali bordi esterni al cerchio normalizzato)
    """
    img = gray_roi.copy()

    # 1. CLAHE per esaltare i rilievi metallici
    if params.get("use_clahe", True):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # 2. Bilateral Filter per abbattere il rumore della superficie metallica preservando i contorni
    if params.get("use_bilateral", True):
        d = int(params.get("bilateral_d", 7))
        sigma = float(params.get("bilateral_sigma", 75))
        # d deve essere dispari o positivo
        if d > 0:
            img = cv2.bilateralFilter(img, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # 3. Canny Edge Detection
    t1 = int(params.get("canny_t1", 50))
    t2 = int(params.get("canny_t2", 150))
    edges = cv2.Canny(img, t1, t2)

    # 4. Azzeramento bordi fuori dal raggio della moneta
    h, w = edges.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    circle_mask = np.hypot(xs - cx, ys - cy) <= mask_radius
    edges[~circle_mask] = 0

    return edges


def extract_and_save_all_edges(
    rois_gray: Dict[str, np.ndarray],
    params: Dict[str, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    conf_path: Optional[Union[str, Path]] = None
) -> Dict[str, np.ndarray]:
    """
    Estrae gli edge per tutte le monete fornite usando i rispettivi parametri e salva le immagini su file.
    
    Args:
        rois_gray: Dizionario {nome_moneta: roi_gray}.
        params: Dizionario {nome_moneta: parametri_singoli}.
        output_dir: Cartella artifacts per salvare i file PNG (default: artifacts/.../module4_edges).
        conf_path: Percorso del file JSON di configurazione (default: conf/Flow_A_Reference/edge_params.json).
        
    Returns:
        Dict[str, np.ndarray]: Dizionario in memoria {nome_moneta: edge_img}.
    """
    base_dir = get_default_artifacts_dir()
    out_path = Path(output_dir) if output_dir is not None else base_dir / "module4_edges"
    if not out_path.is_absolute() and not out_path.exists():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Assicura il salvataggio dei parametri nel json di configurazione
    conf_file = save_edge_params(params, conf_path)

    edges_dict = {}
    for name, gray in rois_gray.items():
        coin_params = params.get(name, get_default_edge_params().get(name, {}))
        edges = apply_edge_pipeline(gray, coin_params)
        edges_dict[name] = edges

        edge_filepath = out_path / f"{name}_edges.png"
        cv2.imwrite(str(edge_filepath), edges)

    print(f"Salvati {len(edges_dict)} file edge PNG in: {out_path} e la configurazione in: {conf_file}")
    return edges_dict


def plot_all_edges(
    edges_dict: Dict[str, np.ndarray],
    figsize: tuple = (16, 8),
    save_plot: bool = True
):
    """Visualizza e facoltativamente salva una griglia 2x4 con tutti gli 8 edge estratti."""
    names = sorted(edges_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        edge_img = edges_dict[name]
        ax.imshow(edge_img, cmap="gray")
        ax.set_title(f"Reference: {name}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Modulo 4: Mappe degli Edge Estratte dalle 8 Reference", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_plot:
        artifacts_edges_dir = get_default_artifacts_dir() / "module4_edges"
        artifacts_edges_dir.mkdir(parents=True, exist_ok=True)
        plot_path = artifacts_edges_dir / "all_edges_summary.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Grafico riassuntivo degli edge salvato in: {plot_path}")

    plt.show()
    plt.close(fig)


def create_edge_tuner_widget(
    rois_gray: Dict[str, np.ndarray],
    params: Dict[str, Dict[str, Any]],
    on_save_callback: Optional[Callable[[Dict[str, Dict[str, Any]]], None]] = None,
    json_path: Optional[Union[str, Path]] = None
) -> Any:
    """
    Costruisce e restituisce l'interfaccia interattiva ipywidgets:
    - Dropdown per selezionare la moneta
    - Checkbox CLAHE
    - Checkbox Bilateral Filter + slider diametro e sigma
    - Slider Canny T1 e T2
    - Anteprima side-by-side: ROI Gray a sinistra, Edge risultante a destra
    - Bottone 'Salva Parametri': salva su file JSON e aggiorna lo stato in memoria
    """
    if not WIDGETS_AVAILABLE:
        print("Attenzione: ipywidgets non è installato. Il widget interattivo non è disponibile.")
        return None

    coin_names = sorted(list(rois_gray.keys()))
    if not coin_names:
        print("Nessuna moneta disponibile per il tuner.")
        return None

    # Inizializza con la prima moneta
    initial_coin = coin_names[0]
    initial_params = params.get(initial_coin, get_default_edge_params().get(initial_coin, {}))

    # --- Controlli Widget ---
    coin_dropdown = widgets.Dropdown(
        options=coin_names,
        value=initial_coin,
        description="Moneta:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="280px")
    )

    clahe_checkbox = widgets.Checkbox(
        value=initial_params.get("use_clahe", True),
        description="Abilita CLAHE",
        style={"description_width": "110px"}
    )

    bilateral_checkbox = widgets.Checkbox(
        value=initial_params.get("use_bilateral", True),
        description="Abilita Bilateral",
        style={"description_width": "110px"}
    )

    bilateral_d_slider = widgets.IntSlider(
        value=initial_params.get("bilateral_d", 7),
        min=1,
        max=15,
        step=2,
        continuous_update=False,
        description="Bilateral diam:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="380px")
    )

    bilateral_sigma_slider = widgets.IntSlider(
        value=initial_params.get("bilateral_sigma", 75),
        min=10,
        max=150,
        step=5,
        continuous_update=False,
        description="Bilateral sigma:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="380px")
    )

    canny_t1_slider = widgets.IntSlider(
        value=initial_params.get("canny_t1", 60),
        min=10,
        max=300,
        step=5,
        continuous_update=False,
        description="Canny T1 (low):",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="380px")
    )

    canny_t2_slider = widgets.IntSlider(
        value=initial_params.get("canny_t2", 150),
        min=20,
        max=400,
        step=5,
        continuous_update=False,
        description="Canny T2 (high):",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="380px")
    )

    save_btn = widgets.Button(
        description="Salva Parametri (JSON & Memoria)",
        button_style="success",
        icon="save",
        layout=widgets.Layout(width="320px", height="38px")
    )

    status_label = widgets.HTML(
        value="<i>Regola gli slider per visualizzare l'anteprima in tempo reale.</i>"
    )

    # Widget nativi Image (evitano qualsiasi leak di Matplotlib / IPython in VS Code)
    gray_img_widget = widgets.Image(
        format="png",
        width=250,
        height=250,
        layout=widgets.Layout(border="1px solid #ccc", border_radius="6px")
    )

    edges_img_widget = widgets.Image(
        format="png",
        width=250,
        height=250,
        layout=widgets.Layout(border="1px solid #ccc", border_radius="6px")
    )

    gray_title = widgets.HTML(
        f"<div style='text-align:center; font-weight:bold; margin-bottom:5px; font-size:13px;'>ROI Grayscale: {initial_coin}</div>"
    )
    edges_title = widgets.HTML(
        f"<div style='text-align:center; font-weight:bold; margin-bottom:5px; font-size:13px;'>Edges Canny (T1={initial_params.get('canny_t1', 60)}, T2={initial_params.get('canny_t2', 150)})</div>"
    )

    preview_box = widgets.HBox([
        widgets.VBox([gray_title, gray_img_widget], layout=widgets.Layout(align_items="center")),
        widgets.VBox([edges_title, edges_img_widget], layout=widgets.Layout(align_items="center", margin="0 0 0 25px"))
    ], layout=widgets.Layout(margin="15px 0 10px 0", justify_content="center"))

    # --- Funzione di Aggiornamento Grafico Anteprima ---
    def update_preview(*args):
        current_coin = coin_dropdown.value
        if current_coin not in rois_gray:
            return

        gray_roi = rois_gray[current_coin]
        current_p = {
            "use_clahe": clahe_checkbox.value,
            "use_bilateral": bilateral_checkbox.value,
            "bilateral_d": bilateral_d_slider.value,
            "bilateral_sigma": bilateral_sigma_slider.value,
            "canny_t1": canny_t1_slider.value,
            "canny_t2": canny_t2_slider.value
        }

        # Calcolo edge con parametri correnti
        edges = apply_edge_pipeline(gray_roi, current_p)

        # Disabilita/abilita slider bilaterale visivamente
        bilateral_d_slider.disabled = not bilateral_checkbox.value
        bilateral_sigma_slider.disabled = not bilateral_checkbox.value

        # Aggiornamento titoli e immagini PNG nei widget Image nativi (in memoria, < 1ms)
        gray_title.value = f"<div style='text-align:center; font-weight:bold; margin-bottom:5px; font-size:13px;'>ROI Grayscale: {current_coin}</div>"
        edges_title.value = f"<div style='text-align:center; font-weight:bold; margin-bottom:5px; font-size:13px;'>Edges Canny (T1={current_p['canny_t1']}, T2={current_p['canny_t2']})</div>"

        gray_img_widget.value = cv2.imencode(".png", gray_roi)[1].tobytes()
        edges_img_widget.value = cv2.imencode(".png", edges)[1].tobytes()

    # --- Gestione Cambio Moneta dal Dropdown ---
    def on_coin_change(change):
        new_coin = change["new"]
        p = params.get(new_coin, get_default_edge_params().get(new_coin, {}))

        # Disattiva temporaneamente i listener per evitare update a catena
        clahe_checkbox.unobserve(update_preview, names="value")
        bilateral_checkbox.unobserve(update_preview, names="value")
        bilateral_d_slider.unobserve(update_preview, names="value")
        bilateral_sigma_slider.unobserve(update_preview, names="value")
        canny_t1_slider.unobserve(update_preview, names="value")
        canny_t2_slider.unobserve(update_preview, names="value")

        clahe_checkbox.value = p.get("use_clahe", True)
        bilateral_checkbox.value = p.get("use_bilateral", True)
        bilateral_d_slider.value = p.get("bilateral_d", 7)
        bilateral_sigma_slider.value = p.get("bilateral_sigma", 75)
        canny_t1_slider.value = p.get("canny_t1", 60)
        canny_t2_slider.value = p.get("canny_t2", 150)

        clahe_checkbox.observe(update_preview, names="value")
        bilateral_checkbox.observe(update_preview, names="value")
        bilateral_d_slider.observe(update_preview, names="value")
        bilateral_sigma_slider.observe(update_preview, names="value")
        canny_t1_slider.observe(update_preview, names="value")
        canny_t2_slider.observe(update_preview, names="value")

        status_label.value = f"<i>Caricati parametri correnti per <b>{new_coin}</b>.</i>"
        update_preview()

    # --- Click su Bottone Salva ---
    def on_save_clicked(btn):
        current_coin = coin_dropdown.value
        params[current_coin] = {
            "use_clahe": clahe_checkbox.value,
            "use_bilateral": bilateral_checkbox.value,
            "bilateral_d": bilateral_d_slider.value,
            "bilateral_sigma": bilateral_sigma_slider.value,
            "canny_t1": canny_t1_slider.value,
            "canny_t2": canny_t2_slider.value
        }

        # 1. Salva file JSON su disco
        json_file = save_edge_params(params, json_path=json_path)

        # 2. Callback per aggiornamento in memoria e riscrittura edge in artifacts
        if on_save_callback is not None:
            on_save_callback(params)

        status_label.value = f"<b style='color:green;'>✓ Parametri salvati per {current_coin}! JSON e memoria aggiornati con successo ({json_file.name}).</b>"

    # Registrazione observer
    coin_dropdown.observe(on_coin_change, names="value")
    clahe_checkbox.observe(update_preview, names="value")
    bilateral_checkbox.observe(update_preview, names="value")
    bilateral_d_slider.observe(update_preview, names="value")
    bilateral_sigma_slider.observe(update_preview, names="value")
    canny_t1_slider.observe(update_preview, names="value")
    canny_t2_slider.observe(update_preview, names="value")
    save_btn.on_click(on_save_clicked)

    # Layout complessivo
    controls_box = widgets.VBox([
        widgets.HTML("<h4>Tuner Parametri Edge Detection (Per Moneta)</h4>"),
        coin_dropdown,
        widgets.HBox([clahe_checkbox, bilateral_checkbox]),
        bilateral_d_slider,
        bilateral_sigma_slider,
        canny_t1_slider,
        canny_t2_slider,
        save_btn,
        status_label
    ], layout=widgets.Layout(padding="10px", border="1px solid #ddd", border_radius="8px", margin="5px"))

    widget_ui = widgets.VBox([
        controls_box,
        preview_box
    ])

    # Disegna l'anteprima iniziale
    update_preview()

    return widget_ui
