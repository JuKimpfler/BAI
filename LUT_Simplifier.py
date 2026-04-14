#!/usr/bin/env python3
"""
LUT_Simplifier.py – LUT-Optimierer und Symmetrisierer
======================================================
Lädt eine generierte LUT-Header-Datei (.h), erzwingt Links-Rechts-Symmetrie
(achsensymmetrisch zur Roboter-Ausrichtung) und glättet die Werte durch
Kreismittelwert über N Nachbarn im 2D-Koordinatenraum.

Zeigt Pfeil-Fahrtrichtungs-Feldkarten vor und nach der Optimierung.
Die optimierte LUT wird als Kopie gespeichert.

Verwendung:
  python LUT_Simplifier.py
"""

import sys
import os
import re
import math

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QGroupBox, QComboBox, QFileDialog,
    QProgressBar, QTabWidget, QDoubleSpinBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─── Optionale scipy-Abhängigkeit ─────────────────────────────────────────────
try:
    from scipy.ndimage import uniform_filter1d as _scipy_ufi
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ─── Farbpalette (identisch mit Analyser) ─────────────────────────────────────
C_BG      = "#0d0d1a"
C_SURFACE = "#13132b"
C_PANEL   = "#1a1a38"
C_BORDER  = "#2a2a55"
C_ACCENT  = "#6c63ff"
C_ACCENT2 = "#00d4ff"
C_SUCCESS = "#22c55e"
C_WARNING = "#f59e0b"
C_DANGER  = "#ef4444"
C_TEXT    = "#e2e8f0"
C_MUTED   = "#8892aa"

GLOBAL_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C_BG}; color: {C_TEXT};
    font-family: "Segoe UI", Arial; font-size: 13px;
}}
QGroupBox {{
    background-color: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 10px; margin-top: 18px;
    padding: 12px 10px 10px 10px; font-weight: bold;
    color: {C_MUTED}; text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 8px; left: 12px;
}}
QDoubleSpinBox, QSpinBox, QComboBox {{
    background-color: {C_PANEL}; border: 1px solid {C_BORDER};
    border-radius: 6px; padding: 6px; color: {C_TEXT};
}}
QPushButton {{
    border-radius: 8px; padding: 10px; font-weight: bold;
    border: none; color: white;
}}
QPushButton:disabled {{ background-color: {C_BORDER}; color: {C_MUTED}; }}
QProgressBar {{
    background-color: {C_PANEL}; border: 1px solid {C_BORDER};
    border-radius: 6px; text-align: center; color: {C_TEXT};
}}
QProgressBar::chunk {{ background-color: {C_ACCENT}; border-radius: 6px; }}
QTabWidget::pane {{
    border: 1px solid {C_BORDER}; border-radius: 8px;
    background: {C_SURFACE};
}}
QTabBar::tab {{
    background: {C_PANEL}; color: {C_MUTED};
    border-radius: 6px 6px 0 0; padding: 8px 16px; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {C_ACCENT}; color: white; }}
"""

# ─── Konstanten ───────────────────────────────────────────────────────────────
ANZAHL_AKTIONEN   = 90
WINKEL_SCHRITT    = 360.0 / ANZAHL_AKTIONEN   # 4° pro Aktion
LUT_WINKEL        = 360    # Winkel-Dimension  (0 – 359°)
LUT_ABSTAND       = 201    # Abstands-Dimension (0 – 200 cm)
ARROW_LENGTH_FACTOR = 0.65
ARROW_WIDTH_FACTOR  = 0.07


# ─── LUT-Verarbeitung ─────────────────────────────────────────────────────────

def parse_lut_header(filepath: str) -> np.ndarray:
    """
    Liest eine .h-LUT-Datei und gibt ein (360, 201) numpy-Array zurück.
    Werte: uint8, Aktionsindizes 0–89.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r"robot_lut\[.*?\]\s*=\s*\{(.*?)\};", content, re.DOTALL)
    if not m:
        raise ValueError(
            "Keine LUT-Daten (robot_lut[...] = {...};) in der Datei gefunden.")

    body = m.group(1)
    # C-Kommentare entfernen, damit Grad-Zahlen (/* 123° */) nicht mitgezählt werden
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    values = [int(v) for v in re.findall(r"\b\d+\b", body)]

    expected = LUT_WINKEL * LUT_ABSTAND
    if len(values) != expected:
        raise ValueError(
            f"Erwartet {expected} Werte, gefunden: {len(values)}.\n"
            "Ist die Datei eine gültige robot_lut.h?")

    return np.array(values, dtype=np.uint8).reshape(LUT_WINKEL, LUT_ABSTAND)


def mirror_action(a: np.ndarray) -> np.ndarray:
    """
    Spiegele Aktionsindizes (0–89) an der Vorwärtsachse des Roboters.

    Aktion A  → Fahrtrichtung  A * 4°
    Gespiegelt → Fahrtrichtung –A * 4° = (360 – A*4)°
               = Aktion (90 – A) mod 90
    """
    return ((90 - a.astype(np.int32)) % 90).astype(np.uint8)


def apply_symmetry(lut: np.ndarray, side: str = "rechts") -> np.ndarray:
    """
    Erzwinge Links-Rechts-Symmetrie der LUT.

    side='rechts': Winkel 1–179° sind Quelle; 181–359° werden als Spiegelbild gesetzt.
    side='links' : Winkel 181–359° sind Quelle; 1–179° werden als Spiegelbild gesetzt.

    Winkel 0° (Ball geradeaus) und 180° (Ball hinten) liegen auf der Symmetrieachse
    und bleiben unverändert.
    """
    result = lut.copy()
    src = np.arange(1, 180)   # 1, 2, ..., 179
    dst = 360 - src           # 359, 358, ..., 181

    if side == "rechts":
        # Rechte Seite (1–179°) bleibt; linke Seite (359–181°) wird gespiegelt.
        result[dst, :] = mirror_action(lut[src, :])
    else:
        # Linke Seite (359–181°) bleibt; rechte Seite (1–179°) wird gespiegelt.
        result[src, :] = mirror_action(lut[dst, :])

    return result


def _box_filter1d_wrap(arr: np.ndarray, n: int, axis: int) -> np.ndarray:
    """1D Box-Filter mit zirkulärem Padding (wrap) entlang einer Achse."""
    size = 2 * n + 1
    # Zirkuläres Padding
    pad = np.concatenate(
        [np.take(arr, np.arange(-n, 0), axis=axis),
         arr,
         np.take(arr, np.arange(0, n), axis=axis)],
        axis=axis
    )
    result = np.zeros_like(arr, dtype=np.float64)
    for k in range(size):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(k, k + arr.shape[axis])
        result += pad[tuple(sl)]
    return result / size


def _box_filter1d_edge(arr: np.ndarray, n: int, axis: int) -> np.ndarray:
    """1D Box-Filter mit Edge-Padding entlang einer Achse."""
    size = 2 * n + 1
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (n, n)
    pad = np.pad(arr, pad_width, mode="edge")
    result = np.zeros_like(arr, dtype=np.float64)
    for k in range(size):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(k, k + arr.shape[axis])
        result += pad[tuple(sl)]
    return result / size


def smooth_lut(lut: np.ndarray, n_radius: int = 2) -> np.ndarray:
    """
    Glättet die LUT durch Kreismittelwert über ein (2·n+1)×(2·n+1) Fenster.

    Die LUT wird in sin/cos-Komponenten zerlegt (Kreismittelwert), um den
    periodischen Charakter der Aktionsrichtungen (0–356° in 4°-Schritten)
    korrekt zu behandeln.

    Achsenbehandlung:
      - Winkelachse (Achse 0): zirkuläres Padding (wrap) – 0° und 359° sind benachbart.
      - Abstandsachse (Achse 1): Edge-Padding – Randwerte werden gespiegelt.

    scipy.ndimage wird verwendet, falls verfügbar; sonst pure-numpy-Fallback.
    """
    if n_radius == 0:
        return lut.copy()

    size = 2 * n_radius + 1
    angles_rad = lut.astype(np.float64) * (WINKEL_SCHRITT * math.pi / 180.0)
    sin_m = np.sin(angles_rad)
    cos_m = np.cos(angles_rad)

    if _SCIPY:
        sin_s = _scipy_ufi(sin_m, size=size, axis=0, mode="wrap")
        sin_s = _scipy_ufi(sin_s, size=size, axis=1, mode="nearest")
        cos_s = _scipy_ufi(cos_m, size=size, axis=0, mode="wrap")
        cos_s = _scipy_ufi(cos_s, size=size, axis=1, mode="nearest")
    else:
        n = n_radius
        # Winkelachse: wrap; Abstandsachse: edge
        sin_s = _box_filter1d_wrap(sin_m, n, axis=0)
        sin_s = _box_filter1d_edge(sin_s, n, axis=1)
        cos_s = _box_filter1d_wrap(cos_m, n, axis=0)
        cos_s = _box_filter1d_edge(cos_s, n, axis=1)

    mean_rad = np.arctan2(sin_s, cos_s)
    mean_deg = (np.degrees(mean_rad) + 360.0) % 360.0
    result = (np.round(mean_deg / WINKEL_SCHRITT).astype(np.int32) % ANZAHL_AKTIONEN)
    return result.astype(np.uint8)


def compute_field_vectors(
    lut: np.ndarray,
    feld_w: float,
    feld_h: float,
    b_x: float,
    b_y: float,
    schritt_m: float,
    rob_deg: float = 0.0,
):
    """
    Berechne das Fahrtrichtungs-Vektorfeld aus der LUT für eine gegebene
    Roboter-Orientierung.

    Gibt (xs, ys, u, v, winkel_arr) zurück.
    u/v: normierte Fahrtrichtungsvektoren; winkel_arr: Fahrtrichtung in Grad.
    """
    xs = np.arange(0.0, feld_w + schritt_m * 0.5, schritt_m)
    ys = np.arange(0.0, feld_h + schritt_m * 0.5, schritt_m)
    nx, ny = len(xs), len(ys)

    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    w_arr = np.full((ny, nx), np.nan, dtype=np.float32)

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            dx, dy = b_x - x, b_y - y
            dist_cm = math.hypot(dx, dy) * 100.0
            if dist_cm < 0.5:
                continue
            abs_w = math.degrees(math.atan2(dx, dy))
            rel_w = (abs_w - rob_deg) % 360.0

            w_idx = int(round(rel_w)) % LUT_WINKEL
            d_idx = min(max(int(round(dist_cm)), 0), LUT_ABSTAND - 1)
            action = int(lut[w_idx, d_idx])

            glo = math.radians(rob_deg) + math.radians(action * WINKEL_SCHRITT)
            u[iy, ix] = math.sin(glo)
            v[iy, ix] = math.cos(glo)
            w_arr[iy, ix] = (math.degrees(glo) + 360.0) % 360.0

    return xs, ys, u, v, w_arr


def write_lut_header(
    lut: np.ndarray,
    original_path: str,
    output_path: str,
    side: str,
    n_radius: int,
) -> None:
    """
    Schreibt die optimierte LUT in eine neue Header-Datei.
    Behält die gesamte Struktur der Original-Datei bei (Kommentare,
    Hilfsfunktionen, Defines); ersetzt nur die Array-Daten.
    """
    with open(original_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Neue Array-Zeilen aufbauen
    lines = []
    for w in range(LUT_WINKEL):
        vals = ", ".join(str(int(lut[w, d])) for d in range(LUT_ABSTAND))
        suffix = "," if w < LUT_WINKEL - 1 else ""
        lines.append(f"  /* {w:>3}° */ {vals}{suffix}")
    new_body = "\n".join(lines)

    # Bereits vorhandenen Optimierungs-Hinweis entfernen (für wiederholtes Speichern)
    content = re.sub(r"// Optimiert mit LUT_Simplifier\.py.*?\n", "", content)

    # Neuen Optimierungs-Hinweis direkt nach dem Generator-Kommentar einfügen
    note = (
        f"// Optimiert mit LUT_Simplifier.py  "
        f"(Symmetrie-Seite: {side}, Glätt-Radius: {n_radius})\n"
    )
    content = content.replace(
        "// Automatisch generiert von generate_lut.py",
        "// Automatisch generiert von generate_lut.py\n" + note,
        1,
    )

    # Array-Daten ersetzen
    new_content = re.sub(
        r"(static const uint8_t PROGMEM robot_lut\[.*?\]\s*=\s*\{)\n.*?(\n\};)",
        lambda m: m.group(1) + "\n" + new_body + m.group(2),
        content,
        flags=re.DOTALL,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)


# ─── Plot-Hilfsfunktionen (identisch mit Analyser) ────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors=C_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=9)


def _add_colorbar(fig, im, ax, label=""):
    import matplotlib.pyplot as plt
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, color=C_TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C_TEXT, labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_TEXT)


def _draw_ball_and_border(ax, b_x, b_y, fw, fh):
    ax.add_patch(
        patches.Rectangle((0, 0), fw, fh, lw=1.5, edgecolor=C_BORDER, facecolor="none")
    )
    ax.plot(b_x, b_y, "wo", markersize=8, zorder=5, label="Ball")


# ─── Optimierungs-Worker ──────────────────────────────────────────────────────

class OptimierungsWorker(QThread):
    """Führt Symmetrisierung und Glättung der LUT im Hintergrund durch."""

    fortschritt_signal = pyqtSignal(int, str)
    ergebnis_signal    = pyqtSignal(object)   # optimierte LUT (np.ndarray)
    fehler_signal      = pyqtSignal(str)

    def __init__(self, lut_original: np.ndarray, side: str, n_radius: int):
        super().__init__()
        self.lut      = lut_original
        self.side     = side
        self.n_radius = n_radius

    def run(self):
        try:
            self.fortschritt_signal.emit(15, "Wende Symmetrie an …")
            sym_lut = apply_symmetry(self.lut, self.side)

            self.fortschritt_signal.emit(50, f"Glätte LUT (Radius {self.n_radius}) …")
            opt_lut = smooth_lut(sym_lut, self.n_radius)

            self.fortschritt_signal.emit(100, "✅ Optimierung abgeschlossen!")
            self.ergebnis_signal.emit(opt_lut)
        except Exception as exc:
            self.fehler_signal.emit(str(exc))


# ─── Haupt-GUI ────────────────────────────────────────────────────────────────

class LutSimplifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUT Simplifier · Symmetrie & Glättung")
        self.setMinimumSize(1380, 840)
        self.setStyleSheet(GLOBAL_STYLE)

        self.lut_original  = None
        self.lut_optimiert = None
        self.lut_pfad      = None
        self.worker        = None

        self._init_ui()
        self._draw_placeholder()

    # ──────────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sidebar.setStyleSheet(
            f"background-color:{C_SURFACE}; border-radius:12px;"
            f" border:1px solid {C_BORDER};"
        )
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(10, 10, 10, 10)
        sb.setSpacing(8)

        # Datei-Gruppe
        grp_file = QGroupBox("LUT-Datei")
        lf = QVBoxLayout(grp_file)
        self.lbl_datei = QLabel("Keine Datei geladen.")
        self.lbl_datei.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        self.lbl_datei.setWordWrap(True)
        lf.addWidget(self.lbl_datei)
        btn_open = QPushButton("📂  Datei öffnen …")
        btn_open.setStyleSheet(f"background-color:{C_PANEL}; color:{C_TEXT};")
        btn_open.clicked.connect(self._oeffne_datei)
        lf.addWidget(btn_open)
        sb.addWidget(grp_file)

        # Optimierungs-Gruppe
        grp_opt = QGroupBox("Optimierung")
        lo = QVBoxLayout(grp_opt)

        lo.addWidget(QLabel("Symmetrie-Seite:"))
        self.combo_seite = QComboBox()
        self.combo_seite.addItems(["Rechts  (0° – 180°)", "Links  (180° – 360°)"])
        self.combo_seite.setToolTip(
            "Rechts: Winkel 1–179° als Quelle; linke Seite wird gespiegelt.\n"
            "Links : Winkel 181–359° als Quelle; rechte Seite wird gespiegelt."
        )
        lo.addWidget(self.combo_seite)

        lo.addWidget(QLabel("Nachbar-Radius N:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(0, 20)
        self.spin_n.setValue(3)
        self.spin_n.setToolTip(
            "Fenstergröße: (2·N+1) × (2·N+1)\n"
            "N=0 → kein Glätten\n"
            "N=1 → 3×3    N=2 → 5×5\n"
            "N=3 → 7×7    N=5 → 11×11"
        )
        lo.addWidget(self.spin_n)
        sb.addWidget(grp_opt)

        # Feldansicht-Gruppe
        grp_feld = QGroupBox("Feldansicht")
        lv = QVBoxLayout(grp_feld)

        lv.addWidget(QLabel("Feldgröße Breite / Höhe (m):"))
        hf = QHBoxLayout()
        self.spin_fw = QDoubleSpinBox()
        self.spin_fw.setRange(1.0, 5.0)
        self.spin_fw.setValue(3.0)
        self.spin_fw.setSingleStep(0.5)
        self.spin_fh = QDoubleSpinBox()
        self.spin_fh.setRange(1.0, 5.0)
        self.spin_fh.setValue(3.0)
        self.spin_fh.setSingleStep(0.5)
        hf.addWidget(self.spin_fw)
        hf.addWidget(self.spin_fh)
        lv.addLayout(hf)

        lv.addWidget(QLabel("Ball-Position X / Y (m):"))
        hb = QHBoxLayout()
        self.spin_bx = QDoubleSpinBox()
        self.spin_bx.setRange(0.0, 5.0)
        self.spin_bx.setValue(1.5)
        self.spin_bx.setSingleStep(0.1)
        self.spin_by = QDoubleSpinBox()
        self.spin_by.setRange(0.0, 5.0)
        self.spin_by.setValue(1.5)
        self.spin_by.setSingleStep(0.1)
        hb.addWidget(self.spin_bx)
        hb.addWidget(self.spin_by)
        lv.addLayout(hb)

        lv.addWidget(QLabel("Roboter-Orientierung (°):"))
        self.spin_rob_ori = QSpinBox()
        self.spin_rob_ori.setRange(0, 359)
        self.spin_rob_ori.setValue(0)
        lv.addWidget(self.spin_rob_ori)

        lv.addWidget(QLabel("Gitterschritt (cm):"))
        self.spin_schritt = QSpinBox()
        self.spin_schritt.setRange(2, 30)
        self.spin_schritt.setValue(10)
        lv.addWidget(self.spin_schritt)

        sb.addWidget(grp_feld)

        # Aktions-Buttons
        self.btn_opt = QPushButton("⚙️  Optimieren")
        self.btn_opt.setStyleSheet(f"background-color:{C_ACCENT}; color:white;")
        self.btn_opt.clicked.connect(self._starte_optimierung)
        self.btn_opt.setEnabled(False)
        sb.addWidget(self.btn_opt)

        self.btn_save = QPushButton("💾  Speichern als …")
        self.btn_save.setStyleSheet(f"background-color:{C_SUCCESS}; color:white;")
        self.btn_save.clicked.connect(self._speichern)
        self.btn_save.setEnabled(False)
        sb.addWidget(self.btn_save)

        # Fortschritt
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        sb.addWidget(self.progress)

        self.lbl_status = QLabel("Bereit. LUT-Datei öffnen.")
        self.lbl_status.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        sb.addWidget(self.lbl_status)

        sb.addStretch()
        main.addWidget(sidebar)

        # ── Tabs ─────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        main.addWidget(self.tabs, stretch=1)

        # Tab 1 – Vor Optimierung
        self.fig_vor = Figure(figsize=(11, 5))
        self.fig_vor.patch.set_facecolor(C_BG)
        self.canvas_vor = FigureCanvas(self.fig_vor)
        tab1 = QWidget()
        l1 = QVBoxLayout(tab1)
        l1.setContentsMargins(0, 0, 0, 0)
        l1.addWidget(self.canvas_vor)
        self.tabs.addTab(tab1, "📊  Vor Optimierung")

        # Tab 2 – Nach Optimierung
        self.fig_nach = Figure(figsize=(11, 5))
        self.fig_nach.patch.set_facecolor(C_BG)
        self.canvas_nach = FigureCanvas(self.fig_nach)
        tab2 = QWidget()
        l2 = QVBoxLayout(tab2)
        l2.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(self.canvas_nach)
        self.tabs.addTab(tab2, "✨  Nach Optimierung")

        # Tab 3 – LUT-Vergleich (Heatmaps nebeneinander)
        self.fig_vergl = Figure(figsize=(12, 5))
        self.fig_vergl.patch.set_facecolor(C_BG)
        self.canvas_vergl = FigureCanvas(self.fig_vergl)
        tab3 = QWidget()
        l3 = QVBoxLayout(tab3)
        l3.setContentsMargins(0, 0, 0, 0)
        l3.addWidget(self.canvas_vergl)
        self.tabs.addTab(tab3, "🔍  LUT-Vergleich")

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_placeholder(self):
        msg = "Noch keine LUT geladen.\nDatei öffnen, dann ⚙️ Optimieren klicken."
        for fig, canvas in [
            (self.fig_vor,   self.canvas_vor),
            (self.fig_nach,  self.canvas_nach),
            (self.fig_vergl, self.canvas_vergl),
        ]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor("#1a1a2e")
            ax.text(
                0.5, 0.5, msg,
                transform=ax.transAxes, ha="center", va="center",
                color=C_MUTED, fontsize=13, style="italic",
                multialignment="center",
            )
            ax.axis("off")
            canvas.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _oeffne_datei(self):
        pfad, _ = QFileDialog.getOpenFileName(
            self, "LUT Header-Datei öffnen", "",
            "C-Header (*.h);;Alle Dateien (*)",
        )
        if not pfad:
            return
        try:
            lut = parse_lut_header(pfad)
            self.lut_original  = lut
            self.lut_pfad      = pfad
            self.lut_optimiert = None

            self.btn_opt.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.progress.setValue(0)

            fname = os.path.basename(pfad)
            self.lbl_datei.setText(fname)
            self.lbl_status.setStyleSheet(f"color:{C_SUCCESS}; font-size:11px;")
            self.lbl_status.setText(
                f"✅ {fname} geladen  ({LUT_WINKEL}×{LUT_ABSTAND} Einträge)"
            )

            self._zeichne_lut_tab(lut, self.fig_vor, self.canvas_vor, "Vor Optimierung")
            # Nach-Tab zurücksetzen
            self._zeichne_placeholder_tab(
                self.fig_nach, self.canvas_nach,
                "Noch nicht optimiert. ⚙️ Optimieren klicken.",
            )
            self._zeichne_placeholder_tab(
                self.fig_vergl, self.canvas_vergl,
                "Noch nicht optimiert. ⚙️ Optimieren klicken.",
            )
        except Exception as exc:
            self.lbl_status.setStyleSheet(f"color:{C_DANGER}; font-size:11px;")
            self.lbl_status.setText(f"Ladefehler: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    def _starte_optimierung(self):
        if self.lut_original is None:
            return

        side = self._get_side()
        n_radius = self.spin_n.value()

        self.btn_opt.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_status.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        self.lbl_status.setText("Optimiere …")

        self.worker = OptimierungsWorker(self.lut_original, side, n_radius)
        self.worker.fortschritt_signal.connect(self._on_fortschritt)
        self.worker.ergebnis_signal.connect(self._on_ergebnis)
        self.worker.fehler_signal.connect(self._on_fehler)
        self.worker.start()

    def _on_fortschritt(self, pct: int, text: str):
        self.progress.setValue(pct)
        self.lbl_status.setText(text)

    def _on_fehler(self, msg: str):
        self.lbl_status.setStyleSheet(f"color:{C_DANGER}; font-size:11px;")
        self.lbl_status.setText(f"Fehler: {msg}")
        self.btn_opt.setEnabled(True)

    def _on_ergebnis(self, opt_lut: np.ndarray):
        self.lut_optimiert = opt_lut
        self.btn_opt.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_status.setStyleSheet(f"color:{C_SUCCESS}; font-size:11px;")
        self.lbl_status.setText("✅ Optimierung abgeschlossen!")

        self._zeichne_lut_tab(opt_lut, self.fig_nach, self.canvas_nach, "Nach Optimierung")
        self._zeichne_vergleich()
        self.tabs.setCurrentIndex(1)   # Wechsel zu „Nach Optimierung"

    # ──────────────────────────────────────────────────────────────────────────
    def _speichern(self):
        if self.lut_optimiert is None or self.lut_pfad is None:
            return

        base, ext = os.path.splitext(self.lut_pfad)
        default = base + "_optimiert" + ext

        pfad, _ = QFileDialog.getSaveFileName(
            self, "Optimierte LUT speichern", default,
            "C-Header (*.h);;Alle Dateien (*)",
        )
        if not pfad:
            return

        side = self._get_side()
        n_radius = self.spin_n.value()

        try:
            write_lut_header(self.lut_optimiert, self.lut_pfad, pfad, side, n_radius)
            self.lbl_status.setStyleSheet(f"color:{C_SUCCESS}; font-size:11px;")
            self.lbl_status.setText(f"💾 Gespeichert: {os.path.basename(pfad)}")
        except Exception as exc:
            self.lbl_status.setStyleSheet(f"color:{C_DANGER}; font-size:11px;")
            self.lbl_status.setText(f"Speicher-Fehler: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    def _get_side(self) -> str:
        """Gibt 'rechts' oder 'links' zurück, basierend auf dem Combo-Box-Index."""
        return "rechts" if self.combo_seite.currentIndex() == 0 else "links"

    # ──────────────────────────────────────────────────────────────────────────
    def _get_feld_params(self):
        return (
            self.spin_fw.value(),
            self.spin_fh.value(),
            self.spin_bx.value(),
            self.spin_by.value(),
            self.spin_schritt.value() / 100.0,   # cm → m
            float(self.spin_rob_ori.value()),
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _zeichne_placeholder_tab(self, fig, canvas, msg):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5, 0.5, msg,
            transform=ax.transAxes, ha="center", va="center",
            color=C_MUTED, fontsize=13, style="italic",
            multialignment="center",
        )
        ax.axis("off")
        canvas.draw()

    def _zeichne_lut_tab(
        self,
        lut: np.ndarray,
        fig: Figure,
        canvas: FigureCanvas,
        titel_prefix: str,
    ):
        """
        Zeichnet einen Tab mit zwei Subplots:
          - Links:  LUT-Heatmap (Winkel × Abstand → Fahrtrichtung)
          - Rechts: Pfeil-Fahrtrichtungs-Feldkarte (X/Y-Feldansicht)
        """
        fw, fh, bx, by, schritt, rob_ori = self._get_feld_params()

        fig.clear()
        axes = fig.subplots(1, 2)

        # ── Linkes Subplot: LUT-Heatmap ───────────────────────────────────────
        ax_lut = axes[0]
        dirs_deg = lut.astype(np.float32) * WINKEL_SCHRITT   # 0–356°
        im = ax_lut.imshow(
            dirs_deg.T,          # Transponiert: Y = Abstand, X = Winkel
            origin="lower",
            aspect="auto",
            extent=[0, LUT_WINKEL - 1, 0, LUT_ABSTAND - 1],
            cmap="hsv",
            vmin=0,
            vmax=360,
            interpolation="nearest",
        )
        _style_ax(
            ax_lut,
            f"{titel_prefix} – LUT-Heatmap",
            "Rel. Winkel Ball→Roboter (°)",
            "Abstand (cm)",
        )
        _add_colorbar(fig, im, ax_lut, "Fahrtrichtung (°)")

        # Symmetrieachsen markieren
        ax_lut.axvline(0,   color=C_WARNING, linewidth=1.0, alpha=0.6,
                       linestyle="--", label="Sym.-Achse 0°/360°")
        ax_lut.axvline(180, color=C_ACCENT2, linewidth=1.0, alpha=0.6,
                       linestyle="--", label="Sym.-Achse 180°")
        ax_lut.legend(
            facecolor=C_PANEL, edgecolor=C_BORDER,
            labelcolor=C_TEXT, fontsize=7, loc="upper right",
        )

        # ── Rechtes Subplot: Pfeil-Fahrtrichtungs-Feldkarte ───────────────────
        ax_feld = axes[1]
        xs, ys, u, v, w_arr = compute_field_vectors(
            lut, fw, fh, bx, by, schritt, rob_ori
        )

        XX, YY = np.meshgrid(xs, ys)
        mag = np.sqrt(u ** 2 + v ** 2)
        mag[mag == 0] = 1.0
        u_n = u / mag
        v_n = v / mag

        C_arr = w_arr.copy()
        C_arr[np.isnan(C_arr)] = 0.0

        q = ax_feld.quiver(
            XX, YY,
            u_n * schritt * ARROW_LENGTH_FACTOR,
            v_n * schritt * ARROW_LENGTH_FACTOR,
            C_arr,
            cmap="hsv",
            norm=mcolors.Normalize(vmin=0, vmax=360),
            units="xy",
            scale=1.0,
            width=schritt * ARROW_WIDTH_FACTOR,
            headwidth=4,
            headlength=5,
            headaxislength=4.5,
            alpha=0.85,
        )
        _add_colorbar(fig, q, ax_feld, "Fahrtrichtung (°)")
        _draw_ball_and_border(ax_feld, bx, by, fw, fh)
        ax_feld.set_xlim(-schritt, fw + schritt)
        ax_feld.set_ylim(-schritt, fh + schritt)
        ax_feld.set_aspect("equal")
        ax_feld.legend(
            facecolor=C_PANEL, edgecolor=C_BORDER,
            labelcolor=C_TEXT, fontsize=8,
        )
        _style_ax(
            ax_feld,
            f"{titel_prefix} – Fahrtrichtungs-Feldkarte  (Ori: {int(rob_ori)}°)",
            "X (m)",
            "Y (m)",
        )

        fig.suptitle(titel_prefix, color=C_TEXT, fontsize=13, fontweight="bold")
        fig.tight_layout()
        canvas.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _zeichne_vergleich(self):
        """Tab 3: LUT-Heatmap-Vergleich (Original vs. Optimiert) nebeneinander."""
        if self.lut_original is None or self.lut_optimiert is None:
            return

        self.fig_vergl.clear()
        axes = self.fig_vergl.subplots(1, 2)

        for ax, lut, titel in [
            (axes[0], self.lut_original,  "Original"),
            (axes[1], self.lut_optimiert, "Optimiert"),
        ]:
            dirs_deg = lut.astype(np.float32) * WINKEL_SCHRITT
            im = ax.imshow(
                dirs_deg.T,
                origin="lower",
                aspect="auto",
                extent=[0, LUT_WINKEL - 1, 0, LUT_ABSTAND - 1],
                cmap="hsv",
                vmin=0,
                vmax=360,
                interpolation="nearest",
            )
            _style_ax(ax, f"LUT-Heatmap: {titel}",
                      "Rel. Winkel (°)", "Abstand (cm)")
            _add_colorbar(self.fig_vergl, im, ax, "Fahrtrichtung (°)")

            # Symmetrieachsen
            ax.axvline(0,   color=C_WARNING, linewidth=1.0, alpha=0.6,
                       linestyle="--")
            ax.axvline(180, color=C_ACCENT2, linewidth=1.0, alpha=0.6,
                       linestyle="--")

        self.fig_vergl.suptitle(
            "LUT-Vergleich: Original vs. Optimiert",
            color=C_TEXT, fontsize=13, fontweight="bold",
        )
        self.fig_vergl.tight_layout()
        self.canvas_vergl.draw()


# ─── Einstiegspunkt ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LutSimplifierWindow()
    window.show()
    sys.exit(app.exec_())
