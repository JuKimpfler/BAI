#!/usr/bin/env python3
"""
LUT_Analyser.py – Umfassende Analyse-GUI für die Roboter-LUT
=============================================================
Testet die Lookup-Tabelle (.h) von allen Positionen (0–3 m in cm-Schritten)
und Orientierungen (0–360° in Grad-Schritten) und zeigt Heatmaps,
ein Vektorfeld sowie Statistiken an.

Dieses Skript ist stark an Analyser.py angelehnt, nutzt jedoch die
LUT-Datei (robot_lut.h) anstelle eines PyTorch-Modells (.pth).
"""

import sys
import math
import os
import re
import numpy as np
from sensor_model import simuliere_ball_sensor_abstand

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QGroupBox,
    QComboBox, QProgressBar, QTabWidget, QSlider, QFileDialog,
    QLineEdit, QScrollArea,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─── Farbpalette (angelehnt an Analyser.py) ──────────────────────────────────
C_BG      = "#0d0d1a"
C_SURFACE = "#13132b"
C_PANEL   = "#1a1a38"
C_BORDER  = "#2a2a55"
C_ACCENT  = "#f59e0b"   # Amber/Orange für LUT
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
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
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
QSlider::groove:horizontal {{
    border: 1px solid {C_BORDER}; height: 8px;
    background: {C_PANEL}; border-radius: 4px;
}}
QSlider::handle:horizontal {{
    background: {C_ACCENT}; width: 14px;
    margin: -3px 0; border-radius: 7px;
}}
"""

# ─── LUT Konfiguration ───────────────────────────────────────────────────────
ANZAHL_AKTIONEN   = 90
WINKEL_SCHRITT    = 360.0 / ANZAHL_AKTIONEN   # 4 Grad pro Aktion
ANZAHL_ABSTAENDE  = 201   # 0 – 200 cm (Schrittweite 1 cm)
ANZAHL_WINKEL     = 360   # 0 – 359 Grad (Schrittweite 1°)

SCHRITT_GROESSE_M = 0.02    # Fahrtschritt des Roboters pro Zeitschritt (m)
BALL_PUFFER_CM    = 2.0     # Zusatzpuffer um den Ball-Radius für Kollisionsprüfung
SUCCESS_BASIS     = 10000   # Basis-Punkte für einen erfolgreichen Lauf
SUCCESS_STEP_MALUS = 5      # Punktabzug pro Schritt im Erfolgsfall
ARROW_LENGTH_FACTOR = 0.65  # Pfeillänge relativ zum Gitterschritt
ARROW_WIDTH_FACTOR  = 0.07  # Pfeilbreite relativ zum Gitterschritt
DISTANCE_BINS       = 30    # Anzahl Bins für Abstands-Histogramm


# ─── LUT Lade- und Nachschlage-Funktionen ────────────────────────────────────
def lade_lut(lut_datei: str) -> list:
    """Lädt die LUT-Werte aus einer robot_lut.h-Datei und gibt sie als Liste zurück.

    Layout der Tabelle: lut[winkel * ANZAHL_ABSTAENDE + abstand]
      winkel  = 0 … 359  (1°-Schritte)
      abstand = 0 … 200  (1 cm-Schritte)
    """
    with open(lut_datei, "r", encoding="utf-8") as f:
        inhalt = f.read()

    # Array-Inhalt zwischen dem öffnenden { und dem schließenden }; extrahieren
    match = re.search(r'robot_lut\[[^\]]*\]\s*=\s*\{([^}]*)\};', inhalt, re.DOTALL)
    if not match:
        raise ValueError(f"Keine LUT-Daten in '{lut_datei}' gefunden.")

    zahlen_text = match.group(1)
    # C-Kommentare entfernen (/* ... */)
    zahlen_text = re.sub(r'/\*.*?\*/', '', zahlen_text, flags=re.DOTALL)
    # Alle Ganzzahlen extrahieren
    werte = [int(x) for x in re.findall(r'\d+', zahlen_text)]

    erwartet = ANZAHL_WINKEL * ANZAHL_ABSTAENDE
    if len(werte) != erwartet:
        raise ValueError(
            f"LUT hat {len(werte)} Einträge, erwartet {erwartet} "
            f"({ANZAHL_WINKEL} Winkel × {ANZAHL_ABSTAENDE} Abstände)."
        )
    return werte


def lut_nachschlagen(lut: list, winkel_deg: float, abstand_cm: float) -> int:
    """Gibt den Aktionsindex (0–89) für den gegebenen Zustand aus der LUT zurück."""
    w = int(round(winkel_deg)) % ANZAHL_WINKEL
    sensor_abstand_cm = simuliere_ball_sensor_abstand(abstand_cm)
    a = max(0, min(200, int(round(sensor_abstand_cm))))
    return lut[w * ANZAHL_ABSTAENDE + a]


def berechne_zustand(r_x, r_y, r_w, b_x, b_y):
    dx, dy = b_x - r_x, b_y - r_y
    abstand_cm = math.hypot(dx, dy) * 100
    abs_winkel_deg = math.degrees(math.atan2(dx, dy))
    rel_winkel = (abs_winkel_deg - r_w) % 360
    if rel_winkel > 180:
        rel_winkel -= 360
    return rel_winkel, abstand_cm


def finde_lut_datei() -> str:
    """Sucht nach robot_lut.h zuerst im Skript-Verzeichnis, dann in teensy/include/."""
    basis = os.path.dirname(os.path.abspath(__file__))
    kandidaten = [
        os.path.join(basis, "robot_lut.h"),
        os.path.join(basis, "teensy", "include", "robot_lut.h"),
    ]
    for k in kandidaten:
        if os.path.exists(k):
            return k
    return kandidaten[0]  # Standardpfad auch wenn nicht vorhanden


# ─── Analyse-Worker ───────────────────────────────────────────────────────────
class LUTAnalyseWorker(QThread):
    """
    Testet die LUT von allen Positionen und Orientierungen.
    Sendet Ergebnis-Dictionary via Signal zurück.
    """
    fortschritt_signal = pyqtSignal(int, str)   # (prozent, text)
    ergebnis_signal    = pyqtSignal(dict)
    fehler_signal      = pyqtSignal(str)

    def __init__(self, lut_datei, feld_w, feld_h, b_x, b_y,
                 schritt_cm, n_orientierungen, max_schritte,
                 rob_radius_cm=11.0, toleranz=8):
        super().__init__()
        self.lut_datei       = lut_datei
        self.feld_w          = feld_w
        self.feld_h          = feld_h
        self.b_x             = b_x
        self.b_y             = b_y
        self.schritt_cm      = schritt_cm
        self.n_ori           = n_orientierungen
        self.max_schritte    = max_schritte
        self.rob_radius_cm   = rob_radius_cm
        self.toleranz        = toleranz
        self._stop           = False

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------------
    def run(self):
        try:
            lut = lade_lut(self.lut_datei)
        except Exception as e:
            self.fehler_signal.emit(str(e))
            return

        schritt_m = self.schritt_cm / 100.0

        xs = np.arange(0.0, self.feld_w + schritt_m * 0.5, schritt_m)
        ys = np.arange(0.0, self.feld_h + schritt_m * 0.5, schritt_m)
        nx, ny = len(xs), len(ys)

        orientierungen = (
            [0.0] if self.n_ori == 1
            else np.linspace(0, 360, self.n_ori, endpoint=False).tolist()
        )

        # Ergebnis-Arrays  (Zeile = y-Index, Spalte = x-Index)
        erfolg_map   = np.full((ny, nx), np.nan, dtype=np.float32)
        schritte_map = np.full((ny, nx), np.nan, dtype=np.float32)
        score_map    = np.full((ny, nx), np.nan, dtype=np.float32)

        # Vektorfeld (für Orientierung 0°)
        vf_u      = np.zeros((ny, nx), dtype=np.float32)
        vf_v      = np.zeros((ny, nx), dtype=np.float32)
        vf_winkel = np.full((ny, nx), np.nan, dtype=np.float32)

        # Globale Statistiken
        outcome_counts    = {"erfolg": 0, "crash": 0, "wand": 0, "timeout": 0}
        success_by_dist   = []          # [(abstand_cm, bool)]
        success_by_orient = np.zeros(len(orientierungen), dtype=np.float32)
        n_valide_pos      = 0

        gesamt      = nx * ny
        verarbeitet = 0

        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                if self._stop:
                    return

                abstand_ball = math.hypot(x - self.b_x,
                                          y - self.b_y) * 100
                # Position zu nah am Ball überspringen
                if abstand_ball < self.rob_radius_cm + BALL_PUFFER_CM:
                    verarbeitet += 1
                    continue

                # ── Vektorfeld für Orientierung 0° ───────────────────
                rw0 = 0.0
                rel_w0, dist0 = berechne_zustand(
                    x, y, rw0, self.b_x, self.b_y)
                if dist0 > 0:
                    a0 = lut_nachschlagen(lut, rel_w0, dist0)
                    glo = math.radians(rw0) + math.radians(
                        a0 * WINKEL_SCHRITT)
                    vf_u[iy, ix] = math.sin(glo)
                    vf_v[iy, ix] = math.cos(glo)
                    vf_winkel[iy, ix] = (math.degrees(glo) + 360) % 360

                # ── Vollsimulation über alle Orientierungen ───────────
                erg_liste, step_liste, sc_liste = [], [], []
                for oi, r_w in enumerate(orientierungen):
                    if self._stop:
                        return
                    outcome, n_steps, score = self._simuliere(
                        lut, x, y, r_w)
                    outcome_counts[outcome] += 1
                    erg = 1 if outcome == "erfolg" else 0
                    erg_liste.append(erg)
                    step_liste.append(n_steps)
                    sc_liste.append(score)
                    success_by_dist.append((abstand_ball, erg))
                    success_by_orient[oi] += erg

                erfolg_map[iy, ix]   = np.mean(erg_liste)
                schritte_map[iy, ix] = np.mean(step_liste)
                score_map[iy, ix]    = np.mean(sc_liste)
                n_valide_pos        += 1

                verarbeitet += 1
                if verarbeitet % max(1, gesamt // 300) == 0:
                    pct = int(100 * verarbeitet / gesamt)
                    self.fortschritt_signal.emit(
                        pct,
                        f"Analysiere … {verarbeitet}/{gesamt} Positionen")

        # Normalisiere success_by_orient auf Wertebereich [0, 1]
        if n_valide_pos > 0:
            success_by_orient = success_by_orient / n_valide_pos
        else:
            success_by_orient[:] = np.nan

        self.fortschritt_signal.emit(100, "✅ Analyse abgeschlossen!")
        self.ergebnis_signal.emit({
            "xs": xs, "ys": ys,
            "b_x": self.b_x, "b_y": self.b_y,
            "feld_w": self.feld_w, "feld_h": self.feld_h,
            "erfolg_map":   erfolg_map,
            "schritte_map": schritte_map,
            "score_map":    score_map,
            "vf_u":         vf_u,
            "vf_v":         vf_v,
            "vf_winkel":    vf_winkel,
            "outcome_counts":    outcome_counts,
            "success_by_dist":   success_by_dist,
            "success_by_orient": success_by_orient,
            "orientierungen":    orientierungen,
            "n_valide_pos":      n_valide_pos,
            "schritt_m":    schritt_m,
            "max_schritte": self.max_schritte,
            "lut_datei":    self.lut_datei,
        })

    # ------------------------------------------------------------------
    def _simuliere(self, lut, start_x, start_y, r_w):
        """Führt einen vollständigen Lauf mit der LUT durch."""
        r_x, r_y = start_x, start_y
        punkte   = 0

        for schritt in range(self.max_schritte):
            rel_w, dist_cm = berechne_zustand(
                r_x, r_y, r_w, self.b_x, self.b_y)
            akt = lut_nachschlagen(lut, rel_w, dist_cm)

            glo  = math.radians(r_w) + math.radians(akt * WINKEL_SCHRITT)
            r_x += SCHRITT_GROESSE_M * math.sin(glo)
            r_y += SCHRITT_GROESSE_M * math.cos(glo)

            neu_rel, neu_dist = berechne_zustand(
                r_x, r_y, r_w, self.b_x, self.b_y)

            if neu_dist <= (self.rob_radius_cm + BALL_PUFFER_CM):
                if abs(neu_rel) <= self.toleranz:
                    punkte += SUCCESS_BASIS - schritt * SUCCESS_STEP_MALUS
                    return "erfolg", schritt + 1, punkte
                punkte -= 1000
                return "crash", schritt + 1, punkte

            if (r_x < 0 or r_x > self.feld_w
                    or r_y < 0 or r_y > self.feld_h):
                punkte -= 500
                return "wand", schritt + 1, punkte

            punkte -= 1

        return "timeout", self.max_schritte, punkte


# ─── Plot-Hilfsfunktionen ─────────────────────────────────────────────────────
def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors=C_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=10,
                     fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=9)


def _add_colorbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, color=C_TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C_TEXT, labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_TEXT)


def _draw_ball_and_border(ax, b_x, b_y, fw, fh):
    ax.add_patch(patches.Rectangle(
        (0, 0), fw, fh, lw=1.5, edgecolor=C_BORDER, facecolor="none"))
    ax.plot(b_x, b_y, "wo", markersize=8, zorder=5, label="Ball")


# ─── Haupt-GUI ────────────────────────────────────────────────────────────────
class LUTAnalyserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboter LUT · Umfassende Analyse")
        self.setMinimumSize(1360, 820)
        self.setStyleSheet(GLOBAL_STYLE)
        self.worker     = None
        self.ergebnisse = None
        self.lut_pfad   = finde_lut_datei()
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
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setFixedWidth(290)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet(
            f"QScrollArea {{ background-color:{C_SURFACE}; border-radius:12px;"
            f" border:1px solid {C_BORDER}; }}"
            f" QScrollBar:vertical {{ background:{C_SURFACE}; width:8px; border:none; }}"
            f" QScrollBar::handle:vertical {{ background:{C_BORDER}; border-radius:4px; }}")
        sidebar = QWidget()
        sidebar.setStyleSheet("border:none;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(10, 10, 10, 10)
        sb.setSpacing(8)

        # LUT-Datei
        grp_lut = QGroupBox("LUT-Datei")
        ll = QVBoxLayout(grp_lut)
        ll.addWidget(QLabel("robot_lut.h Datei:"))
        h_lut = QHBoxLayout()
        self.edit_lut = QLineEdit(self.lut_pfad)
        self.edit_lut.setToolTip("Pfad zur robot_lut.h (generiert von generate_lut.py)")
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(36)
        btn_browse.setStyleSheet(f"background-color: {C_PANEL}; color: {C_TEXT}; padding: 6px;")
        btn_browse.clicked.connect(self._waehle_lut_datei)
        h_lut.addWidget(self.edit_lut)
        h_lut.addWidget(btn_browse)
        ll.addLayout(h_lut)
        sb.addWidget(grp_lut)

        # Spielfeld
        grp_feld = QGroupBox("Spielfeld")
        lf = QVBoxLayout(grp_feld)
        lf.addWidget(QLabel("Breite (m):"))
        self.spin_fw = QDoubleSpinBox()
        self.spin_fw.setRange(1.0, 5.0)
        self.spin_fw.setValue(3.0)
        self.spin_fw.setSingleStep(0.5)
        lf.addWidget(self.spin_fw)
        lf.addWidget(QLabel("Höhe (m):"))
        self.spin_fh = QDoubleSpinBox()
        self.spin_fh.setRange(1.0, 5.0)
        self.spin_fh.setValue(3.0)
        self.spin_fh.setSingleStep(0.5)
        lf.addWidget(self.spin_fh)
        sb.addWidget(grp_feld)

        # Ball
        grp_ball = QGroupBox("Ball-Position")
        lb = QVBoxLayout(grp_ball)
        lb.addWidget(QLabel("X / Y (m):"))
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
        lb.addLayout(hb)
        sb.addWidget(grp_ball)

        # Analyse-Einstellungen
        grp_ana = QGroupBox("Analyse-Einstellungen")
        la = QVBoxLayout(grp_ana)

        la.addWidget(QLabel("Gitterschritt (cm):"))
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 30)
        self.spin_step.setValue(5)
        la.addWidget(self.spin_step)

        la.addWidget(QLabel("Orientierungen (1 = nur 0°, max 36):"))
        self.spin_ori = QSpinBox()
        self.spin_ori.setRange(1, 36)
        self.spin_ori.setValue(8)
        la.addWidget(self.spin_ori)

        la.addWidget(QLabel("Max. Schritte / Simulation:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(50, 400)
        self.spin_steps.setValue(200)
        la.addWidget(self.spin_steps)

        sb.addWidget(grp_ana)

        # Vektorfeld-Einstellungen
        grp_vf = QGroupBox("Vektorfeld")
        lv = QVBoxLayout(grp_vf)
        lv.addWidget(QLabel("Roboter-Orientierung (°):"))
        self.spin_vf_ori = QSpinBox()
        self.spin_vf_ori.setRange(0, 359)
        self.spin_vf_ori.setValue(0)
        lv.addWidget(self.spin_vf_ori)

        lv.addWidget(QLabel("Einfärben nach:"))
        self.combo_vf_color = QComboBox()
        self.combo_vf_color.addItems([
            "Fahrtrichtung (HSV)",
            "Erfolgsrate",
        ])
        lv.addWidget(self.combo_vf_color)

        self.btn_vf_update = QPushButton("🔄 Vektorfeld neu zeichnen")
        self.btn_vf_update.setStyleSheet(
            f"background-color:{C_ACCENT2}; color:{C_BG};")
        self.btn_vf_update.clicked.connect(self._vektorfeld_aktualisieren)
        self.btn_vf_update.setEnabled(False)
        lv.addWidget(self.btn_vf_update)
        sb.addWidget(grp_vf)

        # Aktions-Buttons
        self.btn_analyse = QPushButton("🔍  LUT-Analyse starten")
        self.btn_analyse.setStyleSheet(
            f"background-color:{C_ACCENT}; color:{C_BG};")
        self.btn_analyse.clicked.connect(self._starte_analyse)
        sb.addWidget(self.btn_analyse)

        self.btn_stop = QPushButton("🛑  Stopp")
        self.btn_stop.setStyleSheet(f"background-color:{C_DANGER};")
        self.btn_stop.clicked.connect(self._stoppe_analyse)
        self.btn_stop.setEnabled(False)
        sb.addWidget(self.btn_stop)

        # Fortschritt
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        sb.addWidget(self.progress)

        self.lbl_status = QLabel("Bereit.")
        self.lbl_status.setStyleSheet(
            f"color:{C_MUTED}; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        sb.addWidget(self.lbl_status)

        sb.addStretch()
        sidebar_scroll.setWidget(sidebar)
        main.addWidget(sidebar_scroll)

        # ── Tabs ─────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        main.addWidget(self.tabs, stretch=1)

        # Tab 1 – Vektorfeld
        self.fig_vf = Figure(figsize=(6, 6))
        self.fig_vf.patch.set_facecolor(C_BG)
        self.canvas_vf = FigureCanvas(self.fig_vf)
        tab1 = QWidget()
        l1 = QVBoxLayout(tab1)
        l1.setContentsMargins(0, 0, 0, 0)
        l1.addWidget(self.canvas_vf)
        self.tabs.addTab(tab1, "🧭  Vektorfeld")

        # Tab 2 – Heatmaps (2×2 → jetzt 1×3, kein Q-Wert)
        self.fig_heat = Figure(figsize=(10, 5))
        self.fig_heat.patch.set_facecolor(C_BG)
        self.canvas_heat = FigureCanvas(self.fig_heat)
        tab2 = QWidget()
        l2 = QVBoxLayout(tab2)
        l2.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(self.canvas_heat)
        self.tabs.addTab(tab2, "🌡️  Heatmaps")

        # Tab 3 – Statistiken
        self.fig_stat = Figure(figsize=(11, 7))
        self.fig_stat.patch.set_facecolor(C_BG)
        self.canvas_stat = FigureCanvas(self.fig_stat)
        tab3 = QWidget()
        l3 = QVBoxLayout(tab3)
        l3.setContentsMargins(0, 0, 0, 0)
        l3.addWidget(self.canvas_stat)
        self.tabs.addTab(tab3, "📊  Statistiken")

        # Tab 4 – Orientierungs-Polar-Plot
        self.fig_polar = Figure(figsize=(6, 6))
        self.fig_polar.patch.set_facecolor(C_BG)
        self.canvas_polar = FigureCanvas(self.fig_polar)
        tab4 = QWidget()
        l4 = QVBoxLayout(tab4)
        l4.setContentsMargins(0, 0, 0, 0)
        l4.addWidget(self.canvas_polar)
        self.tabs.addTab(tab4, "🔄  Orientierung")

    # ──────────────────────────────────────────────────────────────────────────
    def _waehle_lut_datei(self):
        pfad, _ = QFileDialog.getOpenFileName(
            self, "LUT-Datei wählen", os.path.dirname(self.edit_lut.text()),
            "Header-Dateien (*.h);;Alle Dateien (*)"
        )
        if pfad:
            self.edit_lut.setText(pfad)

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_placeholder(self):
        msg = ("LUT-Analyse noch nicht gestartet.\n"
               "LUT-Datei wählen und  🔍 LUT-Analyse starten  klicken.")
        for fig in (self.fig_vf, self.fig_heat,
                    self.fig_stat, self.fig_polar):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor("#1a1a2e")
            ax.text(0.5, 0.5, msg,
                    transform=ax.transAxes, ha="center", va="center",
                    color=C_MUTED, fontsize=13, style="italic",
                    multialignment="center")
            ax.axis("off")
        for c in (self.canvas_vf, self.canvas_heat,
                  self.canvas_stat, self.canvas_polar):
            c.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _starte_analyse(self):
        lut_datei = self.edit_lut.text().strip()
        if not lut_datei:
            self.lbl_status.setStyleSheet(f"color:{C_DANGER}; font-size:11px;")
            self.lbl_status.setText("Fehler: Keine LUT-Datei angegeben.")
            return

        self.btn_analyse.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_vf_update.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_status.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        self.lbl_status.setText("Starte …")

        self.worker = LUTAnalyseWorker(
            lut_datei        = lut_datei,
            feld_w           = self.spin_fw.value(),
            feld_h           = self.spin_fh.value(),
            b_x              = self.spin_bx.value(),
            b_y              = self.spin_by.value(),
            schritt_cm       = self.spin_step.value(),
            n_orientierungen = self.spin_ori.value(),
            max_schritte     = self.spin_steps.value(),
        )
        self.worker.fortschritt_signal.connect(self._on_fortschritt)
        self.worker.ergebnis_signal.connect(self._on_ergebnis)
        self.worker.fehler_signal.connect(self._on_fehler)
        self.worker.start()

    def _stoppe_analyse(self):
        if self.worker:
            self.worker.stop()
        self.btn_analyse.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Gestoppt.")

    def _on_fortschritt(self, pct: int, text: str):
        self.progress.setValue(pct)
        self.lbl_status.setText(text)

    def _on_fehler(self, msg: str):
        self.lbl_status.setStyleSheet(
            f"color:{C_DANGER}; font-size:11px;")
        self.lbl_status.setText(f"Fehler: {msg}")
        self.btn_analyse.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_ergebnis(self, ergebnisse: dict):
        self.ergebnisse = ergebnisse
        self.btn_analyse.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_vf_update.setEnabled(True)
        self.lbl_status.setStyleSheet(
            f"color:{C_SUCCESS}; font-size:11px;")
        self.lbl_status.setText("✅ LUT-Analyse abgeschlossen!")
        self._zeichne_vektorfeld(ergebnisse)
        self._zeichne_heatmaps(ergebnisse)
        self._zeichne_statistiken(ergebnisse)
        self._zeichne_orientierung(ergebnisse)

    # ──────────────────────────────────────────────────────────────────────────
    def _vektorfeld_aktualisieren(self):
        """Vektorfeld mit neuer Orientierung neu berechnen und zeichnen."""
        if self.ergebnisse is None:
            return
        vf_ori = self.spin_vf_ori.value()
        if vf_ori == 0:
            self._zeichne_vektorfeld(self.ergebnisse)
            return

        lut_datei = self.edit_lut.text().strip()
        try:
            lut = lade_lut(lut_datei)
        except Exception as e:
            self.lbl_status.setText(f"Fehler: {e}")
            return

        d = self.ergebnisse
        xs, ys = d["xs"], d["ys"]
        nx, ny = len(xs), len(ys)

        u = np.zeros((ny, nx), dtype=np.float32)
        v = np.zeros((ny, nx), dtype=np.float32)
        w = np.full((ny, nx), np.nan, dtype=np.float32)

        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                rel_w, dist_cm = berechne_zustand(
                    x, y, vf_ori, d["b_x"], d["b_y"])
                if dist_cm > 0:
                    a = lut_nachschlagen(lut, rel_w, dist_cm)
                    glo = (math.radians(vf_ori)
                           + math.radians(a * WINKEL_SCHRITT))
                    u[iy, ix] = math.sin(glo)
                    v[iy, ix] = math.cos(glo)
                    w[iy, ix] = (math.degrees(glo) + 360) % 360

        d_copy = dict(d)
        d_copy["vf_u"]     = u
        d_copy["vf_v"]     = v
        d_copy["vf_winkel"] = w
        self._zeichne_vektorfeld(d_copy)

    # ──────────────────────────────────────────────────────────────────────────
    def _zeichne_vektorfeld(self, d: dict):
        """Tab 1: Pfeil-Vektorfeld der berechneten Fahrtrichtung."""
        vf_ori     = self.spin_vf_ori.value()
        color_mode = self.combo_vf_color.currentText()

        self.fig_vf.clear()
        ax = self.fig_vf.add_subplot(111)
        _style_ax(ax,
                  title=(f"LUT Vektorfeld der Fahrtrichtung  "
                         f"(Roboter-Orientierung: {vf_ori}°)"),
                  xlabel="X (m)", ylabel="Y (m)")

        xs, ys   = d["xs"], d["ys"]
        fw, fh   = d["feld_w"], d["feld_h"]
        b_x, b_y = d["b_x"], d["b_y"]
        u, v     = d["vf_u"], d["vf_v"]

        # Normiere auf Einheitsvektor für gleichmäßige Pfeile
        mag = np.sqrt(u**2 + v**2)
        mag[mag == 0] = 1.0
        u_n = u / mag
        v_n = v / mag

        step = d["schritt_m"]
        pfeil_len = step * ARROW_LENGTH_FACTOR

        XX, YY = np.meshgrid(xs, ys)

        # Farb-Array
        if color_mode == "Erfolgsrate" and "erfolg_map" in d:
            C_arr  = d["erfolg_map"].copy()
            C_arr[np.isnan(C_arr)] = 0.0
            cmap   = "RdYlGn"
            norm   = mcolors.Normalize(vmin=0, vmax=1)
            clabel = "Erfolgsrate"
        else:
            # Fahrtrichtung in Grad → HSV
            C_arr  = d["vf_winkel"].copy()
            C_arr[np.isnan(C_arr)] = 0.0
            cmap   = "hsv"
            norm   = mcolors.Normalize(vmin=0, vmax=360)
            clabel = "Fahrtrichtung (°)"

        q = ax.quiver(
            XX, YY,
            u_n * pfeil_len, v_n * pfeil_len,
            C_arr,
            cmap=cmap, norm=norm,
            units="xy", scale=1.0,
            width=step * ARROW_WIDTH_FACTOR,
            headwidth=4, headlength=5, headaxislength=4.5,
            alpha=0.85,
        )

        _add_colorbar(self.fig_vf, q, ax, clabel)
        _draw_ball_and_border(ax, b_x, b_y, fw, fh)
        ax.set_xlim(-step, fw + step)
        ax.set_ylim(-step, fh + step)
        ax.set_aspect("equal")
        ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER,
                  labelcolor=C_TEXT, fontsize=8)

        self.fig_vf.tight_layout()
        self.canvas_vf.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _zeichne_heatmaps(self, d: dict):
        """Tab 2: 1×3-Heatmap-Grid (kein Q-Wert, da LUT)."""
        self.fig_heat.clear()
        axes = self.fig_heat.subplots(1, 3)

        xs, ys   = d["xs"], d["ys"]
        fw, fh   = d["feld_w"], d["feld_h"]
        b_x, b_y = d["b_x"], d["b_y"]
        extent   = [xs[0], xs[-1], ys[0], ys[-1]]

        def add_overlay(ax):
            _draw_ball_and_border(ax, b_x, b_y, fw, fh)
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
            ax.set_aspect("equal")

        # ── 1. Erfolgsrate ────────────────────────────────────────────────────
        ax = axes[0]
        im = ax.imshow(d["erfolg_map"], origin="lower", extent=extent,
                       cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        add_overlay(ax)
        _style_ax(ax, "Erfolgsrate  (Ø über Orientierungen)",
                  "X (m)", "Y (m)")
        _add_colorbar(self.fig_heat, im, ax, "Rate  [0 – 1]")

        # ── 2. Ø Schritte bis Ergebnis ────────────────────────────────────────
        ax = axes[1]
        data_s = d["schritte_map"]
        fallback_max = float(d.get("max_schritte", 200))
        vmax_s = np.nanpercentile(data_s, 95) if not np.all(np.isnan(data_s)) else fallback_max
        im = ax.imshow(data_s, origin="lower", extent=extent,
                       cmap="plasma_r", vmin=0, vmax=vmax_s, aspect="auto",
                       interpolation="nearest")
        add_overlay(ax)
        _style_ax(ax, "Ø Schritte bis Ergebnis", "X (m)", "Y (m)")
        _add_colorbar(self.fig_heat, im, ax, "Schritte")

        # ── 3. Ø Score ────────────────────────────────────────────────────────
        ax = axes[2]
        data_sc = d["score_map"]
        if not np.all(np.isnan(data_sc)):
            vmin_sc = np.nanpercentile(data_sc, 5)
            vmax_sc = np.nanpercentile(data_sc, 95)
        else:
            vmin_sc, vmax_sc = 0, 1
        im = ax.imshow(data_sc, origin="lower", extent=extent,
                       cmap="viridis", vmin=vmin_sc, vmax=vmax_sc,
                       aspect="auto", interpolation="nearest")
        add_overlay(ax)
        _style_ax(ax, "Ø Score", "X (m)", "Y (m)")
        _add_colorbar(self.fig_heat, im, ax, "Score")

        self.fig_heat.suptitle(
            "LUT Positions-Analyse  –  Vogelperspektive",
            color=C_TEXT, fontsize=13, fontweight="bold")
        self.fig_heat.tight_layout()
        self.canvas_heat.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _zeichne_statistiken(self, d: dict):
        """Tab 3: Balkendiagramm, Abstands-Plot, Zusammenfassung."""
        self.fig_stat.clear()
        axes = self.fig_stat.subplots(1, 3)

        counts = d["outcome_counts"]
        total  = max(1, sum(counts.values()))

        # ── 1. Ergebnis-Verteilung ────────────────────────────────────────────
        ax = axes[0]
        labels = ["Erfolg", "Crash", "Wand", "Timeout"]
        vals   = [counts["erfolg"], counts["crash"],
                  counts["wand"],   counts["timeout"]]
        clrs   = [C_SUCCESS, C_DANGER, C_WARNING, C_MUTED]
        bars   = ax.bar(labels, vals, color=clrs,
                        edgecolor=C_BORDER, linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(1, total * 0.005),
                    f"{val:,}", ha="center", va="bottom",
                    color=C_TEXT, fontsize=8)
        _style_ax(ax, "Ergebnis-Verteilung", "Ergebnis", "Anzahl")

        # ── 2. Erfolgsrate vs. Abstand ────────────────────────────────────────
        ax = axes[1]
        by_dist = d.get("success_by_dist", [])
        if by_dist:
            dists_arr = np.array([x[0] for x in by_dist])
            succ_arr  = np.array([x[1] for x in by_dist])
            max_d     = max(dists_arr.max(), 1.0)
            bins      = np.linspace(0, max_d, DISTANCE_BINS)
            centers   = (bins[:-1] + bins[1:]) / 2
            rates     = []
            for i in range(len(bins) - 1):
                mask = (dists_arr >= bins[i]) & (dists_arr < bins[i + 1])
                rates.append(succ_arr[mask].mean() if mask.any() else np.nan)
            rates = np.array(rates)
            mask_v = ~np.isnan(rates)
            if mask_v.any():
                ax.plot(centers[mask_v], rates[mask_v],
                        color=C_ACCENT2, linewidth=2)
                ax.fill_between(centers[mask_v], rates[mask_v],
                                alpha=0.2, color=C_ACCENT2)
                ax.axhline(0.5, color=C_MUTED, linestyle="--",
                           alpha=0.5, linewidth=1)
            ax.set_ylim(0, 1.05)
        _style_ax(ax, "Erfolgsrate vs. Abstand",
                  "Abstand (cm)", "Erfolgsrate")

        # ── 3. Zusammenfassung (Text) ─────────────────────────────────────────
        ax = axes[2]
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)

        er   = counts["erfolg"] / total * 100
        cr   = counts["crash"]  / total * 100
        wr   = counts["wand"]   / total * 100
        tor  = counts["timeout"] / total * 100
        avg  = np.nanmean(d["erfolg_map"]) * 100
        n_v  = d.get("n_valide_pos", 0)
        xs, ys = d["xs"], d["ys"]
        step_cm = int(round(d["schritt_m"] * 100))
        lut_name = os.path.basename(d.get("lut_datei", "?"))

        rows = [
            ("LUT-Datei:",          lut_name),
            ("Gesamt-Tests:",       f"{total:,}"),
            ("Valide Positionen:",  f"{n_v:,}"),
            ("Grid-Auflösung:",     f"{step_cm} cm  "
                                    f"({len(xs)}×{len(ys)})"),
            ("Orientierungen:",
             str(len(d["orientierungen"]))),
            ("", ""),
            ("Erfolgsrate:",        f"{er:.1f} %"),
            ("Crash-Rate:",         f"{cr:.1f} %"),
            ("Wand-Rate:",          f"{wr:.1f} %"),
            ("Timeout-Rate:",       f"{tor:.1f} %"),
            ("", ""),
            ("Ø Erfolg/Position:",  f"{avg:.1f} %"),
        ]
        y = 0.97
        dy = 0.078
        for lbl, val in rows:
            if lbl == "":
                y -= dy * 0.5
                continue
            ax.text(0.02, y, lbl, transform=ax.transAxes,
                    color=C_MUTED, fontsize=9, va="top")
            ax.text(0.98, y, val, transform=ax.transAxes,
                    color=C_TEXT, fontsize=9, fontweight="bold",
                    va="top", ha="right")
            y -= dy
        ax.set_title("Zusammenfassung", color=C_TEXT,
                     fontsize=10, fontweight="bold", pad=6)

        self.fig_stat.suptitle("LUT Statistiken", color=C_TEXT,
                                fontsize=13, fontweight="bold")
        self.fig_stat.tight_layout()
        self.canvas_stat.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _zeichne_orientierung(self, d: dict):
        """Tab 4: Polar-Plot der Erfolgsrate nach Roboter-Orientierung."""
        self.fig_polar.clear()
        orientierungen = d["orientierungen"]
        rates          = d["success_by_orient"]

        # Keine validen Positionen oder zu wenige Orientierungen → Hinweistext
        no_data = d.get("n_valide_pos", 0) == 0 or np.all(np.isnan(rates))

        if len(orientierungen) < 2 or no_data:
            ax = self.fig_polar.add_subplot(111)
            ax.set_facecolor("#1a1a2e")
            msg = ("Nur 1 Orientierung getestet.\n"
                   "Erhöhe 'Orientierungen' für den Polar-Plot."
                   if len(orientierungen) < 2
                   else "Keine validen Positionen – Polar-Plot nicht verfügbar.")
            ax.text(0.5, 0.5, msg,
                    transform=ax.transAxes, ha="center", va="center",
                    color=C_MUTED, fontsize=12, multialignment="center")
            ax.axis("off")
            self.canvas_polar.draw()
            return

        # NaN-Werte auf 0 setzen, damit der Plot keine Lücken hat
        rates_clean = np.where(np.isnan(rates), 0.0, rates)

        ax = self.fig_polar.add_subplot(111, projection="polar")
        ax.set_facecolor("#1a1a2e")

        theta = np.radians(orientierungen)
        # Kurve schließen
        theta_c = np.append(theta, theta[0])
        rates_c = np.append(rates_clean, rates_clean[0])

        ax.plot(theta_c, rates_c, color=C_ACCENT, linewidth=2)
        ax.fill(theta_c, rates_c, alpha=0.25, color=C_ACCENT)

        r_max = max(float(rates_clean.max()) * 1.2, 0.05)
        ax.set_ylim(0, r_max)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)    # Uhrzeigersinn
        ax.set_thetagrids(
            np.arange(0, 360, 45),
            labels=["0° (N)", "45°", "90° (O)", "135°",
                    "180° (S)", "225°", "270° (W)", "315°"],
            color=C_TEXT, fontsize=8)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.grid(color=C_BORDER, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER)

        # Mittelwert-Linie
        mean_rate = float(rates_clean.mean())
        ax.plot([0, 2 * np.pi], [mean_rate, mean_rate],
                color=C_WARNING, linestyle="--", linewidth=1,
                alpha=0.7, label=f"Ø {mean_rate:.2f}")
        ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER,
                  labelcolor=C_TEXT, fontsize=8,
                  loc="upper right", bbox_to_anchor=(1.3, 1.1))

        ax.set_title("LUT Erfolgsrate nach Roboter-Orientierung",
                     color=C_TEXT, fontsize=11,
                     fontweight="bold", pad=22)

        self.fig_polar.tight_layout()
        self.canvas_polar.draw()


# ─── Einstiegspunkt ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LUTAnalyserWindow()
    window.show()
    sys.exit(app.exec_())
