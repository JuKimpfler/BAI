import sys
import math
import os
import re
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox,
                             QGroupBox, QSlider, QFileDialog, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ─── Farbpalette & Style ──────────────────────────────────────────────────────
C_BG        = "#0d0d1a"
C_SURFACE   = "#13132b"
C_PANEL     = "#1a1a38"
C_BORDER    = "#2a2a55"
C_ACCENT    = "#f59e0b"   # Amber/Orange für den LUT-Tester
C_SUCCESS   = "#22c55e"
C_DANGER    = "#ef4444"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#8892aa"

GLOBAL_STYLE = f"""
QMainWindow, QWidget {{ background-color: {C_BG}; color: {C_TEXT}; font-family: "Segoe UI", Arial; font-size: 13px; }}
QGroupBox {{ background-color: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 10px; margin-top: 18px; padding: 12px 10px 10px 10px; font-weight: bold; color: {C_MUTED}; text-transform: uppercase; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; left: 12px; }}
QDoubleSpinBox, QSpinBox, QLineEdit {{ background-color: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 6px; padding: 6px; color: {C_TEXT}; }}
QPushButton {{ border-radius: 8px; padding: 10px; font-weight: bold; border: none; color: white; }}
QPushButton:disabled {{ background-color: {C_BORDER}; color: {C_MUTED}; }}
QSlider::groove:horizontal {{ border: 1px solid {C_BORDER}; height: 8px; background: {C_PANEL}; border-radius: 4px; }}
QSlider::handle:horizontal {{ background: {C_ACCENT}; width: 14px; margin: -3px 0; border-radius: 7px; }}
"""

# ==========================================
# 1. LUT LOGIK
# ==========================================
ANZAHL_AKTIONEN  = 90
WINKEL_SCHRITT   = 360.0 / ANZAHL_AKTIONEN  # 4° pro Aktion
ANZAHL_ABSTAENDE = 201   # 0 – 200 cm (Schrittweite 1 cm)
ANZAHL_WINKEL    = 360   # 0 – 359 Grad (Schrittweite 1°)


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
    a = max(0, min(200, int(round(abstand_cm))))
    return lut[w * ANZAHL_ABSTAENDE + a]


def berechne_zustand(r_x, r_y, r_w, b_x, b_y):
    dx, dy = b_x - r_x, b_y - r_y
    abstand_cm = math.hypot(dx, dy) * 100
    abs_winkel_deg = math.degrees(math.atan2(dx, dy))
    rel_winkel = (abs_winkel_deg - r_w) % 360
    if rel_winkel > 180:
        rel_winkel -= 360
    return rel_winkel, abstand_cm


# ==========================================
# 2. SIMULATIONS-THREAD
# ==========================================
class SimulationWorker(QThread):
    frame_signal    = pyqtSignal(list, list, float, float, float)  # pfad_x, pfad_y, r_x, r_y, r_w
    status_signal   = pyqtSignal(str, str, int)                    # Meldung, Farbe, Punkte
    finished_signal = pyqtSignal()

    def __init__(self, lut_datei, start_pos, feld_dim):
        super().__init__()
        self.lut_datei = lut_datei
        self.r_x, self.r_y, self.r_w = start_pos[0], start_pos[1], start_pos[2]
        self.b_x, self.b_y = start_pos[3], start_pos[4]
        self.feld_w, self.feld_h = feld_dim[0], feld_dim[1]
        self.running = True

    def run(self):
        try:
            lut = lade_lut(self.lut_datei)
        except Exception as e:
            self.status_signal.emit(f"LUT nicht geladen: {e}", C_DANGER, 0)
            self.finished_signal.emit()
            return

        pfad_x, pfad_y = [self.r_x], [self.r_y]
        punkte = 0
        rob_radius_cm = 11.0
        toleranz = 8

        for schritt in range(400):
            if not self.running:
                break

            rw, dist = berechne_zustand(self.r_x, self.r_y, self.r_w, self.b_x, self.b_y)
            aktion = lut_nachschlagen(lut, rw, dist*0.7)

            ziel_rel_rad = math.radians(aktion * WINKEL_SCHRITT)
            global_rad = math.radians(self.r_w) + ziel_rel_rad

            # 2 cm Schritt
            self.r_x += 0.02 * math.sin(global_rad)
            self.r_y += 0.02 * math.cos(global_rad)

            pfad_x.append(self.r_x)
            pfad_y.append(self.r_y)

            neu_rw, neu_dist = berechne_zustand(self.r_x, self.r_y, self.r_w, self.b_x, self.b_y)

            # Status prüfen
            if neu_dist <= (rob_radius_cm + 2):
                if abs(neu_rw) <= toleranz:
                    punkte += 100
                    self.status_signal.emit("🎯 ZIEL ERREICHT! Perfekter Winkel.", C_SUCCESS, punkte)
                else:
                    punkte -= 10
                    self.status_signal.emit("💥 CRASH! Winkel zu steil.", C_DANGER, punkte)
                self.frame_signal.emit(pfad_x, pfad_y, self.r_x, self.r_y, self.r_w)
                break
            elif self.r_x < 0 or self.r_x > self.feld_w or self.r_y < 0 or self.r_y > self.feld_h:
                punkte -= 5
                self.status_signal.emit("🧱 WAND BERÜHRT!", C_DANGER, punkte)
                self.frame_signal.emit(pfad_x, pfad_y, self.r_x, self.r_y, self.r_w)
                break
            else:
                punkte -= 2  # Schritt-Abzug
                self.status_signal.emit("Fährt...", C_ACCENT, punkte)

            # Frame an GUI senden und kurz warten (Animation)
            self.frame_signal.emit(pfad_x, pfad_y, self.r_x, self.r_y, self.r_w)
            time.sleep(0.03)  # 30 ms = ca. 30 FPS

        self.finished_signal.emit()

    def stop(self):
        self.running = False


# ==========================================
# 3. GUI MAIN WINDOW
# ==========================================
class LUTTesterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboter LUT · Live Simulator")
        self.setMinimumSize(1000, 650)
        self.setStyleSheet(GLOBAL_STYLE)

        self.worker = None
        self.rob_durchmesser = 22.0
        self._default_lut_pfad()
        self.initUI()
        self.update_static_plot()  # Erstes Bild zeichnen

    def _default_lut_pfad(self):
        """Sucht nach robot_lut.h zuerst im Skript-Verzeichnis, dann in teensy/include/."""
        basis = os.path.dirname(os.path.abspath(__file__))
        kandidaten = [
            os.path.join(basis, "robot_lut.h"),
            os.path.join(basis, "teensy", "include", "robot_lut.h"),
        ]
        for k in kandidaten:
            if os.path.exists(k):
                self.lut_pfad = k
                return
        self.lut_pfad = kandidaten[1]  # Standardpfad auch wenn nicht vorhanden

    def initUI(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # --- SEITENLEISTE ---
        sidebar = QWidget()
        sidebar.setFixedWidth(290)
        sidebar.setStyleSheet(f"background-color: {C_SURFACE}; border-radius: 12px; border: 1px solid {C_BORDER};")
        sb_layout = QVBoxLayout(sidebar)

        # Feld & LUT-Datei
        grp_feld = QGroupBox("Feld & LUT")
        l_feld = QVBoxLayout(grp_feld)

        l_feld.addWidget(QLabel("Feld Breite (m):"))
        self.spin_fw = QDoubleSpinBox()
        self.spin_fw.setRange(1.0, 5.0); self.spin_fw.setValue(3.0); self.spin_fw.setSingleStep(0.5)
        self.spin_fw.valueChanged.connect(self.update_static_plot)
        l_feld.addWidget(self.spin_fw)

        l_feld.addWidget(QLabel("Feld Höhe (m):"))
        self.spin_fh = QDoubleSpinBox()
        self.spin_fh.setRange(1.0, 5.0); self.spin_fh.setValue(3.0); self.spin_fh.setSingleStep(0.5)
        self.spin_fh.valueChanged.connect(self.update_static_plot)
        l_feld.addWidget(self.spin_fh)

        l_feld.addWidget(QLabel("LUT-Datei (robot_lut.h):"))
        h_lut = QHBoxLayout()
        self.edit_lut = QLineEdit(self.lut_pfad)
        self.edit_lut.setToolTip("Pfad zur robot_lut.h Datei (generiert von generate_lut.py)")
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(36)
        btn_browse.setStyleSheet(f"background-color: {C_PANEL}; color: {C_TEXT}; padding: 6px;")
        btn_browse.clicked.connect(self.waehle_lut_datei)
        h_lut.addWidget(self.edit_lut)
        h_lut.addWidget(btn_browse)
        l_feld.addLayout(h_lut)
        sb_layout.addWidget(grp_feld)

        # Positionen
        grp_pos = QGroupBox("Start Positionen")
        l_pos = QVBoxLayout(grp_pos)

        l_pos.addWidget(QLabel("Roboter X / Y:"))
        h_rob = QHBoxLayout()
        self.s_rx = QDoubleSpinBox(); self.s_rx.setRange(0, 5); self.s_rx.setValue(0.5); self.s_rx.setSingleStep(0.1)
        self.s_ry = QDoubleSpinBox(); self.s_ry.setRange(0, 5); self.s_ry.setValue(0.5); self.s_ry.setSingleStep(0.1)
        h_rob.addWidget(self.s_rx); h_rob.addWidget(self.s_ry)
        l_pos.addLayout(h_rob)

        l_pos.addWidget(QLabel("Roboter Winkel:"))
        self.sl_winkel = QSlider(Qt.Horizontal)
        self.sl_winkel.setRange(-180, 180); self.sl_winkel.setValue(45)
        l_pos.addWidget(self.sl_winkel)

        l_pos.addSpacing(10)
        l_pos.addWidget(QLabel("Ball X / Y:"))
        h_ball = QHBoxLayout()
        self.s_bx = QDoubleSpinBox(); self.s_bx.setRange(0, 5); self.s_bx.setValue(1.5); self.s_bx.setSingleStep(0.1)
        self.s_by = QDoubleSpinBox(); self.s_by.setRange(0, 5); self.s_by.setValue(1.5); self.s_by.setSingleStep(0.1)
        h_ball.addWidget(self.s_bx); h_ball.addWidget(self.s_by)
        l_pos.addLayout(h_ball)

        self.s_rx.valueChanged.connect(self.update_static_plot)
        self.s_ry.valueChanged.connect(self.update_static_plot)
        self.s_bx.valueChanged.connect(self.update_static_plot)
        self.s_by.valueChanged.connect(self.update_static_plot)
        self.sl_winkel.valueChanged.connect(self.update_static_plot)
        sb_layout.addWidget(grp_pos)

        # Aktionen
        self.btn_start = QPushButton("▶️ LUT-Fahrt animieren")
        self.btn_start.setStyleSheet(f"background-color: {C_ACCENT}; color: {C_BG};")
        self.btn_start.clicked.connect(self.start_animation)
        sb_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("🛑 Stopp")
        self.btn_stop.setStyleSheet(f"background-color: {C_DANGER};")
        self.btn_stop.clicked.connect(self.stop_animation)
        self.btn_stop.setEnabled(False)
        sb_layout.addWidget(self.btn_stop)

        sb_layout.addStretch()
        layout.addWidget(sidebar)

        # --- HAUPTBEREICH (Plot) ---
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)

        status_bar = QHBoxLayout()
        self.lbl_info = QLabel("Status: Warte auf Start...")
        self.lbl_info.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {C_MUTED};")
        self.lbl_score = QLabel("Punkte: 0")
        self.lbl_score.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {C_TEXT};")
        status_bar.addWidget(self.lbl_info)
        status_bar.addStretch()
        status_bar.addWidget(self.lbl_score)
        main_layout.addLayout(status_bar)

        # Matplotlib Plot
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor(C_BG)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        layout.addWidget(main_area, stretch=1)

    def waehle_lut_datei(self):
        pfad, _ = QFileDialog.getOpenFileName(
            self, "LUT-Datei wählen", os.path.dirname(self.edit_lut.text()),
            "Header-Dateien (*.h);;Alle Dateien (*)"
        )
        if pfad:
            self.edit_lut.setText(pfad)

    def draw_field(self, pfad_x=None, pfad_y=None, curr_rx=None, curr_ry=None, curr_rw=None):
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")

        fw, fh = self.spin_fw.value(), self.spin_fh.value()
        bx, by = self.s_bx.value(), self.s_by.value()

        if curr_rx is None:
            curr_rx, curr_ry, curr_rw = self.s_rx.value(), self.s_ry.value(), self.sl_winkel.value()

        # Spielfeld Rand
        self.ax.add_patch(patches.Rectangle((0, 0), fw, fh, linewidth=3, edgecolor=C_BORDER, facecolor='none'))

        # Pfad
        if pfad_x and pfad_y:
            self.ax.plot(pfad_x, pfad_y, color=C_ACCENT, linestyle='--', linewidth=2, zorder=2)

        # Roboter
        rob_r = (self.rob_durchmesser / 2) / 100
        self.ax.add_patch(plt.Circle((curr_rx, curr_ry), rob_r, color='#555577', zorder=3))
        # Rote Blickrichtungs-Linie
        front_x = curr_rx + math.sin(math.radians(curr_rw)) * rob_r
        front_y = curr_ry + math.cos(math.radians(curr_rw)) * rob_r
        self.ax.plot([curr_rx, front_x], [curr_ry, front_y], color=C_DANGER, linewidth=3, zorder=4)

        # Ball
        self.ax.add_patch(plt.Circle((bx, by), 0.03, color='white', zorder=3))

        self.ax.set_xlim(-0.2, fw + 0.2)
        self.ax.set_ylim(-0.2, fh + 0.2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()

    def update_static_plot(self):
        # Wird gerufen wenn Slider bewegt werden
        if self.worker and self.worker.running:
            return
        self.draw_field()

    def start_animation(self):
        lut_datei = self.edit_lut.text().strip()
        start_pos = [self.s_rx.value(), self.s_ry.value(), self.sl_winkel.value(),
                     self.s_bx.value(), self.s_by.value()]
        feld_dim = [self.spin_fw.value(), self.spin_fh.value()]

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = SimulationWorker(lut_datei, start_pos, feld_dim)
        self.worker.frame_signal.connect(self.draw_field)
        self.worker.status_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.animation_finished)
        self.worker.start()

    def update_status(self, msg, color, punkte):
        self.lbl_info.setText(f"Status: {msg}")
        self.lbl_info.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        self.lbl_score.setText(f"Punkte: {punkte}")

    def stop_animation(self):
        if self.worker:
            self.worker.stop()

    def animation_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LUTTesterWindow()
    window.show()
    sys.exit(app.exec_())
