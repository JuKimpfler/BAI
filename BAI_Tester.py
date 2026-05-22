import sys
import math
import os
import time
import numpy as np
from sensor_model import simuliere_ball_sensor_abstand

import torch
import torch.nn as nn

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox,
                             QSpinBox, QGroupBox, QComboBox, QSlider, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ─── Farbpalette & Style (Passend zum Trainer) ──────────────────────────────
C_BG        = "#0d0d1a"
C_SURFACE   = "#13132b"
C_PANEL     = "#1a1a38"
C_BORDER    = "#2a2a55"
C_ACCENT    = "#00d4ff"   # Cyan für den Tester
C_SUCCESS   = "#22c55e"
C_DANGER    = "#ef4444"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#8892aa"

GLOBAL_STYLE = f"""
QMainWindow, QWidget {{ background-color: {C_BG}; color: {C_TEXT}; font-family: "Segoe UI", Arial; font-size: 13px; }}
QGroupBox {{ background-color: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 10px; margin-top: 18px; padding: 12px 10px 10px 10px; font-weight: bold; color: {C_MUTED}; text-transform: uppercase; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; left: 12px; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background-color: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 6px; padding: 6px; color: {C_TEXT}; }}
QPushButton {{ border-radius: 8px; padding: 10px; font-weight: bold; border: none; color: white; }}
QPushButton:disabled {{ background-color: {C_BORDER}; color: {C_MUTED}; }}
QSlider::groove:horizontal {{ border: 1px solid {C_BORDER}; height: 8px; background: {C_PANEL}; border-radius: 4px; }}
QSlider::handle:horizontal {{ background: {C_ACCENT}; width: 14px; margin: -3px 0; border-radius: 7px; }}
"""

# ==========================================
# 1. KI MODELL & LOGIK
# ==========================================
# HIER ANGEPASST: 90 Aktionen für stufenlose 4-Grad-Schritte
ANZAHL_AKTIONEN = 90
WINKEL_SCHRITT = 360.0 / ANZAHL_AKTIONEN

class RoboterDQN(nn.Module):
    def __init__(self, neuronen=64):
        super().__init__()
        self.netzwerk = nn.Sequential(
            nn.Linear(3, neuronen), nn.ReLU(),
            nn.Linear(neuronen, neuronen), nn.ReLU(),
            nn.Linear(neuronen, ANZAHL_AKTIONEN)
        )
    def forward(self, x): return self.netzwerk(x)

def normalisiere_zustand(winkel_deg, abstand_cm, max_dist_cm):
    winkel_rad = math.radians(winkel_deg)
    sensor_abstand_cm = simuliere_ball_sensor_abstand(abstand_cm)
    return [math.sin(winkel_rad), math.cos(winkel_rad), sensor_abstand_cm / max_dist_cm]

def berechne_zustand(r_x, r_y, r_w, b_x, b_y):
    dx, dy = b_x - r_x, b_y - r_y
    abstand_cm = math.hypot(dx, dy) * 100
    abs_winkel_deg = math.degrees(math.atan2(dx, dy))
    rel_winkel = (abs_winkel_deg - r_w) % 360
    if rel_winkel > 180: rel_winkel -= 360
    return rel_winkel, abstand_cm

# ==========================================
# 2. SIMULATIONS-THREAD (Für flüssige Animation)
# ==========================================
class SimulationWorker(QThread):
    frame_signal = pyqtSignal(list, list, float, float, float) # pfad_x, pfad_y, r_x, r_y, r_w
    status_signal = pyqtSignal(str, str, int) # Meldung, Farbe, Punkte
    finished_signal = pyqtSignal()

    def __init__(self, modell_datei, neuronen, start_pos, feld_dim):
        super().__init__()
        self.modell_datei = modell_datei
        self.neuronen = neuronen
        self.r_x, self.r_y, self.r_w = start_pos[0], start_pos[1], start_pos[2]
        self.b_x, self.b_y = start_pos[3], start_pos[4]
        self.feld_w, self.feld_h = feld_dim[0], feld_dim[1]
        self.running = True

    def run(self):
        try:
            modell = RoboterDQN(self.neuronen)
            modell.load_state_dict(torch.load(self.modell_datei))
            modell.eval()
        except:
            self.status_signal.emit("Modell nicht gefunden! Bitte erst trainieren.", C_DANGER, 0)
            self.finished_signal.emit()
            return

        max_dist = math.hypot(self.feld_w, self.feld_h) * 100
        pfad_x, pfad_y = [self.r_x], [self.r_y]
        punkte = 0
        rob_radius_cm = 11.0
        toleranz = 8

        with torch.no_grad():
            for schritt in range(400):
                if not self.running: break

                rw, dist = berechne_zustand(self.r_x, self.r_y, self.r_w, self.b_x, self.b_y)
                zustand = normalisiere_zustand(rw, dist, max_dist)
                t_z = torch.tensor([zustand], dtype=torch.float32)
                aktion = torch.argmax(modell(t_z)).item()

                ziel_rel_rad = math.radians(aktion * WINKEL_SCHRITT)
                global_rad = math.radians(self.r_w) + ziel_rel_rad
                
                # 2cm Schritt
                self.r_x += 0.02 * math.sin(global_rad)
                self.r_y += 0.02 * math.cos(global_rad)
                
                pfad_x.append(self.r_x)
                pfad_y.append(self.r_y)

                neu_rw, neu_dist = berechne_zustand(self.r_x, self.r_y, self.r_w, self.b_x, self.b_y)
                
                # Status prüfen
                if neu_dist <= (rob_radius_cm + 2):
                    if abs(neu_rw) <= toleranz:
                        punkte += 10000
                        self.status_signal.emit("🎯 ZIEL ERREICHT! Perfekter Winkel.", C_SUCCESS, punkte)
                    else:
                        punkte -= 1000
                        self.status_signal.emit("💥 CRASH! Winkel zu steil.", C_DANGER, punkte)
                    self.frame_signal.emit(pfad_x, pfad_y, self.r_x, self.r_y, self.r_w)
                    break
                elif self.r_x < 0 or self.r_x > self.feld_w or self.r_y < 0 or self.r_y > self.feld_h:
                    punkte -= 500
                    self.status_signal.emit("🧱 WAND BERÜHRT!", C_DANGER, punkte)
                    self.frame_signal.emit(pfad_x, pfad_y, self.r_x, self.r_y, self.r_w)
                    break
                else:
                    punkte -= 1 # Schritt-Abzug
                    self.status_signal.emit("Fährt...", C_ACCENT, punkte)

                # Frame an GUI senden und kurz warten (Animation)
                self.frame_signal.emit(pfad_x, pfad_y, self.r_x, self.r_y, self.r_w)
                time.sleep(0.03) # 30ms = ca. 30 FPS

        self.finished_signal.emit()

    def stop(self):
        self.running = False


# ==========================================
# 3. GUI MAIN WINDOW
# ==========================================
class TesterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboter RL · Live Simulator")
        self.setMinimumSize(1000, 650)
        self.setStyleSheet(GLOBAL_STYLE)
        
        self.worker = None
        self.rob_durchmesser = 22.0
        self.initUI()
        self.update_static_plot() # Erstes Bild zeichnen

    def initUI(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # --- SEITENLEISTE ---
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet(f"background-color: {C_SURFACE}; border-radius: 12px; border: 1px solid {C_BORDER};")
        sb_layout = QVBoxLayout(sidebar)

        # Feld Setup
        grp_feld = QGroupBox("Feld & KI")
        l_feld = QVBoxLayout(grp_feld)
        
        l_feld.addWidget(QLabel("Feld Breite (m):"))
        self.spin_fw = QDoubleSpinBox(); self.spin_fw.setRange(1.0, 5.0); self.spin_fw.setValue(3.0); self.spin_fw.setSingleStep(0.5)
        self.spin_fw.valueChanged.connect(self.update_static_plot)
        l_feld.addWidget(self.spin_fw)

        l_feld.addWidget(QLabel("Feld Höhe (m):"))
        self.spin_fh = QDoubleSpinBox(); self.spin_fh.setRange(1.0, 5.0); self.spin_fh.setValue(3.0); self.spin_fh.setSingleStep(0.5)
        self.spin_fh.valueChanged.connect(self.update_static_plot)
        l_feld.addWidget(self.spin_fh)

        l_feld.addWidget(QLabel("Modell Neuronen:"))
        self.combo_nn = QComboBox(); self.combo_nn.addItems(["64", "128", "256", "400"]); self.combo_nn.setCurrentText("64")
        l_feld.addWidget(self.combo_nn)
        sb_layout.addWidget(grp_feld)

        # Positionen
        grp_pos = QGroupBox("Start Positionen")
        l_pos = QVBoxLayout(grp_pos)
        
        l_pos.addWidget(QLabel("Roboter X / Y / Winkel:"))
        h_rob = QHBoxLayout()
        self.s_rx = QDoubleSpinBox(); self.s_rx.setRange(0, 5); self.s_rx.setValue(0.5); self.s_rx.setSingleStep(0.1)
        self.s_ry = QDoubleSpinBox(); self.s_ry.setRange(0, 5); self.s_ry.setValue(0.5); self.s_ry.setSingleStep(0.1)
        h_rob.addWidget(self.s_rx); h_rob.addWidget(self.s_ry)
        l_pos.addLayout(h_rob)
        
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
        self.btn_start = QPushButton("▶️ Test-Fahrt animieren")
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
        
        # Status Leiste
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

    def draw_field(self, pfad_x=None, pfad_y=None, curr_rx=None, curr_ry=None, curr_rw=None):
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e") # Dunkles Rasen-Blau/Grün
        
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
        if self.worker and self.worker.running: return
        self.draw_field()

    def start_animation(self):
        neuronen = int(self.combo_nn.currentText())
        modell_datei = f"roboter_rl_modell_{neuronen}.pth"
        start_pos = [self.s_rx.value(), self.s_ry.value(), self.sl_winkel.value(), self.s_bx.value(), self.s_by.value()]
        feld_dim = [self.spin_fw.value(), self.spin_fh.value()]

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = SimulationWorker(modell_datei, neuronen, start_pos, feld_dim)
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
    window = TesterWindow()
    window.show()
    sys.exit(app.exec_())
