import sys
import math
import random
import os
from collections import deque, defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QProgressBar, QGroupBox, QComboBox, QFrame,
                             QSizePolicy, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ─── Farbpalette ──────────────────────────────────────────────────────────────
C_BG        = "#0d0d1a"   # Tiefstes Hintergrundblau
C_SURFACE   = "#13132b"   # Karten / Panels
C_PANEL     = "#1a1a38"   # leicht helleres Panel
C_BORDER    = "#2a2a55"   # Rahmenfarbe
C_ACCENT    = "#6c63ff"   # Primärakzent – Violett
C_ACCENT2   = "#00d4ff"   # Sekundärakzent – Cyan
C_SUCCESS   = "#22c55e"   # Grün
C_WARNING   = "#f59e0b"   # Orange
C_DANGER    = "#ef4444"   # Rot
C_TEXT      = "#e2e8f0"   # Primärtext
C_MUTED     = "#8892aa"   # Gedämpfter Text

GLOBAL_STYLE = f"""
/* ── Basis ── */
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}

/* ── GroupBox ── */
QGroupBox {{
    background-color: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    font-size: 12px;
    font-weight: 600;
    color: {C_MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
    color: {C_MUTED};
}}

/* ── Label ── */
QLabel {{
    background: transparent;
    color: {C_TEXT};
}}

/* ── SpinBox ── */
QSpinBox {{
    background-color: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
}}
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {C_BORDER};
    border-radius: 3px;
    width: 18px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background-color: {C_ACCENT};
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {C_SURFACE};
    border: 1px solid {C_BORDER};
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
}}

/* ── ProgressBar ── */
QProgressBar {{
    background-color: {C_PANEL};
    border: none;
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_ACCENT},
        stop:1 {C_ACCENT2}
    );
    border-radius: 6px;
}}

/* ── Buttons ── */
QPushButton {{
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 700;
    font-size: 13px;
    border: none;
    color: white;
}}
QPushButton:disabled {{
    opacity: 0.4;
    background-color: {C_BORDER};
    color: {C_MUTED};
}}

/* ── ScrollArea ── */
QScrollArea {{
    border: none;
    background: transparent;
}}
"""

# ==========================================
# 1. KI MODELL & LOGIK
# ==========================================
ANZAHL_AKTIONEN = 90
WINKEL_SCHRITT = 360.0 / ANZAHL_AKTIONEN

# ─── Trainings-Stabilitäts-Parameter ─────────────────────────────────────────
MEMORY_SIZE           = 100_000   # Größerer Replay-Puffer für langfristiges Training
LEARNING_RATE         = 0.0005    # Reduzierte Lernrate für stabileres Lernen
GRADIENT_CLIP_NORM    = 500.0     # Gelockert auf 500.0 für 10.000er Rewards
# ─── Reward-Konstanten ───────────────────────────────────────────────────────
REWARD_SUCCESSFUL_HIT = 10_000    # Ball mit korrektem Winkel getroffen
REWARD_CRASH          = -200      # Verringert von -1000, damit die KI keine Angst vorm Ball hat
REWARD_WALL           = -500      # Wand berührt
REWARD_STEP           = -1        # Zeitstrafe pro Schritt
# ─── Action-Jump-Penalty (Reward Shaping) ────────────────────────────────────
DEFAULT_ACTION_JUMP_PENALTY = 0.0 # Standardmäßig deaktiviert! (Stärke der Penalty)

class RoboterDQN(nn.Module):
    def __init__(self, neuronen=256):
        super().__init__()
        self.netzwerk = nn.Sequential(
            nn.Linear(2, neuronen),
            nn.ReLU(),
            nn.Linear(neuronen, neuronen),
            nn.ReLU(),
            nn.Linear(neuronen, ANZAHL_AKTIONEN)
        )
    def forward(self, x):
        return self.netzwerk(x)

def normalisiere_zustand(winkel_deg, abstand_cm, max_dist_cm):
    return [winkel_deg / 180.0, abstand_cm / max_dist_cm]

def berechne_zustand(r_x, r_y, r_w, b_x, b_y):
    dx = b_x - r_x
    dy = b_y - r_y
    abstand_cm = math.hypot(dx, dy) * 100
    abs_winkel_deg = math.degrees(math.atan2(dx, dy))
    rel_winkel = (abs_winkel_deg - r_w) % 360
    if rel_winkel > 180: rel_winkel -= 360
    return rel_winkel, abstand_cm

# ==========================================
# 2. DER TRAININGS-THREAD (Läuft im Hintergrund)
# ==========================================
class TrainingWorker(QThread):
    update_signal = pyqtSignal(int, float, float, float, float, float, float)  # Epoche, Epsilon, Reward, Hit-Rate, Loss, Effizienz, Ø-Schritte
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, epochen, neuronen, modell_datei, action_jump_penalty=0.0):
        super().__init__()
        self.epochen = epochen
        self.neuronen = neuronen
        self.modell_datei = modell_datei
        self.action_jump_penalty = action_jump_penalty
        self.running = True
        # Heatmap-Trainingsdaten: cell → deque of (epoche, reward)
        self.heatmap_cells: dict = defaultdict(lambda: deque(maxlen=1000))
        self.current_epoch = 0

    def run(self):
        modell = RoboterDQN(self.neuronen)
        ziel_modell = RoboterDQN(self.neuronen)
        
        feld_breite, feld_hoehe = 3.0, 3.0
        max_dist = math.hypot(feld_breite, feld_hoehe) * 100
        rob_radius_cm = 11.0
        toleranz = 20

        # Modell laden falls vorhanden
        if os.path.exists(self.modell_datei):
            modell.load_state_dict(torch.load(self.modell_datei))
            self.log_signal.emit("Setze bestehendes Training fort...")
            epsilon = 0.05  # Nur noch 5% Zufall bei bekannten Modellen!
        else:
            self.log_signal.emit("Starte komplett neues Training...")
            epsilon = 1.0

        ziel_modell.load_state_dict(modell.state_dict())
        optimizer = optim.Adam(modell.parameters(), lr=LEARNING_RATE)
        
        criterion = nn.MSELoss() 
        
        memory = deque(maxlen=MEMORY_SIZE)     # Größerer Replay-Puffer
        
        
        gamma = 0.95
        epsilon_min = 0.05
        ziel_epoche = int(self.epochen * 0.3)
        epsilon_decay = math.pow(epsilon_min / epsilon, 1.0 / ziel_epoche) if ziel_epoche > 0 else 0.995

        hit_history = deque(maxlen=100)   # Speichert die letzten 100 Ergebnisse (1 = Hit, 0 = Fail)
        belohnungen_fenster = []
        loss_val = 0.0

        # Analytics
        efficiency_fenster: deque = deque(maxlen=200)
        steps_hit_fenster: deque = deque(maxlen=200)

        # Heatmap-Binning
        N_ANGLE_BINS = 36   # je 10°, -180°…+180°
        N_DIST_BINS  = 30   # je ~10 cm, 0…300 cm

        def angle_bin(a: float) -> int:
            return int((a + 180.0) / 360.0 * N_ANGLE_BINS) % N_ANGLE_BINS

        def dist_bin(d: float) -> int:
            return min(int(d / 300.0 * N_DIST_BINS), N_DIST_BINS - 1)

        for epoche in range(self.epochen):
            if not self.running:
                break

            self.current_epoch = epoche

            b_x = random.uniform(0.5, feld_breite - 0.5)
            b_y = random.uniform(0.5, feld_hoehe - 0.5)
            
            r_x_start = random.uniform(0.2, feld_breite - 0.2)
            r_y_start = random.uniform(0.2, feld_hoehe - 0.2)
            r_x, r_y = r_x_start, r_y_start

            # --- SMART SPAWNING (70% Chance auf schwere Position) ---
            if random.random() < 0.70:
                abs_winkel = math.degrees(math.atan2(b_x - r_x, b_y - r_y))
                versatz = random.uniform(90, 270) # Ball ist im Rücken
                r_w = (abs_winkel + versatz) % 360
                if r_w > 180: r_w -= 360
            else:
                r_w = random.uniform(-180, 180)

            gesamt_belohnung = 0
            prev_aktion = None
            schritte = 0
            episode_hit = False
            
            for schritt in range(300):
                rel_w, dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                zustand = normalisiere_zustand(rel_w, dist, max_dist)
                
                if random.random() < epsilon:
                    aktion = random.randint(0, ANZAHL_AKTIONEN - 1)
                else:
                    with torch.no_grad():
                        aktion = torch.argmax(modell(torch.tensor([zustand], dtype=torch.float32))).item()

                # Action-Jump-Penalty: Kreisabstand im Aktionsraum
                belohnung_extra = 0.0
                if prev_aktion is not None and self.action_jump_penalty > 0.0:
                    diff = abs(aktion - prev_aktion)
                    circ_diff = min(diff, ANZAHL_AKTIONEN - diff)
                    belohnung_extra = -self.action_jump_penalty * circ_diff
                prev_aktion = aktion

                ziel_rel_rad = math.radians(aktion * WINKEL_SCHRITT)
                global_rad = math.radians(r_w) + ziel_rel_rad
                r_x += 0.02 * math.sin(global_rad)
                r_y += 0.02 * math.cos(global_rad)
                
                neu_rel_w, neu_dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                neuer_zustand = normalisiere_zustand(neu_rel_w, neu_dist, max_dist)
                
                belohnung = float(REWARD_STEP) + belohnung_extra
                done = False
                schritte += 1
                
                if neu_dist <= (rob_radius_cm + 2):
                    if abs(neu_rel_w) <= toleranz:
                        belohnung += REWARD_SUCCESSFUL_HIT
                        hit_history.append(1) # ERFOLG!
                        episode_hit = True
                    else:
                        belohnung += REWARD_CRASH
                        hit_history.append(0) # CRASH
                    done = True
                elif r_x < 0 or r_x > feld_breite or r_y < 0 or r_y > feld_hoehe:
                    belohnung += REWARD_WALL
                    hit_history.append(0) # WAND
                    done = True
                elif schritt == 299: # Timeout-Strafe! Zwingt den Roboter, sich zu beeilen!
                    belohnung += -500
                    hit_history.append(0)
                    done = True
                else:
                    # Faktor für Annäherung von 10 auf 2.0 verringert! Verhindert Punkte-Farmen
                    belohnung += (dist - neu_dist) * 2.0
                    
                gesamt_belohnung += belohnung
                memory.append((zustand, aktion, belohnung, neuer_zustand, done))

                # Heatmap-Besuch protokollieren
                self.heatmap_cells[(angle_bin(rel_w), dist_bin(dist))].append((epoche, belohnung))

                if done: break

            # Effizienz-Metrik (nur bei Treffer)
            if episode_hit:
                ideal_dist_m  = math.hypot(b_x - r_x_start, b_y - r_y_start)
                path_length_m = schritte * 0.02
                efficiency = min(ideal_dist_m / path_length_m, 1.0) if path_length_m > 0 else 0.0
                efficiency_fenster.append(efficiency)
                steps_hit_fenster.append(schritte)
                
            # Training aus dem Gedächtnis
            if len(memory) > 128:
                batch = random.sample(memory, 128)
                z_batch  = torch.tensor([x[0] for x in batch], dtype=torch.float32)
                a_batch  = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
                r_batch  = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                nz_batch = torch.tensor([x[3] for x in batch], dtype=torch.float32)
                d_batch  = torch.tensor([x[4] for x in batch], dtype=torch.float32)
                
                q_werte = modell(z_batch).gather(1, a_batch).squeeze()

                # Double DQN: Hauptmodell wählt Aktion, Zielmodell bewertet sie
                with torch.no_grad():
                    naechste_aktionen  = modell(nz_batch).argmax(1, keepdim=True)
                    naechste_q_werte   = ziel_modell(nz_batch).gather(1, naechste_aktionen).squeeze()

                erwartete_q_werte = r_batch + gamma * naechste_q_werte * (1 - d_batch)
                
                loss = criterion(q_werte, erwartete_q_werte.detach())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modell.parameters(), max_norm=GRADIENT_CLIP_NORM)
                optimizer.step()
                loss_val = loss.item()

            if epsilon > epsilon_min: epsilon *= epsilon_decay
            if epoche % 20 == 0: ziel_modell.load_state_dict(modell.state_dict())
            
            belohnungen_fenster.append(gesamt_belohnung)
                
            # GUI Update Interval (z.B. alle 10 Epochen)
            if epoche % 10 == 0 and len(belohnungen_fenster) > 0:
                durchschnitt_reward = sum(belohnungen_fenster) / len(belohnungen_fenster)
                hit_rate      = (sum(hit_history) / len(hit_history)) * 100 if len(hit_history) > 0 else 0
                avg_efficiency = sum(efficiency_fenster) / len(efficiency_fenster) if efficiency_fenster else 0.0
                avg_steps      = sum(steps_hit_fenster)  / len(steps_hit_fenster)  if steps_hit_fenster  else 0.0
                
                self.update_signal.emit(epoche, epsilon, durchschnitt_reward, hit_rate, loss_val,
                                        avg_efficiency, avg_steps)
                belohnungen_fenster.clear()

            # Auto-Save
            if epoche > 0 and epoche % 1000 == 0:
                torch.save(modell.state_dict(), self.modell_datei)
                self.log_signal.emit(f"Auto-Save bei Epoche {epoche} durchgeführt.")

        # Finales Speichern
        torch.save(modell.state_dict(), self.modell_datei)
        self.log_signal.emit(f"Training beendet! Gespeichert in {self.modell_datei}.")
        self.finished_signal.emit()

    def stop(self):
        self.running = False


# ==========================================
# 3. HILFS-WIDGETS
# ==========================================

class MetricCard(QFrame):
    """Kompakte Karte für eine einzelne Kennzahl."""

    def __init__(self, icon: str, title: str, initial: str, accent: str = C_ACCENT2):
        super().__init__()
        self.accent = accent
        self.setObjectName("MetricCard")
        self.setStyleSheet(f"""
            #MetricCard {{
                background-color: {C_PANEL};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(80)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(4)

        # Header: Icon + Titel
        header = QHBoxLayout()
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet(f"color: {accent}; font-size: 16px; background: transparent;")
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {C_MUTED}; font-size: 11px; font-weight: 600; "
                                f"text-transform: uppercase; letter-spacing: 0.5px; background: transparent;")
        header.addWidget(lbl_icon)
        header.addWidget(lbl_title)
        header.addStretch()
        outer.addLayout(header)

        # Wert
        self.lbl_value = QLabel(initial)
        self.lbl_value.setStyleSheet(f"color: {C_TEXT}; font-size: 20px; font-weight: 700; background: transparent;")
        outer.addWidget(self.lbl_value)

    def set_value(self, text: str, color: str = None):
        self.lbl_value.setText(text)
        c = color if color else C_TEXT
        self.lbl_value.setStyleSheet(f"color: {c}; font-size: 20px; font-weight: 700; background: transparent;")


class Divider(QFrame):
    def __init__(self, orientation=QFrame.HLine):
        super().__init__()
        self.setFrameShape(orientation)
        self.setStyleSheet(f"background-color: {C_BORDER}; border: none; max-height: 1px;")


# ==========================================
# 4. GUI (PyQt5)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboter RL · Trainings-Station")
        self.setMinimumSize(1100, 680)
        self.resize(1200, 760)

        self.worker = None
        self.reward_data = []
        self.hitrate_data = []
        self.epochen_data = []
        self.efficiency_data = []
        self.steps_data = []

        self._apply_theme()
        self.initUI()

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        self.setStyleSheet(GLOBAL_STYLE)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(C_BG))
        palette.setColor(QPalette.WindowText, QColor(C_TEXT))
        palette.setColor(QPalette.Base, QColor(C_PANEL))
        palette.setColor(QPalette.Text, QColor(C_TEXT))
        self.setPalette(palette)

    # ── Layout ────────────────────────────────────────────────────────────────
    def initUI(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header-Leiste ──────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1a0533, stop:0.5 #0d1f4a, stop:1 #001a3a);
            border-bottom: 1px solid {C_BORDER};
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 0, 24, 0)

        lbl_logo = QLabel("⬡")
        lbl_logo.setStyleSheet(f"color: {C_ACCENT}; font-size: 26px; background: transparent;")
        lbl_title = QLabel("Roboter RL  <span style='color:{C_MUTED}; font-weight:400;'>Trainings-Station</span>")
        lbl_title.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 700; background: transparent;")
        lbl_title.setTextFormat(Qt.RichText)
        lbl_version = QLabel("v2.0")
        lbl_version.setStyleSheet(f"""
            background-color: {C_ACCENT};
            color: white;
            border-radius: 8px;
            padding: 2px 10px;
            font-size: 11px;
            font-weight: 700;
        """)

        h_layout.addWidget(lbl_logo)
        h_layout.addSpacing(8)
        h_layout.addWidget(lbl_title)
        h_layout.addStretch()
        h_layout.addWidget(lbl_version)
        root_layout.addWidget(header)

        # ── Hauptbereich ───────────────────────────────────────────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(16)
        root_layout.addWidget(body, stretch=1)

        body_layout.addWidget(self._build_sidebar(), stretch=0)
        body_layout.addLayout(self._build_main_panel(), stretch=1)

        # ── Status-Leiste ──────────────────────────────────────────────────
        statusbar = QWidget()
        statusbar.setFixedHeight(32)
        statusbar.setStyleSheet(f"background-color: {C_SURFACE}; border-top: 1px solid {C_BORDER};")
        sb_layout = QHBoxLayout(statusbar)
        sb_layout.setContentsMargins(16, 0, 16, 0)
        self.lbl_status = QLabel("● Bereit")
        self.lbl_status.setStyleSheet(f"color: {C_SUCCESS}; font-size: 12px; background: transparent;")
        sb_layout.addWidget(self.lbl_status)
        sb_layout.addStretch()
        root_layout.addWidget(statusbar)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet(f"""
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 12px;
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Abschnitts-Header
        def section_label(text):
            lbl = QLabel(text.upper())
            lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 10px; font-weight: 700; "
                              f"letter-spacing: 1.5px; background: transparent;")
            return lbl

        # ── Konfiguration ──
        layout.addWidget(section_label("Konfiguration"))
        layout.addWidget(Divider())

        lbl_ep = QLabel("Epochen")
        lbl_ep.setStyleSheet(f"color: {C_MUTED}; font-size: 12px; background: transparent;")
        layout.addWidget(lbl_ep)
        self.spin_epochen = QSpinBox()
        self.spin_epochen.setRange(100, 500000)
        self.spin_epochen.setValue(50000)
        self.spin_epochen.setSingleStep(1000)
        self.spin_epochen.setFixedHeight(36)
        layout.addWidget(self.spin_epochen)

        lbl_nn = QLabel("KI-Neuronen")
        lbl_nn.setStyleSheet(f"color: {C_MUTED}; font-size: 12px; background: transparent;")
        layout.addWidget(lbl_nn)
        self.combo_neuronen = QComboBox()
        self.combo_neuronen.addItems(["64", "128", "256", "512"])
        self.combo_neuronen.setCurrentText("256")
        self.combo_neuronen.setFixedHeight(36)
        layout.addWidget(self.combo_neuronen)

        layout.addSpacing(8)
        layout.addWidget(section_label("Reward Shaping"))
        layout.addWidget(Divider())

        self.chk_jump_penalty = QPushButton("✓  Action-Jump-Penalty")
        self.chk_jump_penalty.setCheckable(True)
        self.chk_jump_penalty.setChecked(False) # <--- Standardmäßig deaktiviert!
        self.chk_jump_penalty.setFixedHeight(32)
        self.chk_jump_penalty.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_PANEL};
                color: {C_MUTED};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                text-align: left;
                padding-left: 8px;
            }}
            QPushButton:checked {{
                background-color: {C_ACCENT};
                color: white;
                border-color: {C_ACCENT};
            }}
        """)
        layout.addWidget(self.chk_jump_penalty)

        lbl_penalty = QLabel("Penalty-Stärke")
        lbl_penalty.setStyleSheet(f"color: {C_MUTED}; font-size: 12px; background: transparent;")
        layout.addWidget(lbl_penalty)
        self.spin_penalty = QSpinBox()
        self.spin_penalty.setRange(0, 50)
        self.spin_penalty.setValue(int(DEFAULT_ACTION_JUMP_PENALTY * 10))
        self.spin_penalty.setSuffix(" × 0.1")
        self.spin_penalty.setFixedHeight(36)
        layout.addWidget(self.spin_penalty)

        layout.addSpacing(8)

        # ── Aktionen ──
        layout.addWidget(section_label("Aktionen"))
        layout.addWidget(Divider())

        self.btn_start = QPushButton("▶  Training starten")
        self.btn_start.setFixedHeight(42)
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #16a34a, stop:1 #22c55e);
                color: white;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22c55e, stop:1 #4ade80);
            }}
            QPushButton:disabled {{
                background-color: {C_BORDER};
                color: {C_MUTED};
            }}
        """)
        self.btn_start.clicked.connect(self.start_training)
        layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  Training stoppen")
        self.btn_stop.setFixedHeight(42)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #b91c1c, stop:1 #ef4444);
                color: white;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ef4444, stop:1 #f87171);
            }}
            QPushButton:disabled {{
                background-color: {C_BORDER};
                color: {C_MUTED};
            }}
        """)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        layout.addSpacing(8)
        layout.addWidget(section_label("Analyse"))
        layout.addWidget(Divider())

        lbl_last_n = QLabel("Heatmap: letzte N Epochen")
        lbl_last_n.setStyleSheet(f"color: {C_MUTED}; font-size: 12px; background: transparent;")
        layout.addWidget(lbl_last_n)
        self.spin_last_n = QSpinBox()
        self.spin_last_n.setRange(50, 10000)
        self.spin_last_n.setValue(1000)
        self.spin_last_n.setSingleStep(100)
        self.spin_last_n.setFixedHeight(36)
        layout.addWidget(self.spin_last_n)

        layout.addSpacing(4)

        self.btn_heatmap = QPushButton("◈  Analyse Heatmap")
        self.btn_heatmap.setFixedHeight(38)
        self.btn_heatmap.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_PANEL};
                color: {C_ACCENT2};
                border: 1px solid {C_ACCENT2};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {C_ACCENT2};
                color: {C_BG};
            }}
        """)
        self.btn_heatmap.clicked.connect(self.show_heatmap)
        layout.addWidget(self.btn_heatmap)

        layout.addStretch()
        return sidebar

    # ── Hauptpanel ────────────────────────────────────────────────────────────
    def _build_main_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # ── Metriken-Zeile ──
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(10)

        self.card_epoche      = MetricCard("⏱", "Epoche",        "0 / 0",     C_ACCENT)
        self.card_hitrate     = MetricCard("🎯", "Trefferquote",  "0 %",       C_SUCCESS)
        self.card_epsilon     = MetricCard("🎲", "Zufall ε",      "100.0 %",   C_WARNING)
        self.card_loss        = MetricCard("📉", "Loss",          "—",         C_DANGER)
        self.card_efficiency  = MetricCard("📐", "Ø Effizienz",   "—",         C_ACCENT2)
        self.card_steps       = MetricCard("👣", "Ø Schritte/Hit","—",         C_MUTED)

        for card in (self.card_epoche, self.card_hitrate, self.card_epsilon,
                     self.card_loss, self.card_efficiency, self.card_steps):
            metrics_layout.addWidget(card)
        layout.addLayout(metrics_layout)

        # ── Fortschrittsleiste ──
        progress_container = QWidget()
        progress_container.setStyleSheet(f"background: transparent;")
        pc_layout = QVBoxLayout(progress_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(4)

        progress_header = QHBoxLayout()
        lbl_prog = QLabel("Trainingsfortschritt")
        lbl_prog.setStyleSheet(f"color: {C_MUTED}; font-size: 11px; font-weight: 600; background: transparent;")
        self.lbl_progress_pct = QLabel("0 %")
        self.lbl_progress_pct.setStyleSheet(f"color: {C_ACCENT2}; font-size: 11px; font-weight: 700; background: transparent;")
        progress_header.addWidget(lbl_prog)
        progress_header.addStretch()
        progress_header.addWidget(self.lbl_progress_pct)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(10)
        self.progress.setTextVisible(False)

        pc_layout.addLayout(progress_header)
        pc_layout.addWidget(self.progress)
        layout.addWidget(progress_container)

        # ── Chart-Bereich ──
        chart_frame = QFrame()
        chart_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 12px;
            }}
        """)
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(8, 8, 8, 8)

        # Matplotlib – dunkles Theme
        plt.style.use("dark_background")
        self.fig, (self.ax_reward, self.ax_hit) = plt.subplots(
            2, 1, figsize=(8, 4), sharex=True,
            gridspec_kw={"hspace": 0.08}
        )
        self.fig.patch.set_facecolor(C_SURFACE)
        for ax in (self.ax_reward, self.ax_hit):
            ax.set_facecolor("#0f0f22")
            ax.tick_params(colors=C_MUTED, labelsize=9)
            ax.spines[:].set_color(C_BORDER)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        self.ax_reward.set_ylabel("Ø Reward", color=C_MUTED, fontsize=9)
        self.ax_hit.set_ylabel("Trefferquote %", color=C_MUTED, fontsize=9)
        self.ax_hit.set_xlabel("Epoche", color=C_MUTED, fontsize=9)
        self.ax_hit.set_ylim(0, 105)
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background: transparent;")
        chart_layout.addWidget(self.canvas)
        layout.addWidget(chart_frame, stretch=1)

        return layout

    # ── Slots ─────────────────────────────────────────────────────────────────
    def start_training(self):
        neuronen = int(self.combo_neuronen.currentText())
        epochen = self.spin_epochen.value()
        modell_datei = f"roboter_rl_modell_{neuronen}.pth"

        self.reward_data.clear()
        self.hitrate_data.clear()
        self.epochen_data.clear()
        self.efficiency_data.clear()
        self.steps_data.clear()
        self.ax_reward.clear()
        self.ax_hit.clear()
        self.canvas.draw()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setMaximum(epochen)
        self.progress.setValue(0)
        self.lbl_progress_pct.setText("0 %")
        self._set_status("● Training läuft …", C_WARNING)

        penalty = (self.spin_penalty.value() * 0.1) if self.chk_jump_penalty.isChecked() else 0.0
        self.worker = TrainingWorker(epochen, neuronen, modell_datei, action_jump_penalty=penalty)
        self.worker.update_signal.connect(self.update_gui)
        self.worker.log_signal.connect(self._log)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self._log("Beende Training sicher … Bitte warten!")
            self._set_status("● Wird gestoppt …", C_WARNING)

    def training_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("● Training abgeschlossen", C_SUCCESS)

    def update_gui(self, epoche, epsilon, reward, hit_rate, loss, efficiency, avg_steps):
        epochen_total = self.spin_epochen.value()
        self.progress.setValue(epoche)
        pct = (epoche / epochen_total * 100) if epochen_total > 0 else 0
        self.lbl_progress_pct.setText(f"{pct:.1f} %")

        self.card_epoche.set_value(f"{epoche:,} / {epochen_total:,}")
        self.card_epsilon.set_value(f"{epsilon * 100:.1f} %")
        self.card_loss.set_value(f"{loss:.4f}")

        if hit_rate > 80:
            hr_color = C_SUCCESS
        elif hit_rate > 50:
            hr_color = C_WARNING
        else:
            hr_color = C_DANGER
        self.card_hitrate.set_value(f"{hit_rate:.1f} %", hr_color)

        self.card_efficiency.set_value(f"{efficiency * 100:.1f} %" if efficiency > 0 else "—")
        self.card_steps.set_value(f"{avg_steps:.0f}" if avg_steps > 0 else "—")

        self.epochen_data.append(epoche)
        self.reward_data.append(reward)
        self.hitrate_data.append(hit_rate)
        self.efficiency_data.append(efficiency)
        self.steps_data.append(avg_steps)

        if len(self.epochen_data) % 5 == 0:
            self._redraw_charts()

    def _redraw_charts(self):
        # Reward
        self.ax_reward.clear()
        self.ax_reward.set_facecolor("#0f0f22")
        self.ax_reward.tick_params(colors=C_MUTED, labelsize=9)
        for sp in self.ax_reward.spines.values():
            sp.set_color(C_BORDER); sp.set_linewidth(0.7)
        self.ax_reward.set_ylabel("Ø Reward", color=C_MUTED, fontsize=9)
        self.ax_reward.plot(self.epochen_data, self.reward_data,
                            color=C_ACCENT, linewidth=1.4, alpha=0.9)
        self.ax_reward.fill_between(self.epochen_data, self.reward_data,
                                    alpha=0.12, color=C_ACCENT)
        self.ax_reward.axhline(0, color=C_BORDER, linewidth=0.7, linestyle="--")
        self.ax_reward.grid(True, color=C_BORDER, linewidth=0.4, alpha=0.5)

        # Hit-Rate
        self.ax_hit.clear()
        self.ax_hit.set_facecolor("#0f0f22")
        self.ax_hit.tick_params(colors=C_MUTED, labelsize=9)
        for sp in self.ax_hit.spines.values():
            sp.set_color(C_BORDER); sp.set_linewidth(0.7)
        self.ax_hit.set_ylabel("Trefferquote %", color=C_MUTED, fontsize=9)
        self.ax_hit.set_xlabel("Epoche", color=C_MUTED, fontsize=9)
        self.ax_hit.set_ylim(0, 105)
        self.ax_hit.plot(self.epochen_data, self.hitrate_data,
                         color=C_SUCCESS, linewidth=1.4, alpha=0.9)
        self.ax_hit.fill_between(self.epochen_data, self.hitrate_data,
                                 alpha=0.12, color=C_SUCCESS)
        self.ax_hit.axhline(80, color=C_SUCCESS, linewidth=0.6,
                            linestyle="--", alpha=0.4, label="80 % Ziel")
        self.ax_hit.grid(True, color=C_BORDER, linewidth=0.4, alpha=0.5)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    def show_heatmap(self):
        """Zeigt entweder die trainingsbasierte Reward-Heatmap (letzten N Epochen)
        oder – falls noch keine Trainingsdaten vorhanden – die modellbasierte
        Konfidenz-Heatmap."""

        N_ANGLE_BINS = 36
        N_DIST_BINS  = 30

        has_training_data = (self.worker is not None and
                             len(self.worker.heatmap_cells) > 0)

        if has_training_data:
            # ── Trainingsbasierte Reward-Heatmap ──
            last_n = self.spin_last_n.value()
            current_ep = self.worker.current_epoch
            min_epoch  = max(0, current_ep - last_n)

            heatmap = np.full((N_DIST_BINS, N_ANGLE_BINS), np.nan)

            for (ab, db), visits in self.worker.heatmap_cells.items():
                # Besuche im Fenster
                relevant = [(e, r) for e, r in visits if e >= min_epoch]
                if not relevant:
                    continue
                # Ø der letzten 3 Besuche
                last3 = relevant[-3:]
                heatmap[db, ab] = float(np.mean([r for _, r in last3]))

            # Wertebereich für sinnvolle Farbgebung begrenzen
            finite = heatmap[np.isfinite(heatmap)]
            if finite.size > 0:
                vmin = max(np.percentile(finite, 5), -200)
                vmax = min(np.percentile(finite, 95),  500)
            else:
                vmin, vmax = -100, 100

            cmap = plt.cm.RdYlGn.copy()
            cmap.set_bad(color='#3a3a5c')   # Grau für fehlende Zellen

            masked = np.ma.masked_invalid(heatmap)

            fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
            fig_heat.patch.set_facecolor(C_BG)
            ax_heat.set_facecolor(C_SURFACE)

            c = ax_heat.imshow(masked, cmap=cmap, origin="lower", aspect="auto",
                               extent=[-180, 180, 0, 300],
                               vmin=vmin, vmax=vmax)

            ax_heat.set_xlabel("Relativer Winkel zum Ball (Grad)", color=C_MUTED)
            ax_heat.set_ylabel("Abstand zum Ball (cm)", color=C_MUTED)
            ax_heat.set_title(
                f"Reward-Heatmap  ·  letzte {last_n} Epochen  (Ø letzte 3 Besuche je Zelle)\n"
                f"Grau = keine Besuche  ·  Grün = positiv  ·  Rot = negativ",
                color=C_TEXT, fontsize=12, pad=10
            )
            ax_heat.tick_params(colors=C_MUTED)
            for sp in ax_heat.spines.values():
                sp.set_color(C_BORDER)

            cbar = fig_heat.colorbar(c, ax=ax_heat, label="Ø Step-Reward")
            cbar.ax.yaxis.label.set_color(C_MUTED)
            cbar.ax.tick_params(colors=C_MUTED)

        else:
            # ── Modellbasierte Konfidenz-Heatmap (Fallback) ──
            neuronen = int(self.combo_neuronen.currentText())
            modell_datei = f"roboter_rl_modell_{neuronen}.pth"
            if not os.path.exists(modell_datei):
                self._log("Kein trainiertes Modell für die Heatmap gefunden!")
                return

            modell = RoboterDQN(neuronen)
            modell.load_state_dict(torch.load(modell_datei))
            modell.eval()

            abstaende = np.linspace(10, 300, N_DIST_BINS)
            winkel    = np.linspace(-180, 180, N_ANGLE_BINS)
            heatmap   = np.zeros((len(abstaende), len(winkel)))
            max_dist  = math.hypot(3.0, 3.0) * 100

            with torch.no_grad():
                for i, d in enumerate(abstaende):
                    for j, w in enumerate(winkel):
                        z   = normalisiere_zustand(w, d, max_dist)
                        t_z = torch.tensor([z], dtype=torch.float32)
                        q_werte  = modell(t_z)[0]
                        q_mean   = q_werte.mean()
                        q_std    = q_werte.std() + 1e-6
                        q_norm   = (q_werte - q_mean) / q_std
                        wahrsch  = torch.nn.functional.softmax(q_norm * 2.0, dim=0)
                        heatmap[i, j] = torch.max(wahrsch).item() * 100.0

            fig_heat, ax_heat = plt.subplots(figsize=(9, 6))
            fig_heat.patch.set_facecolor(C_BG)
            ax_heat.set_facecolor(C_SURFACE)

            c = ax_heat.imshow(heatmap, cmap="RdYlGn", origin="lower", aspect="auto",
                               extent=[-180, 180, 10, 300], vmin=0, vmax=100)

            ax_heat.set_xlabel("Relativer Winkel zum Ball (Grad)", color=C_MUTED)
            ax_heat.set_ylabel("Abstand zum Ball (cm)", color=C_MUTED)
            ax_heat.set_title("Trefferwahrscheinlichkeit  ·  Grün = 100%, Rot = 0%",
                              color=C_TEXT, fontsize=13, pad=12)
            ax_heat.tick_params(colors=C_MUTED)
            for sp in ax_heat.spines.values():
                sp.set_color(C_BORDER)

            cbar = fig_heat.colorbar(c, ax=ax_heat, label="Sicherheit (%)")
            cbar.ax.yaxis.label.set_color(C_MUTED)
            cbar.ax.tick_params(colors=C_MUTED)

            ax_heat.axvline(x=-90, color="white", linestyle="--", alpha=0.4)
            ax_heat.axvline(x=90,  color="white", linestyle="--", alpha=0.4)
            ax_heat.text(0,    285, "Ball vorne",  color="white", ha="center", fontsize=9)
            ax_heat.text(-140, 285, "Ball hinten", color="white", ha="center", fontsize=9)

        fig_heat.tight_layout()
        plt.show()

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────
    def _log(self, text: str):
        self.lbl_status.setText(text)

    def _set_status(self, text: str, color: str = C_TEXT):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 12px; background: transparent;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())