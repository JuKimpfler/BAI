import sys
import math
import random
import os
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                             QProgressBar, QGroupBox, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ==========================================
# 1. KI MODELL & LOGIK
# ==========================================
ANZAHL_AKTIONEN = 32
WINKEL_SCHRITT = 360.0 / ANZAHL_AKTIONEN

class RoboterDQN(nn.Module):
    def __init__(self, neuronen=64):
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
    update_signal = pyqtSignal(int, float, float, float, float) # Epoche, Epsilon, Reward, Hit-Rate, Loss
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, epochen, neuronen, modell_datei):
        super().__init__()
        self.epochen = epochen
        self.neuronen = neuronen
        self.modell_datei = modell_datei
        self.running = True

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
            epsilon = 0.5  # Startet mit etwas weniger Zufall
        else:
            self.log_signal.emit("Starte komplett neues Training...")
            epsilon = 1.0

        ziel_modell.load_state_dict(modell.state_dict())
        optimizer = optim.Adam(modell.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        memory = deque(maxlen=20000)
        
        gamma = 0.95
        epsilon_min = 0.05
        ziel_epoche = int(self.epochen * 0.8)
        epsilon_decay = math.pow(epsilon_min / epsilon, 1.0 / ziel_epoche) if ziel_epoche > 0 else 0.995

        hit_history = deque(maxlen=100) # Speichert die letzten 100 Ergebnisse (1 = Hit, 0 = Fail)
        belohnungen_fenster = []
        loss_val = 0.0

        for epoche in range(self.epochen):
            if not self.running:
                break

            b_x = random.uniform(0.5, feld_breite - 0.5)
            b_y = random.uniform(0.5, feld_hoehe - 0.5)
            
            r_x = random.uniform(0.2, feld_breite - 0.2)
            r_y = random.uniform(0.2, feld_hoehe - 0.2)

            # --- SMART SPAWNING (70% Chance auf schwere Position) ---
            if random.random() < 0.70:
                abs_winkel = math.degrees(math.atan2(b_x - r_x, b_y - r_y))
                versatz = random.uniform(90, 270) # Ball ist im Rücken
                r_w = (abs_winkel + versatz) % 360
                if r_w > 180: r_w -= 360
            else:
                r_w = random.uniform(-180, 180)

            gesamt_belohnung = 0
            
            for schritt in range(300):
                rel_w, dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                zustand = normalisiere_zustand(rel_w, dist, max_dist)
                
                if random.random() < epsilon:
                    aktion = random.randint(0, ANZAHL_AKTIONEN - 1)
                else:
                    with torch.no_grad():
                        aktion = torch.argmax(modell(torch.tensor([zustand], dtype=torch.float32))).item()
                        
                ziel_rel_rad = math.radians(aktion * WINKEL_SCHRITT)
                global_rad = math.radians(r_w) + ziel_rel_rad
                r_x += 0.02 * math.sin(global_rad)
                r_y += 0.02 * math.cos(global_rad)
                
                neu_rel_w, neu_dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                neuer_zustand = normalisiere_zustand(neu_rel_w, neu_dist, max_dist)
                
                belohnung = -1 
                done = False
                
                if neu_dist <= (rob_radius_cm + 2):
                    if abs(neu_rel_w) <= toleranz:
                        belohnung = 10000
                        hit_history.append(1) # ERFOLG!
                    else:
                        belohnung = -1000
                        hit_history.append(0) # CRASH
                    done = True
                elif r_x < 0 or r_x > feld_breite or r_y < 0 or r_y > feld_hoehe:
                    belohnung = -500
                    hit_history.append(0) # WAND
                    done = True
                else:
                    belohnung += (dist - neu_dist) * 10
                    
                gesamt_belohnung += belohnung
                memory.append((zustand, aktion, belohnung, neuer_zustand, done))
                
                if done: break
                
            # Training aus dem Gedächtnis
            if len(memory) > 128:
                batch = random.sample(memory, 128)
                z_batch = torch.tensor([x[0] for x in batch], dtype=torch.float32)
                a_batch = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
                r_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                nz_batch = torch.tensor([x[3] for x in batch], dtype=torch.float32)
                d_batch = torch.tensor([x[4] for x in batch], dtype=torch.float32)
                
                q_werte = modell(z_batch).gather(1, a_batch).squeeze()
                naechste_q_werte = ziel_modell(nz_batch).max(1)[0]
                erwartete_q_werte = r_batch + gamma * naechste_q_werte * (1 - d_batch)
                
                loss = criterion(q_werte, erwartete_q_werte.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()

            if epsilon > epsilon_min: epsilon *= epsilon_decay
            if epoche % 20 == 0: ziel_modell.load_state_dict(modell.state_dict())
            
            belohnungen_fenster.append(gesamt_belohnung)
                
            # GUI Update Interval (z.B. alle 10 Epochen)
            if epoche % 10 == 0 and len(belohnungen_fenster) > 0:
                durchschnitt_reward = sum(belohnungen_fenster) / len(belohnungen_fenster)
                hit_rate = (sum(hit_history) / len(hit_history)) * 100 if len(hit_history) > 0 else 0
                
                self.update_signal.emit(epoche, epsilon, durchschnitt_reward, hit_rate, loss_val)
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
# 3. GUI (PyQt5)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboter RL Trainings-Station (Pro)")
        self.setGeometry(100, 100, 1000, 600)
        
        self.worker = None
        self.reward_data = []
        self.epochen_data = []

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LINKE SPALTE (Controls) ---
        control_layout = QVBoxLayout()
        control_group = QGroupBox("Training Setup")
        control_group.setLayout(control_layout)
        control_group.setFixedWidth(250)

        control_layout.addWidget(QLabel("Anzahl Epochen:"))
        self.spin_epochen = QSpinBox()
        self.spin_epochen.setRange(100, 500000)
        self.spin_epochen.setValue(20000)
        self.spin_epochen.setSingleStep(1000)
        control_layout.addWidget(self.spin_epochen)

        control_layout.addWidget(QLabel("KI-Neuronen:"))
        self.combo_neuronen = QComboBox()
        self.combo_neuronen.addItems(["32", "50", "64", "128"])
        self.combo_neuronen.setCurrentText("64")
        control_layout.addWidget(self.combo_neuronen)

        self.btn_start = QPushButton("🚀 Training Starten")
        self.btn_start.setStyleSheet("background-color: #2e8b57; color: white; padding: 10px; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_training)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("🛑 Training Stoppen")
        self.btn_stop.setStyleSheet("background-color: #b22222; color: white; padding: 10px;")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        control_layout.addSpacing(30)
        self.btn_heatmap = QPushButton("🗺️ Analyse Heatmap anzeigen")
        self.btn_heatmap.clicked.connect(self.show_heatmap)
        control_layout.addWidget(self.btn_heatmap)
        
        control_layout.addStretch()
        layout.addWidget(control_group)

        # --- RECHTE SPALTE (Analyse & Plots) ---
        right_layout = QVBoxLayout()
        
        # Stats Info
        stats_layout = QHBoxLayout()
        self.lbl_epoche = QLabel("Epoche: 0 / 0")
        self.lbl_hitrate = QLabel("Trefferquote: 0%")
        self.lbl_hitrate.setFont(QFont("Arial", 12, QFont.Bold))
        self.lbl_hitrate.setStyleSheet("color: orange;")
        self.lbl_epsilon = QLabel("Zufall (Epsilon): 100%")
        
        stats_layout.addWidget(self.lbl_epoche)
        stats_layout.addWidget(self.lbl_hitrate)
        stats_layout.addWidget(self.lbl_epsilon)
        right_layout.addLayout(stats_layout)

        self.progress = QProgressBar()
        right_layout.addWidget(self.progress)

        self.lbl_log = QLabel("Bereit.")
        right_layout.addWidget(self.lbl_log)

        # Matplotlib Plot für Rewards
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)

        layout.addLayout(right_layout)

    def start_training(self):
        neuronen = int(self.combo_neuronen.currentText())
        epochen = self.spin_epochen.value()
        modell_datei = f"roboter_rl_modell_{neuronen}.pth"

        self.reward_data.clear()
        self.epochen_data.clear()
        self.ax.clear()
        self.canvas.draw()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setMaximum(epochen)
        self.progress.setValue(0)

        self.worker = TrainingWorker(epochen, neuronen, modell_datei)
        self.worker.update_signal.connect(self.update_gui)
        self.worker.log_signal.connect(self.lbl_log.setText)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.lbl_log.setText("Beende Training sicher... Bitte warten!")

    def training_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_log.setText(self.lbl_log.text() + " (Fertig)")

    def update_gui(self, epoche, epsilon, reward, hit_rate, loss):
        self.progress.setValue(epoche)
        self.lbl_epoche.setText(f"Epoche: {epoche} / {self.spin_epochen.value()}")
        self.lbl_epsilon.setText(f"Zufall (Epsilon): {epsilon*100:.1f}%")
        
        # Farbe der Hitrate anpassen
        color = "red"
        if hit_rate > 50: color = "orange"
        if hit_rate > 80: color = "green"
        self.lbl_hitrate.setText(f"Trefferquote: {hit_rate:.1f}%")
        self.lbl_hitrate.setStyleSheet(f"color: {color};")

        # Plot aktualisieren (Performance-optimiert: nur alle 50 Epochen zeichnen)
        self.epochen_data.append(epoche)
        self.reward_data.append(reward)
        
        if len(self.epochen_data) % 5 == 0:
            self.ax.clear()
            self.ax.set_title("Durchschnittlicher Reward")
            self.ax.plot(self.epochen_data, self.reward_data, color='blue')
            self.canvas.draw()

    def show_heatmap(self):
        neuronen = int(self.combo_neuronen.currentText())
        modell_datei = f"roboter_rl_modell_{neuronen}.pth"
        if not os.path.exists(modell_datei):
            self.lbl_log.setText("Kein trainiertes Modell für die Heatmap gefunden!")
            return

        # Modell laden
        modell = RoboterDQN(neuronen)
        modell.load_state_dict(torch.load(modell_datei))
        modell.eval()

        # Grid erstellen
        abstaende = np.linspace(10, 300, 30) # 10cm bis 3m
        winkel = np.linspace(-180, 180, 36)
        heatmap = np.zeros((len(abstaende), len(winkel)))

        max_dist = math.hypot(3.0, 3.0) * 100

        with torch.no_grad():
            for i, d in enumerate(abstaende):
                for j, w in enumerate(winkel):
                    z = normalisiere_zustand(w, d, max_dist)
                    t_z = torch.tensor([z], dtype=torch.float32)
                    q_werte = modell(t_z)
                    max_q = torch.max(q_werte).item()
                    heatmap[i, j] = max_q

        # Heatmap plotten (als separates Fenster)
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
        c = ax_heat.imshow(heatmap, cmap='RdYlGn', origin='lower', aspect='auto',
                           extent=[-180, 180, 10, 300])
        ax_heat.set_xlabel("Relativer Winkel zum Ball (Grad)")
        ax_heat.set_ylabel("Abstand zum Ball (cm)")
        ax_heat.set_title("KI Zuversicht (Grün = Weiß was zu tun ist, Rot = Unsicher)")
        fig_heat.colorbar(c, label="Max Q-Value (Erwartete Punkte)")
        
        # Hilfslinie für "Ball ist hinten"
        ax_heat.axvline(x=-90, color='white', linestyle='--', alpha=0.5)
        ax_heat.axvline(x=90, color='white', linestyle='--', alpha=0.5)
        ax_heat.text(0, 280, "Ball Vorne", color="white", ha='center')
        ax_heat.text(-140, 280, "Ball Hinten", color="white", ha='center')
        
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())