import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

st.set_page_config(page_title="Roboter RL Trainer", layout="wide")

# ==========================================
# 🧠 1. REINFORCEMENT LEARNING KI (DQN)
# ==========================================
ANZAHL_AKTIONEN = 32 
WINKEL_SCHRITT = 360.0 / ANZAHL_AKTIONEN # Dynamische Berechnung der Grad-Schritte (bei 8 = 45°)

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
    # Dynamische Normalisierung basierend auf Feldgröße
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
# ⚙️ 2. SEITENLEISTE (PARAMETER & KOORDINATEN)
# ==========================================
st.sidebar.header("📏 Spielfeld & Toleranzen")
feld_breite = st.sidebar.number_input("Feld Breite (X in m)", 1.0, 5.0, 3.0, step=0.1)
feld_hoehe = st.sidebar.number_input("Feld Höhe (Y in m)", 1.0, 5.0, 3.0, step=0.1)
max_feld_distanz = math.hypot(feld_breite, feld_hoehe) * 100 # Für die Normalisierung
ziel_winkel_toleranz = st.sidebar.slider("Toleranz Treffer-Winkel (°)", 5, 90, 20)

st.sidebar.markdown("---")
st.sidebar.header("📍 Positionen auf dem Feld")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.subheader("Roboter")
    rob_x = st.number_input("X (m)", 0.0, feld_breite, 0.2, step=0.1)
    rob_y = st.number_input("Y (m)", 0.0, feld_hoehe, (feld_hoehe/2)-0.4, step=0.1)
    rob_winkel = st.slider("Blickrichtung (°)", -180, 180, 0)
with col2:
    st.subheader("Golfball")
    ball_x = st.number_input("Ball X (m)", 0.0, feld_breite, feld_breite/2, step=0.1)
    ball_y = st.number_input("Ball Y (m)", 0.0, feld_hoehe, feld_hoehe/2, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Modell Parameter")
rob_durchmesser = st.sidebar.slider("Roboter Ø (cm)", 10.0, 40.0, 22.0)
neuronen_anzahl = st.sidebar.selectbox("KI-Neuronen", [16, 32, 50, 64], index=1)
modell_datei = f"roboter_rl_modell_{neuronen_anzahl}.pth"


# ==========================================
# 🖥️ 3. HAUPTBEREICH (TABS)
# ==========================================
st.title("🤖 Reinforcement Learning - Trial & Error")
tab1, tab2 = st.tabs(["🎮 Live Test & Pfad-Vorschau", "🧠 KI Selbst-Training (RL)"])

# ---------------- TAB 1: SIMULATOR & ANIMATION ----------------
with tab1:
    rel_w, abstand_cm = berechne_zustand(rob_x, rob_y, rob_winkel, ball_x, ball_y)
    st.write(f"**Sensordaten:** Abstand = `{abstand_cm:.1f} cm` | Relativer Winkel = `{rel_w:.1f}°`")
    
    col_btn_test, col_btn_anim = st.columns(2)
    bild_platzhalter = st.empty()
    text_platzhalter = st.empty()

    if col_btn_anim.button("▶️ Test-Fahrt animieren (Live zuschauen)"):
        if not os.path.exists(modell_datei):
            st.error("Kein Modell gefunden! Bitte erst trainieren.")
        else:
            modell = RoboterDQN(neuronen_anzahl)
            modell.load_state_dict(torch.load(modell_datei))
            modell.eval()
            
            sim_rx, sim_ry = rob_x, rob_y
            rob_radius = rob_durchmesser / 2 / 100
            pfad_x, pfad_y = [sim_rx], [sim_ry]
            
            import time
            gesamt_punkte = 0
            
            with torch.no_grad():
                for schritt in range(500):
                    rw, dist = berechne_zustand(sim_rx, sim_ry, rob_winkel, ball_x, ball_y)
                    
                    zustand = normalisiere_zustand(rw, dist, max_feld_distanz)
                    tensor_z = torch.tensor([zustand], dtype=torch.float32)
                    aktion = torch.argmax(modell(tensor_z)).item() 
                    
                    # Dynamische Winkel-Berechnung!
                    ziel_rel_rad = math.radians(aktion * WINKEL_SCHRITT)
                    global_rad = math.radians(rob_winkel) + ziel_rel_rad
                    # Vorher 0.05, jetzt 0.02 (2 cm Schritte für weiche Animation)
                    sim_rx += 0.02 * math.sin(global_rad)
                    sim_ry += 0.02 * math.cos(global_rad)
                    pfad_x.append(sim_rx)
                    pfad_y.append(sim_ry)
                    
                    neu_rw, neu_dist = berechne_zustand(sim_rx, sim_ry, rob_winkel, ball_x, ball_y)
                    punkte = -1
                    status_msg = "Fährt..."
                    
                    if neu_dist <= (rob_durchmesser/2 + 2):
                        if abs(neu_rw) <= ziel_winkel_toleranz: 
                            punkte = 1000
                            status_msg = "🎯 PERFEKT ANGEKOMMEN!"
                        else: 
                            punkte = -1000
                            status_msg = "💥 CRASH! Falscher Winkel."
                    elif sim_rx < 0 or sim_rx > feld_breite or sim_ry < 0 or sim_ry > feld_hoehe:
                        punkte = -200
                        status_msg = "🧱 WAND BERÜHRT!"
                        
                    gesamt_punkte += punkte
                    
                    # Bild zeichnen (Dynamische Größe)
                    fig, ax = plt.subplots(figsize=(6, 6 * (feld_hoehe/feld_breite)))
                    fig.patch.set_facecolor('#0e1117')
                    ax.add_patch(patches.Rectangle((0, 0), feld_breite, feld_hoehe, linewidth=2, edgecolor='white', facecolor='#2e8b57'))
                    
                    ax.plot(pfad_x, pfad_y, color='orange', linestyle='--', linewidth=2, zorder=4)
                    ax.add_patch(plt.Circle((sim_rx, sim_ry), rob_radius, color='gray', zorder=5))
                    ax.plot([sim_rx, sim_rx + math.sin(math.radians(rob_winkel)) * rob_radius], 
                            [sim_ry, sim_ry + math.cos(math.radians(rob_winkel)) * rob_radius], color='red', linewidth=3, zorder=6)
                    ax.add_patch(plt.Circle((ball_x, ball_y), 0.02, color='white', zorder=5))
                    
                    ax.set_xlim(-0.1, feld_breite + 0.1); ax.set_ylim(-0.1, feld_hoehe + 0.1); ax.set_aspect('equal'); ax.axis('off')
                    plt.tight_layout()
                    
                    bild_platzhalter.pyplot(fig)
                    text_platzhalter.markdown(f"**Schritt:** {schritt+1}/100 | **Aktion (Winkel):** {aktion * WINKEL_SCHRITT}° | **Punkte:** {gesamt_punkte} | **Status:** {status_msg}")
                    
                    plt.close(fig)
                    time.sleep(0.1)
                    
                    if punkte == 1000 or punkte == -1000 or punkte == -200:
                        break 

    else:
        fig, ax = plt.subplots(figsize=(6, 6 * (feld_hoehe/feld_breite)))
        fig.patch.set_facecolor('#0e1117')
        ax.add_patch(patches.Rectangle((0, 0), feld_breite, feld_hoehe, linewidth=2, edgecolor='white', facecolor='#2e8b57'))
        rob_radius = rob_durchmesser / 2 / 100
        
        ax.add_patch(plt.Circle((rob_x, rob_y), rob_radius, color='gray', zorder=5))
        ax.plot([rob_x, rob_x + math.sin(math.radians(rob_winkel)) * rob_radius], 
                [rob_y, rob_y + math.cos(math.radians(rob_winkel)) * rob_radius], color='red', linewidth=3, zorder=6)
        ax.add_patch(plt.Circle((ball_x, ball_y), 0.02, color='white', zorder=5))
        ax.set_xlim(-0.1, feld_breite + 0.1); ax.set_ylim(-0.1, feld_hoehe + 0.1); ax.set_aspect('equal'); ax.axis('off')
        plt.tight_layout()
        bild_platzhalter.pyplot(fig)
        plt.close(fig)


# ---------------- TAB 2: TRAINING (TRIAL & ERROR) ----------------
with tab2:
    st.subheader("Trainings-Arena")
    epochen = st.number_input("Anzahl der Trainings-Epochen (Versuche)", 100, 100000, 5000, step=100)
    
    if st.button("🚀 Selbst-Training (RL) starten!"):
        modell = RoboterDQN(neuronen_anzahl)
        ziel_modell = RoboterDQN(neuronen_anzahl)
        ziel_epoche = int(epochen * 0.99)
        if os.path.exists(modell_datei):
            modell.load_state_dict(torch.load(modell_datei))
            st.info("Setze Training von bestehendem Gehirn fort... (Zufallsrate reduziert)")
            start_epsilon = 0.7  
            epsilon_min = 0.1
            ziel_epoche = int(epochen * 0.8)
        else:
            st.info("Neues Modell wird erstellt... (100% Zufall zum Erkunden)")
            start_epsilon = 1.0   
            epsilon_min = 0.6
            ziel_epoche = int(epochen * 0.99)
            
        ziel_modell.load_state_dict(modell.state_dict())
        
        optimizer = optim.Adam(modell.parameters(), lr=0.002)
        criterion = nn.MSELoss()
        memory = deque(maxlen=10000)
        
        gamma = 0.95
        epsilon = start_epsilon
        
        epsilon_decay = math.pow(epsilon_min / epsilon, 1.0 / ziel_epoche) if ziel_epoche > 0 else 0.995
        
        # UI Elemente
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart = st.line_chart([]) # Leerer Graph
        
        rob_radius_cm = rob_durchmesser / 2
        
        # UI & Save Intervalle (Jetzt ca. 500 Datenpunkte für einen schönen, detaillierten Graphen)
        update_intervall = max(5, epochen // 500) 
        save_intervall = max(500, epochen // 10)   
        
        # NEU: Liste zum Sammeln der Punkte für den Durchschnitt
        belohnungen_fenster = []
        
        try:
            for epoche in range(epochen):
                # Ball irgendwo ins Feld
                b_x = random.uniform(0.5, feld_breite - 0.5)
                b_y = random.uniform(0.5, feld_hoehe - 0.5)
                
                # Roboter zufällig platzieren
                r_x = random.uniform(0.2, feld_breite - 0.2)
                r_y = random.uniform(0.2, feld_hoehe - 0.2)
                r_w = random.uniform(-180, 180)
                
                gesamt_belohnung = 0
                
                for schritt in range(500):
                    rel_w, dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                    zustand = normalisiere_zustand(rel_w, dist, max_feld_distanz)
                    
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
                    neuer_zustand = normalisiere_zustand(neu_rel_w, neu_dist, max_feld_distanz)
                    
                    belohnung = -1 
                    done = False
                    
                    if neu_dist <= (rob_radius_cm + 2):
                        if abs(neu_rel_w) <= ziel_winkel_toleranz:
                            belohnung = 10000 
                        else:
                            belohnung = -1000 
                        done = True
                    elif r_x < 0 or r_x > feld_breite or r_y < 0 or r_y > feld_hoehe:
                        belohnung = -200 
                        done = True
                    else:
                        belohnung += (dist - neu_dist) * 10
                        
                    gesamt_belohnung += belohnung
                    memory.append((zustand, aktion, belohnung, neuer_zustand, done))
                    
                    if done: break
                    
                if len(memory) > 64:
                    batch = random.sample(memory, 64)
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

                if epsilon > epsilon_min: epsilon *= epsilon_decay
                if epoche % 10 == 0: ziel_modell.load_state_dict(modell.state_dict())
                
                # JEDE Belohnung sammeln
                belohnungen_fenster.append(gesamt_belohnung)
                    
                # --- UI UPDATES (Durchschnitt plotten!) ---
                if epoche > 0 and epoche % update_intervall == 0:
                    # Durchschnitt der letzten X Epochen berechnen
                    durchschnitt = sum(belohnungen_fenster) / len(belohnungen_fenster)
                    
                    chart.add_rows([durchschnitt]) 
                    progress_bar.progress((epoche + 1) / epochen)
                    status_text.text(f"Epoche {epoche}/{epochen} | Epsilon: {epsilon*100:.1f}% | Ø Punkte: {durchschnitt:.1f}")
                    
                    # Fenster wieder leeren für das nächste Intervall
                    belohnungen_fenster.clear()

                # --- AUTO-SAVE ---
                if epoche > 0 and epoche % save_intervall == 0:
                    torch.save(modell.state_dict(), modell_datei)
                    
        except Exception as e:
            st.error(f"⚠️ Training wurde unterbrochen: {e}")
            
        finally:
            torch.save(modell.state_dict(), modell_datei)
            st.success(f"✅ Status gesichert! Modell in `{modell_datei}` gespeichert.")