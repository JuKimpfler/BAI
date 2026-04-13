# BAI – Ball Approach Intelligence (Robocup Soccer)

KI-gesteuerte Ballanfahrt für einen omnidirektionalen Roboter auf Basis von **Deep Reinforcement Learning (DQN)**. Das Modell wird offline auf dem PC trainiert und als kompakte **Lookup-Table (LUT)** in den Flash-Speicher des Teensy deployt – kein Raspberry Pi, keine Laufzeit-Inferenz auf dem Roboter.

---

## 🎯 1. Projektziel

Der Roboter soll aus seiner aktuellen Position optimal an den Ball heranfahren und ihn mit dem richtigen Winkel (Frontkontakt, ±20°) treffen. Die Entscheidung – welche Fahrtrichtung zu wählen ist – wird durch eine vorab berechnete Lookup-Table getroffen, die aus einem trainierten DQN-Modell generiert wurde.

**Zustandsbeschreibung (2 Inputs):**

| Input | Wertebereich | Normalisierung |
|---|---|---|
| Relativer Winkel zum Ball | −180° … +180° | `/180.0` → [−1, 1] |
| Abstand zum Ball | 0 … ~424 cm | `/max_dist_cm` → [0, 1] |

**Ausgabe:** Action-ID `0..89` (eine von 90 diskreten Fahrtrichtungen, je 4° Schritt).

---

## ⚙️ 2. System-Architektur

```
┌─────────────────────────────────┐      ┌──────────────────────────────────┐
│        PC (Offline)             │      │        Teensy 4.x (Echtzeit)     │
│                                 │      │                                  │
│  Trainer_V2.py                  │      │  Flash: roboter_lut_actions.h    │
│  ├─ DQN trainieren (PyTorch)    │─────▶│  ├─ lut_get_action(dist, angle)  │
│  └─ .pth Modell speichern       │      │  ├─ action_to_rel_angle_deg()    │
│                                 │      │  └─ Motor-PID / Omnidrive-Kin.   │
│  tools/generate_lut_actions.py  │      │                                  │
│  └─ LUT generieren (.h + .bin)  │      │  500–1000 Hz Echtzeit-Regelung   │
└─────────────────────────────────┘      └──────────────────────────────────┘
```

- **Kein Raspberry Pi** auf dem Roboter – der Teensy ist für NN-Inferenz zu ausgelastet.
- **Lookup statt Inferenz:** Für jede diskretisierte Zustandskombination (Winkel + Abstand) wird die beste Action-ID einmalig offline berechnet und als `uint8_t`-Array im Flash abgelegt.
- **Kommunikation Teensy:** Sensoren (Kamera/IR) liefern relativen Winkel und Abstand, der Teensy schlägt direkt in der LUT nach und steuert die Motoren.

---

## 🧠 3. DQN-Modell (`RoboterDQN`)

```
Input (2)  →  Linear(2 → N)  →  ReLU  →  Linear(N → N)  →  ReLU  →  Linear(N → 90)
```

- `N` = Anzahl Neuronen (konfigurierbar, z.B. 32, 50, 64, 128)
- **90 Aktionen**, jede entspricht einer Fahrtrichtung (0° … 356° in 4°-Schritten)
- Modell-Datei: `roboter_rl_modell_<N>.pth` (PyTorch state_dict)

### Reward-Struktur

| Ereignis | Reward |
|---|---|
| Ball getroffen (Winkel ≤ ±20°) | +10 000 |
| Crash (Ball getroffen, falscher Winkel) | −1 000 |
| Wand getroffen | −500 |
| Annäherung an Ball | +(Δdistanz × 10) |
| Jeder Schritt (Zeitstrafe) | −1 |

### Training-Parameter

- Experience Replay, Batch-Größe 128
- Target-Network-Update alle 20 Epochen
- Epsilon-Greedy: ε = 1.0 → 0.05 (über 30 % der Epochen)
- Optimizer: Adam, lr = 0.001, Loss: MSELoss
- Auto-Save alle 1 000 Epochen, Resume möglich

---

## 📊 4. GUI-Monitoring (`Trainer_V2.py`)

PyQt5 Dark-Theme Dashboard mit Echtzeit-Anzeige:
- **Hit-Rate** (Ziel: > 80 %)
- **Ø Reward** pro Episode
- **Epsilon** (Explorationsrate)
- **Loss** (MSE)
- **Heatmap-Analyse** (Konfidenz über Winkel-/Distanz-Raum)

---

## 🗂️ 5. LUT-Format & Indexierung

### Diskretisierung

| Dimension | Wertebereich | Schrittweite | Anzahl Werte |
|---|---|---|---|
| Abstand | 0 … 200 cm | 1 cm | 201 |
| Winkel | 0 … 359° | 1° | 360 |
| **Gesamt** | | | **72 360 Einträge** |

### Index-Formel

```
idx = dist_idx * LUT_ANGLE_COUNT + angle_idx

dist_idx  = clamp(round(distance_cm / LUT_DIST_STEP_CM), 0, LUT_DIST_COUNT - 1)
angle_idx = round((rel_angle_deg % 360) / LUT_ANGLE_STEP_DEG) % LUT_ANGLE_COUNT
```

### Datentyp

```c
const uint8_t LUT_ACTIONS[LUT_DIST_COUNT * LUT_ANGLE_COUNT];
// 72 360 × 1 Byte ≈ 70 KB  →  passt in Teensy 4.1 Flash (> 1 MB)
```

### Action → Fahrwinkel

```c
// 90 Aktionen, Schrittweite 4°
float angle_deg = action * (360.0f / 90.0f);  // 0..356°
if (angle_deg > 180.0f) angle_deg -= 360.0f;  // → [-180..+180]
// 0° = vorwärts, +90°/-90° = seitlich, ±180° = rückwärts
```

---

## 🚀 6. Quickstart

### Voraussetzungen

```bash
pip install torch numpy pyqt5 matplotlib
```

### Training starten (`Trainer_V2.py`)

```bash
python Trainer_V2.py
```

- Neuronen-Anzahl in der GUI auswählen (z.B. 64)
- „Training starten" klicken
- Modell wird automatisch als `roboter_rl_modell_64.pth` gespeichert
- Training kann jederzeit unterbrochen und fortgesetzt werden (Resume)

### LUT generieren (`tools/generate_lut_actions.py`)

```bash
python tools/generate_lut_actions.py \
    --model roboter_rl_modell_64.pth \
    --out-prefix roboter_lut \
    --max-distance-cm 200 \
    --distance-step 1 \
    --angle-step 1
```

Ausgabe:
- `roboter_lut_actions.bin` – Rohdaten (72 360 Bytes)
- `roboter_lut_actions.h` – C-Header-Datei für PlatformIO/Teensy

Vortrainierte Modelle liegen unter `Models/` (z.B. `Models/roboter_rl_modell_32.pth`).

### Teensy / PlatformIO (`Teensy.cpp`)

1. `roboter_lut_actions.h` in `src/` des PlatformIO-Projekts kopieren.
2. In `main.cpp` einbinden:

```cpp
#include "roboter_lut_actions.h"

// Relativen Fahrwinkel für aktuelle Sensorwerte ermitteln:
uint8_t action      = lut_get_action(distance_cm, rel_ball_angle_deg);
float   drive_angle = action_to_rel_angle_deg(action);  // [-180..+180]
// → drive_angle in Omnidrive-Kinematik einspeisen
```

Ein vollständiges Beispiel mit `lut_get_action()` und `action_to_rel_angle_deg()` findet sich in `Teensy.cpp`.

---

## 📁 7. Dateiübersicht

| Datei / Ordner | Beschreibung |
|---|---|
| `Trainer_V2.py` | DQN-Training mit PyQt5-GUI (aktueller Trainer) |
| `Trainer.py` | Ältere Streamlit-Version des Trainers |
| `Tester.py` | Simulations-Tester für trainierte Modelle |
| `Lut_Generator.py` | Einfacher LUT-Generator (Root-Verzeichnis) |
| `tools/generate_lut_actions.py` | Erweiterter LUT-Generator mit CLI-Argumenten |
| `Teensy.cpp` | C++ Beispielcode für Teensy (LUT-Lookup) |
| `Models/` | Vortrainierte Modelle (`.pth`) |
| `roboter_rl_modell_32.pth` | Trainiertes Modell mit 32 Neuronen |

---

## 🔧 8. Deployment-Flow (Zusammenfassung)

```
1.  PC: python Trainer_V2.py
        → roboter_rl_modell_64.pth

2.  PC: python tools/generate_lut_actions.py --model roboter_rl_modell_64.pth ...
        → roboter_lut_actions.h  (C-Array, ~70 KB)

3.  PlatformIO: roboter_lut_actions.h → src/
                pio run --target upload

4.  Teensy: lut_get_action(dist, angle) → action → Motorsteuerung
```