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
| Action-Jump-Penalty (optional) | −Stärke × Kreisabstand |

### Training-Parameter (langzeitstabil)

| Parameter | Wert | Beschreibung |
|---|---|---|
| Optimizer | Adam, lr = 0.0005 | Reduzierte Lernrate für stabiles Langzeit-Training |
| Loss | `SmoothL1Loss` (Huber) | Robuster gegenüber großen Reward-Ausreißern |
| Target-Computation | **Double DQN** | Hauptmodell wählt Aktion, Zielmodell bewertet sie |
| Gradient Clipping | max_norm = 10.0 | Verhindert explodierende Gradienten bei großen Rewards |
| Replay-Buffer | 100 000 Transitionen | Größerer Puffer für stabileres Lernen |
| Target-Update | alle 20 Epochen | Unverändert |
| Epsilon-Greedy | ε = 1.0 → 0.05 (über 30% der Epochen) | Unverändert |
| Auto-Save | alle 1 000 Epochen | Unverändert |

---

## 📊 4. GUI-Monitoring (`Trainer_V2.py`)

PyQt5 Dark-Theme Dashboard mit Echtzeit-Anzeige:

### Metriken-Karten
| Karte | Beschreibung |
|---|---|
| ⏱ Epoche | Aktueller Trainingsfortschritt |
| 🎯 Trefferquote | Hit-Rate der letzten 100 Episoden (%) |
| 🎲 Zufall ε | Aktuelle Explorationsrate |
| 📉 Loss | Aktueller Huber-Loss |
| 📐 Ø Effizienz | Ø Pfadeffizienz der letzten 200 Treffer-Episoden (%) |
| 👣 Ø Schritte/Hit | Durchschnittliche Schritte bis zum Treffer |

### Sidebar-Einstellungen
- **Epochen** – Trainings-Dauer
- **KI-Neuronen** – Netzwerk-Größe
- **Action-Jump-Penalty** – an/aus und Stärke (×0.1)
- **Heatmap: letzte N Epochen** – Rolling-Window für die Analyse-Heatmap

---

## 🗺️ 5. Reward-Heatmap (Trainingsbasiert)

Die Heatmap visualisiert die während des Trainings gesammelten Erfahrungen.

### Darstellung
- **X-Achse:** Relativer Winkel zum Ball (−180° … +180°)
- **Y-Achse:** Abstand zum Ball (0 … 300 cm)
- **Farbe:** Durchschnittlicher Step-Reward der **letzten 3 Besuche** in der Zelle, gefiltert auf das **Rolling-Window** der letzten N Epochen
  - Grün = positiver Reward (Annäherung / Treffer)
  - Rot = negativer Reward (Wand / Crash)
  - Grau = Zelle wurde im Fenster nicht besucht

### Rolling-Window
Der Schieberegler „Heatmap: letzte N Epochen" in der Sidebar bestimmt, wie weit in der Vergangenheit die Besuche berücksichtigt werden. Standard: 500 Epochen.

### Fallback
Wurde noch kein Training gestartet, zeigt die Heatmap die modellbasierte Konfidenz-Analyse (wie bisher).

---

## 📐 6. Pfad-Effizienz-Metriken

Während des Trainings werden folgende Kennzahlen berechnet (nur für Treffer-Episoden):

```
step_size_m    = 0.02  # 2 cm pro Schritt
path_length_m  = steps_used × step_size_m
ideal_dist_m   = hypot(ball_pos - start_pos)
efficiency     = min(ideal_dist_m / path_length_m, 1.0)
```

- `efficiency = 1.0` → Roboter fuhr auf direktem Weg
- `efficiency = 0.5` → Roboter brauchte doppelt so viele Schritte wie nötig
- Rolling Average der letzten 200 Treffer-Episoden

---

## 🔀 7. Action-Jump-Penalty (Reward Shaping)

Eine optionale Strafe für große Richtungssprünge zwischen aufeinanderfolgenden Schritten.

```
circ_diff = min(|curr - prev|, N_ACTIONS - |curr - prev|)  # Kreisabstand
penalty   = -stärke × circ_diff
```

**Konfiguration in der GUI:**
- Checkbox „✓ Action-Jump-Penalty" aktiviert/deaktiviert die Penalty
- „Penalty-Stärke ×0.1": Wert 5 → Stärke 0.5 (Standard)

---

## 🗂️ 8. LUT-Format & Indexierung

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

## 🚀 9. Quickstart

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

### Tests ausführen

```bash
pip install pytest
python -m pytest tests/ -v
```

Die Tests prüfen:
- **Double DQN** Ziel-Berechnung (Form, Auswahllogik)
- **Heatmap-Binning** und Rolling-Window-Filter
- **Action-Jump-Penalty** (Kreisabstand-Berechnung)
- **Pfad-Effizienz-Metrik**
- **Smoke-Test**: Trainingsschleife läuft deterministisch (Seed) ohne Fehler

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

## 📁 10. Dateiübersicht

| Datei / Ordner | Beschreibung |
|---|---|
| `Trainer_V2.py` | DQN-Training mit PyQt5-GUI (aktueller Trainer) |
| `Trainer.py` | Ältere Streamlit-Version des Trainers |
| `Tester.py` | Simulations-Tester für trainierte Modelle |
| `Lut_Generator.py` | Einfacher LUT-Generator (Root-Verzeichnis) |
| `tools/generate_lut_actions.py` | Erweiterter LUT-Generator mit CLI-Argumenten |
| `tests/test_training.py` | Automatisierte pytest-Tests für Trainer-Logik |
| `Teensy.cpp` | C++ Beispielcode für Teensy (LUT-Lookup) |
| `Models/` | Vortrainierte Modelle (`.pth`) |
| `roboter_rl_modell_32.pth` | Trainiertes Modell mit 32 Neuronen |

---

## 🔧 11. Deployment-Flow (Zusammenfassung)

```
1.  PC: python Trainer_V2.py
        → roboter_rl_modell_64.pth

2.  PC: python tools/generate_lut_actions.py --model roboter_rl_modell_64.pth ...
        → roboter_lut_actions.h  (C-Array, ~70 KB)

3.  PlatformIO: roboter_lut_actions.h → src/
                pio run --target upload

4.  Teensy: lut_get_action(dist, angle) → action → Motorsteuerung
```

---

## ⚠️ 12. Kompatibilitätshinweise

- **Modell-Inkompatibilität durch Änderung der Action-Anzahl:** Ältere `.pth`-Dateien mit 32 Aktionen können nicht mit dem aktuellen Modell (90 Aktionen) geladen werden. `roboter_rl_modell_32.pth` im Root-Verzeichnis ist ein altes Modell – für das aktuelle Training neue Modelle mit 64 oder mehr Neuronen erstellen.
- **Huber-Loss vs. MSE:** Der Wechsel von `MSELoss` zu `SmoothL1Loss` verändert die Loss-Skala; historische Loss-Werte sind nicht direkt vergleichbar.
- **Double DQN:** Beeinflusst die Q-Ziel-Werte; Fortsetzung eines mit Standard-DQN trainierten Modells ist problemlos möglich (keine Architektur-Änderung).

