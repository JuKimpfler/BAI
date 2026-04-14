# Teensy 4.0 – Anleitung: KI-Lookup-Tabelle erzeugen und verwenden

Diese Anleitung erklärt Schritt für Schritt, wie du das trainierte neuronale
Netz (`roboter_rl_modell_*.pth`) in eine statische Lookup-Tabelle (LUT) für
den Teensy 4.0 umwandelst und damit einen Roboter steuerst.

---

## Überblick

Das neuronale Netz wird **einmalig** auf dem PC ausgewertet und die Ergebnisse
als kompakte C-Tabelle gespeichert.  Auf dem Teensy ist dann **kein Netzwerk
mehr nötig** – der Lookup dauert konstant O(1) und belegt nur ≈ 72 KB Flash.

| Dimension | Bereich | Schritte | Werte |
|-----------|---------|----------|-------|
| Winkel    | 0 – 359 ° | 1 ° | 360 |
| Abstand   | 0 – 200 cm | 1 cm | 201 |
| **Gesamt** | | | **72.360 Einträge** |

Jeder Eintrag ist ein `uint8_t` (0 – 89), der den besten Aktionsindex
des Netzes für diese Kombination aus Winkel und Abstand darstellt.  
Ein Aktionswert `A` entspricht einer Fahrtrichtung von `A × 4 °`.

---

## Voraussetzungen

### PC (LUT-Generierung)

- Python ≥ 3.8
- PyTorch (`pip install torch`)
- Eine trainierte Modelldatei, z. B. `roboter_rl_modell_128.pth`

### Teensy 4.0

- [PlatformIO](https://platformio.org/) (VS Code Extension oder CLI)
- Teensy 4.0 Board
- USB-Kabel

---

## Schritt 1 – LUT auf dem PC erzeugen

Öffne ein Terminal im Stammverzeichnis des Projekts und führe aus:

```bash
# Standard (128 Neuronen, Modell im gleichen Verzeichnis):
python generate_lut.py

# Mit expliziten Parametern:
python generate_lut.py --modell roboter_rl_modell_128.pth --neuronen 128

# Weiteres Beispiel (256 Neuronen, andere Max-Distanz):
python generate_lut.py --neuronen 256 --max-dist 424.26 --out robot_lut.h
```

Das Skript gibt Fortschrittsmeldungen aus und erzeugt die Datei `robot_lut.h`.

### Parameter

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `--modell` | `roboter_rl_modell_<neuronen>.pth` | Pfad zur Modelldatei |
| `--neuronen` | `128` | Neuronen pro Schicht (muss mit Training übereinstimmen) |
| `--max-dist` | `≈ 424.26` | Normalisierungs-Distanz in cm (Diagonale 3 × 3 m) |
| `--out` | `robot_lut.h` | Name der Ausgabedatei |

---

## Schritt 2 – Header-Datei ins Teensy-Projekt kopieren

```bash
cp robot_lut.h teensy/include/robot_lut.h
```

Die Ordnerstruktur sieht dann so aus:

```
teensy/
├── platformio.ini
├── include/
│   └── robot_lut.h      ← hier hin kopieren
└── src/
    └── main.cpp
```

---

## Schritt 3 – Teensy-Sketch anpassen

Öffne `teensy/src/main.cpp` und passe die Pin-Definitionen an deine Verdrahtung an:

```cpp
// Ultraschall-Sensor HC-SR04
static constexpr uint8_t PIN_TRIG = 6;
static constexpr uint8_t PIN_ECHO = 7;

// Winkel-Sensor (Kompass / IMU)
static constexpr uint8_t PIN_WINKEL_ANALOG = A0;  // Demo: Analogpoti

// Motortreiber L298N
static constexpr uint8_t PIN_ENA = 2;
static constexpr uint8_t PIN_IN1 = 3;
// ...
```

> **Hinweis:** Die Funktion `leseWinkelDeg()` ist aktuell ein Platzhalter, der
> ein Analogpoti liest.  Ersetze sie durch deine IMU-Bibliothek, z. B.:
>
> ```cpp
> // Beispiel mit Adafruit BNO055:
> #include <Adafruit_BNO055.h>
> Adafruit_BNO055 bno;
>
> int leseWinkelDeg() {
>   sensors_event_t event;
>   bno.getEvent(&event);
>   return (int)event.orientation.x;  // 0–359°
> }
> ```

---

## Schritt 4 – Kompilieren und auf Teensy flashen

```bash
# Ins Teensy-Verzeichnis wechseln
cd teensy

# Kompilieren und flashen
pio run --target upload

# Seriellen Monitor öffnen (Baudrate 115200)
pio device monitor
```

Die serielle Ausgabe sieht z. B. so aus:

```
Roboter LUT-Steuerung startet...
LUT-Eintraege: 72360
Aktionen     : 0–89
Winkel/Aktion: 4.00°
Bereit.
Winkel=45°  Abst=123cm  Akt=11  →44.0°
Winkel=45°  Abst=120cm  Akt=11  →44.0°
...
```

---

## Schritt 5 – LUT im eigenen Code nutzen

Die Header-Datei stellt zwei Hilfsfunktionen bereit:

```cpp
#include "robot_lut.h"

// Beste Aktion (0–89) für Winkel + Abstand
uint8_t aktion = robot_lut_lookup(winkel_deg, abstand_cm);

// Umrechnung: Aktionsindex → Fahrtrichtung in Grad (0–356°)
float richtung = robot_aktion_zu_winkel(aktion);
```

### Vollständiges Minimal-Beispiel

```cpp
#include <Arduino.h>
#include "robot_lut.h"

void setup() {
  Serial.begin(115200);
}

void loop() {
  int winkel  = 90;   // 90° = Ziel liegt direkt links
  int abstand = 50;   // 50 cm entfernt

  uint8_t aktion   = robot_lut_lookup(winkel, abstand);
  float   richtung = robot_aktion_zu_winkel(aktion);

  Serial.print("Aktion: ");   Serial.print(aktion);
  Serial.print("  Richtung: "); Serial.print(richtung);
  Serial.println("°");

  delay(500);
}
```

---

## Speicherbedarf

| Was | Größe |
|-----|-------|
| LUT (`robot_lut`) | 72.360 Byte ≈ **71 KB** |
| Teensy 4.0 Flash gesamt | **2.048 KB** |
| Verbleibender Flash | **> 1.970 KB** |

Die LUT liegt vollständig im Flash (`PROGMEM`) und belastet den RAM **nicht**.

---

## Häufige Probleme

| Problem | Lösung |
|---------|--------|
| `robot_lut.h` nicht gefunden | Datei in `teensy/include/` kopieren |
| Kompilierung schlägt fehl: *board not found* | `pio platform install teensy` |
| Modell lässt sich nicht laden | `--neuronen` prüfen – muss mit dem Training übereinstimmen |
| Alle Aktionen sind 0 | Modell noch nicht trainiert oder Normalisierung prüfen |
| Teensy wird nicht erkannt | Teensy Loader installieren, USB-Kabel prüfen |

---

## Technische Details

### Normalisierung der Eingaben

Das Netz erwartet drei normalisierte Eingaben:

```
sin(winkel_rad)        – x-Komponente der Richtung
cos(winkel_rad)        – y-Komponente der Richtung
abstand_cm / max_dist  – normierter Abstand (0.0–1.0+)
```

`max_dist` ist die Diagonale des Trainingsfeldes:

```python
import math
max_dist = math.hypot(3.0, 3.0) * 100  # ≈ 424.26 cm (3×3 m Feld)
```

### Tabellen-Index

```c
uint32_t idx = winkel_deg * 201 + abstand_cm;
uint8_t  aktion = pgm_read_byte(&robot_lut[idx]);
```

### Aktionsbedeutung

```
Aktion  A  →  Fahrtrichtung = A × 4°

Aktion  0  →   0°  (vorwärts)
Aktion  9  →  36°
Aktion 22  →  88°
Aktion 45  → 180°  (rückwärts)
Aktion 67  → 268°
Aktion 89  → 356°
```
