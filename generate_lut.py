"""
generate_lut.py
===============
Generiert eine Lookup-Tabelle (LUT) aus dem trainierten PyTorch-Modell
und schreibt sie als C-Header-Datei für den Teensy 4.0 heraus.

Tabellen-Dimension:
  Winkel  : 0 – 359 Grad (360 Werte, Schrittweite 1°)
  Abstand : 0 – 200 cm   (201 Werte, Schrittweite 1 cm)
  ──────────────────────────────────────────────────────
  Gesamt  : 360 × 201 = 72.360 Einträge  (uint8_t, 0 – 89)

Verwendung:
  pip install torch
  python generate_lut.py                          # nimmt roboter_rl_modell_128.pth
  python generate_lut.py --modell mein_modell.pth --neuronen 256
  python generate_lut.py --neuronen 64 --max-dist 424.26 --out robot_lut.h
"""

import argparse
import math
import os
import sys
import time
from sensor_model import simuliere_ball_sensor_abstand

# ─── Argumente ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LUT-Generator für Teensy 4.0")
parser.add_argument(
    "--modell",
    default=None,
    help="Pfad zur .pth-Modelldatei (Standard: roboter_rl_modell_<neuronen>.pth)",
)
parser.add_argument(
    "--neuronen",
    type=int,
    default=128,
    help="Anzahl der Neuronen pro versteckter Schicht (Standard: 128)",
)
parser.add_argument(
    "--max-dist",
    type=float,
    default=None,
    help=(
        "Maximale Normalisierungs-Distanz in cm (Standard: Diagonale eines 3×3 m Feldes "
        "≈ 424.26 cm)"
    ),
)
parser.add_argument(
    "--out",
    default="robot_lut.h",
    help="Ausgabe-Header-Datei (Standard: robot_lut.h)",
)
args = parser.parse_args()

# ─── PyTorch laden ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit(
        "FEHLER: PyTorch nicht gefunden.\n"
        "Bitte installieren: pip install torch\n"
        "Danach dieses Skript erneut ausführen."
    )

# ─── Modell-Definition (muss mit dem Training übereinstimmen) ─────────────────
ANZAHL_AKTIONEN = 90  # 0 – 89  (4°-Schritte, 90 × 4° = 360°)


class RoboterDQN(nn.Module):
    def __init__(self, neuronen: int = 64):
        super().__init__()
        self.netzwerk = nn.Sequential(
            nn.Linear(3, neuronen),
            nn.ReLU(),
            nn.Linear(neuronen, neuronen),
            nn.ReLU(),
            nn.Linear(neuronen, ANZAHL_AKTIONEN),
        )

    def forward(self, x):
        return self.netzwerk(x)


def normalisiere_zustand(winkel_deg: float, abstand_cm: float, max_dist_cm: float):
    winkel_rad = math.radians(winkel_deg)
    sensor_abstand_cm = simuliere_ball_sensor_abstand(abstand_cm)
    return [math.sin(winkel_rad), math.cos(winkel_rad), sensor_abstand_cm / max_dist_cm]


# ─── Parameter ────────────────────────────────────────────────────────────────
NEURONEN = args.neuronen
MODELL_DATEI = args.modell or f"roboter_rl_modell_{NEURONEN}.pth"
MAX_DIST = args.max_dist if args.max_dist else math.hypot(3.0, 3.0) * 100  # ≈ 424.26 cm
AUSGABE_DATEI = args.out

WINKEL_MIN, WINKEL_MAX = 0, 359   # 360 Werte
ABSTAND_MIN, ABSTAND_MAX = 0, 200  # 201 Werte
ANZAHL_WINKEL = WINKEL_MAX - WINKEL_MIN + 1    # 360
ANZAHL_ABSTAENDE = ABSTAND_MAX - ABSTAND_MIN + 1  # 201
GESAMT_EINTRAEGE = ANZAHL_WINKEL * ANZAHL_ABSTAENDE  # 72.360

print(f"╔══════════════════════════════════════════════════╗")
print(f"║         LUT-Generator für Teensy 4.0            ║")
print(f"╠══════════════════════════════════════════════════╣")
print(f"║  Modelldatei : {MODELL_DATEI:<35}║")
print(f"║  Neuronen    : {NEURONEN:<35}║")
print(f"║  Max-Distanz : {MAX_DIST:<35.2f}║")
print(f"║  Ausgabe     : {AUSGABE_DATEI:<35}║")
print(f"║  Einträge    : {GESAMT_EINTRAEGE:<35,}║")
print(f"╚══════════════════════════════════════════════════╝")

# ─── Modell laden ─────────────────────────────────────────────────────────────
if not os.path.exists(MODELL_DATEI):
    sys.exit(
        f"FEHLER: Modelldatei '{MODELL_DATEI}' nicht gefunden.\n"
        "Bitte erst trainieren oder --modell / --neuronen korrekt angeben."
    )

modell = RoboterDQN(NEURONEN)
try:
    modell.load_state_dict(torch.load(MODELL_DATEI, map_location="cpu"))
except RuntimeError as e:
    sys.exit(
        f"FEHLER beim Laden des Modells: {e}\n"
        "Überprüfe die --neuronen Angabe (muss mit dem Training übereinstimmen)."
    )
modell.eval()
print(f"\nModell '{MODELL_DATEI}' erfolgreich geladen.\n")

# ─── LUT berechnen ────────────────────────────────────────────────────────────
print(f"Berechne {GESAMT_EINTRAEGE:,} Einträge ...")
t_start = time.time()

lut = []  # Flache Liste: lut[winkel * 201 + abstand]

# Alle Zustände als Batch vorbereiten – deutlich schneller als Einzelberechnungen
batch_inputs = []
for winkel in range(WINKEL_MIN, WINKEL_MAX + 1):
    for abstand in range(ABSTAND_MIN, ABSTAND_MAX + 1):
        batch_inputs.append(normalisiere_zustand(winkel, abstand, MAX_DIST))

tensor_batch = torch.tensor(batch_inputs, dtype=torch.float32)

with torch.no_grad():
    ausgaben = modell(tensor_batch)           # Shape: (72360, 90)
    aktionen = torch.argmax(ausgaben, dim=1)  # Shape: (72360,)  Werte 0–89
    lut = aktionen.numpy().tolist()

t_ende = time.time()
print(f"Fertig in {t_ende - t_start:.2f} s  –  {len(lut):,} Einträge berechnet.\n")

# ─── C-Header schreiben ───────────────────────────────────────────────────────
print(f"Schreibe '{AUSGABE_DATEI}' ...")

ZEILEN_PRO_BLOCK = 201  # Eine Zeile = ein Winkelwert (alle 201 Abstände)

header_text = f"""\
// robot_lut.h
// ===========
// Automatisch generiert von generate_lut.py
// Modell      : {MODELL_DATEI}
// Neuronen    : {NEURONEN}
// Max-Distanz : {MAX_DIST:.4f} cm
// Winkel      : {WINKEL_MIN}–{WINKEL_MAX}° ({ANZAHL_WINKEL} Werte, 1°-Schritte)
// Abstand     : {ABSTAND_MIN}–{ABSTAND_MAX} cm ({ANZAHL_ABSTAENDE} Werte, 1 cm-Schritte)
// Einträge    : {GESAMT_EINTRAEGE} (uint8_t, 0–89)
//
// Zugriff:
//   uint8_t aktion = robot_lut_lookup(winkel_deg, abstand_cm);
//
// Jeder Aktionswert steht für eine Fahrtrichtung:
//   Aktion A → Winkel = A * 4°  (z.B. Aktion 0 = 0°, Aktion 45 = 180°, Aktion 89 = 356°)

#pragma once
#include <Arduino.h>
#include <stdint.h>

// Tabellen-Dimensionen
#define LUT_WINKEL_MIN     {WINKEL_MIN}
#define LUT_WINKEL_MAX     {WINKEL_MAX}
#define LUT_ANZAHL_WINKEL  {ANZAHL_WINKEL}
#define LUT_ABSTAND_MIN    {ABSTAND_MIN}
#define LUT_ABSTAND_MAX    {ABSTAND_MAX}
#define LUT_ANZAHL_ABST    {ANZAHL_ABSTAENDE}
#define LUT_GESAMT         {GESAMT_EINTRAEGE}
#define LUT_AKTIONEN       {ANZAHL_AKTIONEN}
#define LUT_WINKEL_SCHRITT (360.0f / LUT_AKTIONEN)   // 4°

// ─── Lookup-Tabelle im Flash-Speicher (PROGMEM) ───────────────────────────────
// Layout: lut[winkel * LUT_ANZAHL_ABST + abstand]
//   winkel  = 0…359   (1°-Schritte)
//   abstand = 0…200   (1 cm-Schritte)
static const uint8_t PROGMEM robot_lut[LUT_GESAMT] = {{
"""

with open(AUSGABE_DATEI, "w", encoding="utf-8") as f:
    f.write(header_text)

    for winkel in range(ANZAHL_WINKEL):
        zeile_werte = []
        for abstand in range(ANZAHL_ABSTAENDE):
            idx = winkel * ANZAHL_ABSTAENDE + abstand
            zeile_werte.append(str(lut[idx]))
        zeile = ", ".join(zeile_werte)
        # Komma nach letzter Zeile nur wenn nicht letzter Winkel
        if winkel < ANZAHL_WINKEL - 1:
            f.write(f"  /* {winkel:>3}° */ {zeile},\n")
        else:
            f.write(f"  /* {winkel:>3}° */ {zeile}\n")

    f.write("};\n\n")

    # Hilfsfunktion für einfachen Zugriff
    hilfsfunktionen = f"""\
// ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

/**
 * Liest die empfohlene Aktion (0–{ANZAHL_AKTIONEN - 1}) aus der LUT.
 *
 * @param winkel_deg  Relativer Winkel zum Ziel in Grad (0–359).
 *                    Werte außerhalb werden per Modulo eingeklemmt.
 * @param abstand_cm  Abstand zum Ziel in cm (0–{ABSTAND_MAX}).
 *                    Werte außerhalb werden auf den Rand eingeklemmt.
 * @return            Aktionsindex 0–{ANZAHL_AKTIONEN - 1}.
 *                    Umrechnung in Grad: aktion * {360 // ANZAHL_AKTIONEN}°
 */
inline uint8_t robot_lut_lookup(int winkel_deg, int abstand_cm) {{
  // Winkel in den Bereich 0–359 bringen
  winkel_deg = ((winkel_deg % LUT_ANZAHL_WINKEL) + LUT_ANZAHL_WINKEL) % LUT_ANZAHL_WINKEL;
  // Abstand einschränken
  if (abstand_cm < LUT_ABSTAND_MIN) abstand_cm = LUT_ABSTAND_MIN;
  if (abstand_cm > LUT_ABSTAND_MAX) abstand_cm = LUT_ABSTAND_MAX;

  uint32_t idx = (uint32_t)winkel_deg * LUT_ANZAHL_ABST + (uint32_t)abstand_cm;
  return pgm_read_byte(&robot_lut[idx]);
}}

/**
 * Gibt den Fahrtwinkel in Grad zurück, der dem Aktionsindex entspricht.
 * Bereich: 0–{360 - 360 // ANZAHL_AKTIONEN}°
 */
inline float robot_aktion_zu_winkel(uint8_t aktion) {{
  return aktion * LUT_WINKEL_SCHRITT;
}}
"""
    f.write(hilfsfunktionen)

print(f"Header-Datei '{AUSGABE_DATEI}' geschrieben ({os.path.getsize(AUSGABE_DATEI):,} Bytes).\n")
print("Fertig! Nächste Schritte:")
print(f"  1. '{AUSGABE_DATEI}' in den Ordner 'teensy/include/' kopieren")
print("  2. PlatformIO-Projekt öffnen: code teensy/")
print("  3. 'pio run --target upload' ausführen")
print()
print("Beispiel-Zugriff im Teensy-Sketch:")
print("  #include \"robot_lut.h\"")
print("  uint8_t aktion = robot_lut_lookup(winkel, abstand);")
print(f"  float   grad   = robot_aktion_zu_winkel(aktion);  // 0–{360 - 360 // ANZAHL_AKTIONEN}°")
