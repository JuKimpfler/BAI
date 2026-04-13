#!/usr/bin/env python3
"""
tools/generate_lut_actions.py

Generiert eine Lookup-Table (LUT) aus einem trainierten PyTorch DQN-Modell
(roboter_rl_modell_<neuronen>.pth) und gibt sie als:
  - <out_prefix>_actions.bin   (uint8 Rohdaten, ein Byte pro Zelle)
  - <out_prefix>_actions.h     (C-Header für PlatformIO / Teensy)
aus.

Beispiel:
    python tools/generate_lut_actions.py \\
        --model roboter_rl_modell_64.pth \\
        --out-prefix roboter_lut \\
        --max-distance-cm 200 \\
        --distance-step 1 \\
        --angle-step 1

Indexformel (C-seitig):
    idx = dist_idx * LUT_ANGLE_COUNT + angle_idx
    dist_idx  = clamp(round(distance_cm / LUT_DIST_STEP_CM), 0, LUT_DIST_COUNT - 1)
    angle_idx = round((rel_angle_deg % 360) / LUT_ANGLE_STEP_DEG) % LUT_ANGLE_COUNT

Action → Fahrwinkel:
    angle_deg = action * (360.0 / ANZAHL_AKTIONEN)   # 0..356° (Schrittweite 4° bei 90 Aktionen)
    if angle_deg > 180: angle_deg -= 360              # → [-180..+180]
    # 0° = vorwärts, ±180° = rückwärts
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn

# ─── Modell-Definition (muss mit Trainer_V2.py übereinstimmen) ────────────────

ANZAHL_AKTIONEN = 90


class RoboterDQN(nn.Module):
    def __init__(self, neuronen: int = 64):
        super().__init__()
        self.netzwerk = nn.Sequential(
            nn.Linear(2, neuronen), nn.ReLU(),
            nn.Linear(neuronen, neuronen), nn.ReLU(),
            nn.Linear(neuronen, ANZAHL_AKTIONEN),
        )

    def forward(self, x):
        return self.netzwerk(x)


def normalisiere_zustand(winkel_deg: float, abstand_cm: float, max_dist_cm: float):
    return [winkel_deg / 180.0, abstand_cm / max_dist_cm]


def infer_neurons(state_dict: dict) -> int:
    """Leitet Neuronen-Anzahl aus dem state_dict ab (netzwerk.0.weight: [N, 2])."""
    w = state_dict.get("netzwerk.0.weight")
    if w is not None and len(w.shape) == 2 and w.shape[1] == 2:
        return int(w.shape[0])
    return 64  # Fallback


# ─── LUT-Generator ────────────────────────────────────────────────────────────

def generate_lut(
    model_path: str,
    out_prefix: str,
    max_distance_cm: int = 200,
    distance_step: int = 1,
    angle_step: int = 1,
) -> None:
    # Modell laden
    state_dict = torch.load(model_path, map_location="cpu")
    neuronen = infer_neurons(state_dict)
    model = RoboterDQN(neuronen)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modell geladen: {model_path} (neuronen={neuronen})")

    # Gitterdefinition
    # Abstände: 0 .. max_distance_cm (inklusiv)
    distances = np.arange(0, max_distance_cm + 1, distance_step, dtype=np.int32)
    # Winkel: 0 .. 359° (exklusiv 360)
    angles = np.arange(0, 360, angle_step, dtype=np.int32)
    D, A = len(distances), len(angles)
    total = D * A
    print(f"LUT-Gitter: {D} Abstände × {A} Winkel = {total} Einträge ({total} Bytes)")

    # Normalisierungs-Referenz wie in Trainer_V2.py (3×3 m Feld)
    max_dist_normalize = math.hypot(3.0, 3.0) * 100.0  # ≈ 424.26 cm

    lut = np.zeros(total, dtype=np.uint8)

    with torch.no_grad():
        idx = 0
        for d in distances:
            for a in angles:
                # Winkel [0..360) → relativer Winkel [-180..180]
                rel_angle = float(a) if a <= 180 else float(a) - 360.0
                state = normalisiere_zustand(rel_angle, float(d), max_dist_normalize)
                q = model(torch.tensor([state], dtype=torch.float32))[0]
                lut[idx] = np.uint8(int(torch.argmax(q).item()))
                idx += 1
            if (int(d) % max(1, max_distance_cm // 10)) == 0:
                print(f"  Abstand {d} cm / {max_distance_cm} cm verarbeitet …")

    # ── Binär-Datei ──────────────────────────────────────────────────────────
    bin_path = f"{out_prefix}_actions.bin"
    lut.tofile(bin_path)
    print(f"Geschrieben: {bin_path} ({os.path.getsize(bin_path)} Bytes)")

    # ── C-Header ─────────────────────────────────────────────────────────────
    h_path = f"{out_prefix}_actions.h"
    with open(h_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generierter LUT-Header – NICHT manuell bearbeiten!\n")
        f.write(f"// Quellmodell : {os.path.basename(model_path)}\n")
        f.write(f"// Eintraege   : {total} (uint8, Action-ID 0..{ANZAHL_AKTIONEN - 1})\n")
        f.write(f"// Groesse     : {total} Bytes\n")
        f.write("//\n")
        f.write("// Indexformel:\n")
        f.write("//   idx = dist_idx * LUT_ANGLE_COUNT + angle_idx\n")
        f.write("//   dist_idx  = clamp(round(dist_cm / LUT_DIST_STEP_CM), 0, LUT_DIST_COUNT-1)\n")
        f.write("//   angle_idx = round((angle_deg % 360) / LUT_ANGLE_STEP_DEG) % LUT_ANGLE_COUNT\n")
        f.write("//\n")
        f.write("// Action → Fahrwinkel (Beispiel fuer 90 Aktionen, 4° Schritt):\n")
        f.write("//   float ang = action * (360.0f / 90.0f);\n")
        f.write("//   if (ang > 180.0f) ang -= 360.0f;  // [-180..+180], 0°=vorwaerts\n")
        f.write("\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define LUT_DIST_COUNT      {D}\n")
        f.write(f"#define LUT_ANGLE_COUNT     {A}\n")
        f.write(f"#define LUT_MAX_DISTANCE_CM {max_distance_cm}\n")
        f.write(f"#define LUT_DIST_STEP_CM    {distance_step}\n")
        f.write(f"#define LUT_ANGLE_STEP_DEG  {angle_step}\n\n")
        f.write("const uint8_t LUT_ACTIONS[LUT_DIST_COUNT * LUT_ANGLE_COUNT] = {\n")
        for i, v in enumerate(lut):
            if i % 16 == 0:
                f.write("  ")
            f.write(f"{int(v)}, ")
            if (i + 1) % 16 == 0:
                f.write("\n")
        if total % 16 != 0:
            f.write("\n")
        f.write("};\n")
    print(f"Geschrieben: {h_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generiert LUT-Header (.h) und Binärdatei (.bin) aus trainiertem DQN-Modell.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Pfad zur .pth-Datei (state_dict), z.B. roboter_rl_modell_64.pth")
    parser.add_argument("--out-prefix", required=True,
                        help="Ausgabe-Präfix, z.B. 'roboter_lut' → roboter_lut_actions.h/.bin")
    parser.add_argument("--max-distance-cm", type=int, default=200,
                        help="Maximaler Abstand in cm (inklusiv)")
    parser.add_argument("--distance-step", type=int, default=1,
                        help="Abstandsschrittweite in cm")
    parser.add_argument("--angle-step", type=int, default=1,
                        help="Winkelschrittweite in Grad (muss 360 teilen)")
    args = parser.parse_args()

    if 360 % args.angle_step != 0:
        parser.error("--angle-step muss ein Teiler von 360 sein (z.B. 1, 2, 4, 5, ...)")

    generate_lut(
        model_path=args.model,
        out_prefix=args.out_prefix,
        max_distance_cm=args.max_distance_cm,
        distance_step=args.distance_step,
        angle_step=args.angle_step,
    )


if __name__ == "__main__":
    main()
