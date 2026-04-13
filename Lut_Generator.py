import math
import numpy as np
import torch
import torch.nn as nn

ANZAHL_AKTIONEN = 90

class RoboterDQN(nn.Module):
    def __init__(self, neuronen=32):
        super().__init__()
        self.netzwerk = nn.Sequential(
            nn.Linear(2, neuronen), nn.ReLU(),
            nn.Linear(neuronen, neuronen), nn.ReLU(),
            nn.Linear(neuronen, ANZAHL_AKTIONEN)
        )
    def forward(self, x): return self.netzwerk(x)

def normalisiere_zustand(winkel_deg, abstand_cm, max_dist_cm):
    return [winkel_deg / 180.0, abstand_cm / max_dist_cm]

def infer_neurons(sd):
    # netzwerk.0.weight: [neuronen, 2]
    w = sd.get("netzwerk.0.weight", None)
    return int(w.shape[0]) if w is not None else 32

def main():
    model_path = "roboter_rl_modell_32.pth"

    # 72.360 Zellen: 0..200cm (1cm) und 0..359° (1°)
    max_distance_cm = 200
    dist_step = 1
    angle_step = 1

    distances = np.arange(0, max_distance_cm + 1, dist_step, dtype=np.int32)  # 201
    angles = np.arange(0, 360, angle_step, dtype=np.int32)                    # 360
    D, A = len(distances), len(angles)
    total = D * A
    print(f"LUT: D={D} A={A} total={total} bytes={total}")

    max_dist_normalize = math.hypot(3.0, 3.0) * 100.0  # wie Trainer_V2.py

    sd = torch.load(model_path, map_location="cpu")
    neuronen = infer_neurons(sd)
    model = RoboterDQN(neuronen)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded {model_path} (neuronen={neuronen})")

    lut = np.zeros(total, dtype=np.uint8)

    idx = 0
    with torch.no_grad():
        for d in distances:
            for a in angles:
                rel_angle = a if a <= 180 else a - 360  # -> [-180..180]
                state = normalisiere_zustand(rel_angle, float(d), max_dist_normalize)
                q = model(torch.tensor([state], dtype=torch.float32))[0]
                action = int(torch.argmax(q).item())   # 0..89
                lut[idx] = np.uint8(action)
                idx += 1

    # Binär (minimal)
    lut.tofile("lut_actions.bin")
    print("Wrote lut_actions.bin")

    # Header (direkt in Flash)
    with open("roboter_lut_actions.h", "w", encoding="utf-8") as f:
        f.write("// Auto-generated LUT (uint8 action id)\n")
        f.write(f"// Source model: {model_path}\n")
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write(f"#define LUT_DIST_COUNT {D}\n")
        f.write(f"#define LUT_ANGLE_COUNT {A}\n")
        f.write(f"#define LUT_MAX_DISTANCE_CM {max_distance_cm}\n")
        f.write(f"#define LUT_DIST_STEP_CM {dist_step}\n")
        f.write(f"#define LUT_ANGLE_STEP_DEG {angle_step}\n\n")
        f.write("const uint8_t LUT_ACTIONS[LUT_DIST_COUNT * LUT_ANGLE_COUNT] = {\n")
        for i, v in enumerate(lut):
            if i % 16 == 0: f.write("  ")
            f.write(f"{int(v)}, ")
            if (i + 1) % 16 == 0: f.write("\n")
        if total % 16 != 0: f.write("\n")
        f.write("};\n")
    print("Wrote roboter_lut_actions.h")

if __name__ == "__main__":
    main()
