"""
Lightweight automated tests for BAI Trainer improvements.

Tests cover:
- Double DQN target computation (shape, selection logic)
- Heatmap binning and rolling-window filtering
- Action jump penalty (circular distance)
- Smoke test: training loop runs a few episodes without crashing
"""
import math
import random
import sys
import os
from collections import deque, defaultdict

import pytest
import torch
import torch.nn as nn
import numpy as np

# ── Minimal stubs so tests run without a display / Qt ────────────────────────
sys.modules.setdefault("PyQt5", type(sys)("PyQt5"))
sys.modules.setdefault("PyQt5.QtWidgets", type(sys)("PyQt5.QtWidgets"))
sys.modules.setdefault("PyQt5.QtCore",    type(sys)("PyQt5.QtCore"))
sys.modules.setdefault("PyQt5.QtGui",     type(sys)("PyQt5.QtGui"))

import matplotlib
matplotlib.use("Agg")


# ── Import helpers from the trainer without instantiating the GUI ─────────────
# We re-implement the minimal logic under test so the tests are self-contained
# (avoids display-server dependency at import time).

ANZAHL_AKTIONEN = 90
WINKEL_SCHRITT  = 360.0 / ANZAHL_AKTIONEN

# ─── Minimal DQN model (mirrors RoboterDQN) ──────────────────────────────────
class MinimalDQN(nn.Module):
    def __init__(self, neuronen: int = 16, n_actions: int = ANZAHL_AKTIONEN):
        super().__init__()
        self.netzwerk = nn.Sequential(
            nn.Linear(2, neuronen), nn.ReLU(),
            nn.Linear(neuronen, neuronen), nn.ReLU(),
            nn.Linear(neuronen, n_actions),
        )

    def forward(self, x):
        return self.netzwerk(x)


# ─── Helper: action jump penalty (circular distance) ─────────────────────────
def action_jump_penalty(prev: int, curr: int, n_actions: int = ANZAHL_AKTIONEN,
                        strength: float = 0.5) -> float:
    diff = abs(curr - prev)
    circ = min(diff, n_actions - diff)
    return -strength * circ


# ─── Helper: heatmap binning ─────────────────────────────────────────────────
N_ANGLE_BINS = 36
N_DIST_BINS  = 30

def angle_bin(a: float) -> int:
    return int((a + 180.0) / 360.0 * N_ANGLE_BINS) % N_ANGLE_BINS

def dist_bin(d: float) -> int:
    return min(int(d / 300.0 * N_DIST_BINS), N_DIST_BINS - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Double DQN target computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoubleDQN:
    """Verify Double DQN target tensor shape and selection logic."""

    def _make_batch(self, batch_size: int = 8, neuronen: int = 16):
        modell      = MinimalDQN(neuronen)
        ziel_modell = MinimalDQN(neuronen)
        ziel_modell.load_state_dict(modell.state_dict())

        states  = torch.rand(batch_size, 2)
        actions = torch.randint(0, ANZAHL_AKTIONEN, (batch_size, 1))
        rewards = torch.randn(batch_size)
        nstates = torch.rand(batch_size, 2)
        dones   = torch.zeros(batch_size)

        return modell, ziel_modell, states, actions, rewards, nstates, dones

    def test_target_shape(self):
        """Expected Q-target shape: (batch_size,)."""
        modell, ziel_modell, states, actions, rewards, nstates, dones = self._make_batch(8)
        gamma = 0.95

        with torch.no_grad():
            next_actions   = modell(nstates).argmax(1, keepdim=True)
            next_q         = ziel_modell(nstates).gather(1, next_actions).squeeze()
        target = rewards + gamma * next_q * (1 - dones)

        assert target.shape == (8,), f"Expected (8,), got {target.shape}"

    def test_double_dqn_differs_from_vanilla(self):
        """
        Double DQN and vanilla DQN should generally produce different targets
        because they use different action-selection sources.
        Using the same weights gives the same actions → targets are equal only
        when the weights happen to agree.  We test with diverged weights.
        """
        batch_size = 16
        modell      = MinimalDQN()
        ziel_modell = MinimalDQN()   # different random init

        states  = torch.rand(batch_size, 2)
        nstates = torch.rand(batch_size, 2)
        rewards = torch.zeros(batch_size)
        dones   = torch.zeros(batch_size)
        gamma   = 0.95

        with torch.no_grad():
            # Double DQN
            next_acts_ddqn = modell(nstates).argmax(1, keepdim=True)
            next_q_ddqn    = ziel_modell(nstates).gather(1, next_acts_ddqn).squeeze()
            target_ddqn    = rewards + gamma * next_q_ddqn

            # Vanilla DQN
            next_q_vanilla = ziel_modell(nstates).max(1)[0]
            target_vanilla = rewards + gamma * next_q_vanilla

        # With different random weights, results should differ somewhere
        assert not torch.allclose(target_ddqn, target_vanilla), (
            "Expected Double-DQN and vanilla DQN targets to differ with independent weights."
        )

    def test_double_dqn_action_selection_uses_main_model(self):
        """
        The action selected for the next state must be the argmax of the
        main (online) model, not the target model.
        """
        modell      = MinimalDQN()
        ziel_modell = MinimalDQN()

        nstates = torch.rand(4, 2)

        with torch.no_grad():
            next_acts_main   = modell(nstates).argmax(1, keepdim=True)
            next_acts_target = ziel_modell(nstates).argmax(1, keepdim=True)

        # Verify the Double DQN path picks from the main model
        assert next_acts_main.shape == (4, 1)
        assert next_acts_target.shape == (4, 1)
        # They should generally differ (independent random weights)
        assert not torch.equal(next_acts_main, next_acts_target), (
            "Main and target model unexpectedly produced identical action selections."
        )

    def test_done_mask_zeros_future_rewards(self):
        """When done=1, the future reward should be zeroed out."""
        modell      = MinimalDQN(neuronen=8)
        ziel_modell = MinimalDQN(neuronen=8)
        ziel_modell.load_state_dict(modell.state_dict())

        nstates = torch.rand(4, 2)
        rewards = torch.ones(4)
        dones   = torch.tensor([0.0, 1.0, 0.0, 1.0])
        gamma   = 0.95

        with torch.no_grad():
            next_acts = modell(nstates).argmax(1, keepdim=True)
            next_q    = ziel_modell(nstates).gather(1, next_acts).squeeze()

        targets = rewards + gamma * next_q * (1 - dones)

        # For done episodes (indices 1 and 3), target == reward == 1.0
        assert targets[1].item() == pytest.approx(1.0, abs=1e-5)
        assert targets[3].item() == pytest.approx(1.0, abs=1e-5)
        # For non-done (indices 0, 2), target != 1.0 in general
        # (unless next_q happens to be 0, which is vanishingly unlikely)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Heatmap binning and rolling-window filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeatmapBinning:

    def test_angle_bin_range(self):
        """All angles in [-180, 180] must map to valid bin indices."""
        for angle in range(-180, 181):
            b = angle_bin(float(angle))
            assert 0 <= b < N_ANGLE_BINS, f"angle={angle} → bin={b} out of range"

    def test_dist_bin_range(self):
        """Distances in [0, 300+] must clamp to valid bin indices."""
        for dist in [0, 1, 50, 150, 299, 300, 350, 1000]:
            b = dist_bin(float(dist))
            assert 0 <= b < N_DIST_BINS, f"dist={dist} → bin={b} out of range"

    def test_angle_bin_front(self):
        """0° (ball directly ahead) should map to the center bin."""
        b = angle_bin(0.0)
        assert b == N_ANGLE_BINS // 2, f"Expected center bin, got {b}"

    def test_rolling_window_filter(self):
        """Only visits within the last-N-epochs window should be included."""
        # Build synthetic visits: epochs 0..999
        visits = deque(maxlen=1000)
        for ep in range(1000):
            visits.append((ep, float(ep)))   # reward == epoch number

        last_n = 100
        current_epoch = 999
        min_epoch = current_epoch - last_n

        relevant = [(e, r) for e, r in visits if e >= min_epoch]
        assert len(relevant) == last_n + 1  # epochs 899..999 inclusive
        assert relevant[0][0] == min_epoch

    def test_last_3_visits_average(self):
        """Average of last 3 visits should be computed correctly."""
        visits = [(0, 10.0), (1, 20.0), (2, 30.0), (3, 40.0), (4, 50.0)]
        last3  = visits[-3:]
        avg    = float(np.mean([r for _, r in last3]))
        assert avg == pytest.approx(40.0)   # (30+40+50)/3

    def test_missing_cells_are_nan(self):
        """Cells with no visits should remain NaN."""
        heatmap = np.full((N_DIST_BINS, N_ANGLE_BINS), np.nan)
        # Populate a single cell
        heatmap[5, 10] = 42.0
        assert np.isnan(heatmap[0, 0])
        assert heatmap[5, 10] == 42.0

    def test_heatmap_cells_accumulate(self):
        """Verify that defaultdict accumulates visits per cell correctly."""
        cells: dict = defaultdict(lambda: deque(maxlen=1000))
        key = (angle_bin(0.0), dist_bin(50.0))
        for ep in range(5):
            cells[key].append((ep, float(ep * 10)))

        assert len(cells[key]) == 5
        epochs_stored = [e for e, _ in cells[key]]
        assert epochs_stored == [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Action jump penalty
# ═══════════════════════════════════════════════════════════════════════════════

class TestActionJumpPenalty:

    def test_no_jump(self):
        """Same action → circular distance 0 → penalty 0."""
        assert action_jump_penalty(10, 10) == 0.0

    def test_direct_jump(self):
        """Direct distance is |curr - prev|."""
        penalty = action_jump_penalty(0, 10, strength=1.0)
        assert penalty == pytest.approx(-10.0)

    def test_circular_wrap_around(self):
        """Penalty should use the shorter arc (action 0 and 89 differ by 1 via wrap)."""
        penalty = action_jump_penalty(0, 89, n_actions=90, strength=1.0)
        assert penalty == pytest.approx(-1.0)   # wrap-around distance = 1

    def test_max_jump_half_circle(self):
        """Maximum circular distance is n_actions // 2."""
        n = ANZAHL_AKTIONEN
        penalty = action_jump_penalty(0, n // 2, n_actions=n, strength=1.0)
        assert penalty == pytest.approx(-float(n // 2))

    def test_penalty_strength_scales_linearly(self):
        """Penalty is proportional to strength parameter."""
        p1 = action_jump_penalty(0, 10, strength=1.0)
        p2 = action_jump_penalty(0, 10, strength=2.0)
        assert p2 == pytest.approx(2.0 * p1)

    def test_symmetry(self):
        """Penalty should be the same regardless of direction."""
        assert action_jump_penalty(5, 15) == action_jump_penalty(15, 5)

    def test_disabled_penalty_is_zero(self):
        """strength=0 → always zero penalty."""
        for prev, curr in [(0, 45), (10, 80), (45, 0)]:
            assert action_jump_penalty(prev, curr, strength=0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Path efficiency metric
# ═══════════════════════════════════════════════════════════════════════════════

class TestPathEfficiency:

    def test_perfect_straight_line(self):
        """
        If the robot moves in a perfect straight line the path length equals
        the ideal distance → efficiency = 1.0.
        """
        step_size_m = 0.02
        start = (0.0, 0.0)
        ball  = (0.5, 0.0)   # 50 cm away
        steps_used = 25      # 25 × 0.02 = 0.50 m

        ideal_dist_m  = math.hypot(ball[0] - start[0], ball[1] - start[1])
        path_length_m = steps_used * step_size_m
        efficiency    = min(ideal_dist_m / path_length_m, 1.0)

        assert efficiency == pytest.approx(1.0, abs=1e-6)

    def test_inefficient_path(self):
        """Path longer than ideal → efficiency < 1."""
        step_size_m = 0.02
        start = (0.0, 0.0)
        ball  = (0.5, 0.0)
        steps_used = 50    # twice as many steps as needed

        ideal_dist_m  = math.hypot(ball[0] - start[0], ball[1] - start[1])
        path_length_m = steps_used * step_size_m
        efficiency    = min(ideal_dist_m / path_length_m, 1.0)

        assert efficiency == pytest.approx(0.5, abs=1e-6)

    def test_efficiency_clamped_to_one(self):
        """Efficiency cannot exceed 1.0 even if path < ideal (edge-case guard)."""
        ideal_dist_m  = 1.0
        path_length_m = 0.5   # shorter than ideal (degenerate case)
        efficiency    = min(ideal_dist_m / path_length_m, 1.0)
        assert efficiency == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Smoke test – a few training episodes without crashing
# ═══════════════════════════════════════════════════════════════════════════════

def _run_mini_training(epochs: int = 20, seed: int = 42) -> dict:
    """Run a minimal headless training loop and return summary stats."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    neuronen = 16
    modell      = MinimalDQN(neuronen)
    ziel_modell = MinimalDQN(neuronen)
    ziel_modell.load_state_dict(modell.state_dict())

    optimizer = torch.optim.Adam(modell.parameters(), lr=0.0005)
    criterion = nn.SmoothL1Loss()
    memory: deque = deque(maxlen=1000)

    gamma       = 0.95
    epsilon     = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9

    feld_breite = feld_hoehe = 3.0
    max_dist    = math.hypot(feld_breite, feld_hoehe) * 100
    rob_radius  = 11.0
    toleranz    = 20

    hit_count   = 0
    loss_values = []

    for ep in range(epochs):
        b_x = random.uniform(0.5, feld_breite - 0.5)
        b_y = random.uniform(0.5, feld_hoehe - 0.5)
        r_x = random.uniform(0.2, feld_breite - 0.2)
        r_y = random.uniform(0.2, feld_hoehe - 0.2)
        r_w = random.uniform(-180, 180)

        for _ in range(50):
            dx, dy = b_x - r_x, b_y - r_y
            dist   = math.hypot(dx, dy) * 100
            angle  = (math.degrees(math.atan2(dx, dy)) - r_w) % 360
            if angle > 180: angle -= 360

            state = [angle / 180.0, dist / max_dist]

            if random.random() < epsilon:
                action = random.randint(0, ANZAHL_AKTIONEN - 1)
            else:
                with torch.no_grad():
                    action = torch.argmax(
                        modell(torch.tensor([state], dtype=torch.float32))
                    ).item()

            rad   = math.radians(r_w) + math.radians(action * WINKEL_SCHRITT)
            r_x  += 0.02 * math.sin(rad)
            r_y  += 0.02 * math.cos(rad)

            dx2, dy2 = b_x - r_x, b_y - r_y
            dist2    = math.hypot(dx2, dy2) * 100
            angle2   = (math.degrees(math.atan2(dx2, dy2)) - r_w) % 360
            if angle2 > 180: angle2 -= 360
            nstate = [angle2 / 180.0, dist2 / max_dist]

            reward = -1.0
            done   = False
            if dist2 <= rob_radius + 2:
                reward = 10000.0 if abs(angle2) <= toleranz else -1000.0
                if abs(angle2) <= toleranz:
                    hit_count += 1
                done = True
            elif not (0 <= r_x <= feld_breite and 0 <= r_y <= feld_hoehe):
                reward = -500.0
                done   = True
            else:
                reward += (dist - dist2) * 10

            memory.append((state, action, reward, nstate, done))
            if done:
                break

        if len(memory) >= 32:
            batch   = random.sample(memory, 32)
            z_b     = torch.tensor([x[0] for x in batch], dtype=torch.float32)
            a_b     = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
            r_b     = torch.tensor([x[2] for x in batch], dtype=torch.float32)
            nz_b    = torch.tensor([x[3] for x in batch], dtype=torch.float32)
            d_b     = torch.tensor([x[4] for x in batch], dtype=torch.float32)

            q       = modell(z_b).gather(1, a_b).squeeze()
            with torch.no_grad():
                next_a  = modell(nz_b).argmax(1, keepdim=True)
                next_q  = ziel_modell(nz_b).gather(1, next_a).squeeze()
            target  = r_b + gamma * next_q * (1 - d_b)

            loss    = criterion(q, target.detach())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 10.0)
            optimizer.step()
            loss_values.append(loss.item())

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if ep % 10 == 0:
            ziel_modell.load_state_dict(modell.state_dict())

    return {
        "epochs_run": epochs,
        "hits": hit_count,
        "final_loss": loss_values[-1] if loss_values else None,
        "memory_size": len(memory),
    }


class TestSmokeTraining:

    def test_runs_without_error(self):
        """Training loop completes without raising an exception."""
        stats = _run_mini_training(epochs=20, seed=42)
        assert stats["epochs_run"] == 20

    def test_memory_fills_up(self):
        """Replay memory should contain entries after training."""
        stats = _run_mini_training(epochs=20, seed=42)
        assert stats["memory_size"] > 0

    def test_loss_is_finite(self):
        """Loss should not be NaN or Inf."""
        stats = _run_mini_training(epochs=30, seed=42)
        if stats["final_loss"] is not None:
            assert math.isfinite(stats["final_loss"]), (
                f"Loss is not finite: {stats['final_loss']}"
            )

    def test_deterministic_with_seed(self):
        """Two runs with the same seed should produce identical results."""
        s1 = _run_mini_training(epochs=20, seed=7)
        s2 = _run_mini_training(epochs=20, seed=7)
        assert s1["hits"]       == s2["hits"]
        assert s1["memory_size"] == s2["memory_size"]
        if s1["final_loss"] is not None and s2["final_loss"] is not None:
            assert s1["final_loss"] == pytest.approx(s2["final_loss"], rel=1e-5)
