#!/usr/bin/env python3
import math
import os
import random
import subprocess
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Analyser import AnalyseWorker
from LUT_Analyser import LUTAnalyseWorker
from LUT_Simplifier import apply_symmetry, parse_lut_header, smooth_lut, write_lut_header
from Trainer_V2 import ANZAHL_AKTIONEN, WINKEL_SCHRITT, RoboterDQN, berechne_zustand


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class WorkflowWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            results = {}
            enabled_steps = [step for step in self.config["steps"] if step["enabled"]]
            total = max(1, len(enabled_steps))

            model_path = os.path.abspath(self.config["model_path"])
            lut_path = os.path.abspath(self.config["lut_path"])
            lut_optimized = os.path.abspath(self.config["lut_optimized_path"])

            done = 0
            for step in enabled_steps:
                if not self._running:
                    self.log_signal.emit("Workflow gestoppt.")
                    break

                name = step["name"]
                self.log_signal.emit(f"Starte Schritt: {name}")

                if name == "Training":
                    train_result = self._run_training(model_path)
                    results["training"] = train_result
                    self.log_signal.emit(f"Training beendet: {model_path}")

                elif name == "LUT erstellen":
                    self._run_generate_lut(model_path, lut_path)
                    results["lut_path"] = lut_path
                    self.log_signal.emit(f"LUT erstellt: {lut_path}")

                elif name == "LUT glätten":
                    self._run_simplify_lut(lut_path, lut_optimized)
                    results["lut_optimized_path"] = lut_optimized
                    self.log_signal.emit(f"Optimierte LUT gespeichert: {lut_optimized}")

                elif name == "Analyser":
                    analysis = self._run_model_analysis()
                    results["analysis"] = analysis
                    self.log_signal.emit("Model-Analyse abgeschlossen.")

                elif name == "LUT Analyser":
                    target_lut = lut_optimized if os.path.exists(lut_optimized) else lut_path
                    lut_analysis = self._run_lut_analysis(target_lut)
                    results["lut_analysis"] = lut_analysis
                    self.log_signal.emit("LUT-Analyse abgeschlossen.")

                done += 1
                self.progress_signal.emit(int(done / total * 100), f"{name} abgeschlossen")

            self.finished_signal.emit(results)
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _epsilon_schedule(self, epochs: int):
        raw = self.config["epsilon_schedule"]
        points = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            pct = max(0.0, min(100.0, float(left.strip())))
            eps = max(0.0, min(1.0, float(right.strip())))
            points.append((int(round((pct / 100.0) * max(1, epochs - 1))), eps))

        if not points:
            points = [(0, 1.0), (int(epochs * 0.6), 0.2), (epochs - 1, 0.02)]

        points.sort(key=lambda x: x[0])
        if points[0][0] != 0:
            points.insert(0, (0, points[0][1]))
        if points[-1][0] != epochs - 1:
            points.append((epochs - 1, points[-1][1]))
        return points

    @staticmethod
    def _epsilon_at(epoch: int, points):
        if epoch <= points[0][0]:
            return points[0][1]
        for i in range(1, len(points)):
            e0, v0 = points[i - 1]
            e1, v1 = points[i]
            if epoch <= e1:
                t = 0.0 if e1 == e0 else (epoch - e0) / (e1 - e0)
                return v0 + (v1 - v0) * t
        return points[-1][1]

    def _run_training(self, model_path: str):
        epochs = self.config["epochs"]
        neurons = self.config["neurons"]
        batch_size = self.config["batch_size"]
        cpu_threads = self.config["cpu_threads"]

        if cpu_threads <= 0:
            cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)

        model = RoboterDQN(neurons)
        target_model = RoboterDQN(neurons)

        field_w, field_h = 3.0, 3.0
        max_dist = math.hypot(field_w, field_h) * 100
        robot_radius = 11.0
        tolerance = 6

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.log_signal.emit("Vorhandenes Modell wird weiter trainiert.")

        target_model.load_state_dict(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        memory = deque(maxlen=50000)

        gamma = 0.95
        hit_history = deque(maxlen=200)
        reward_window = []

        schedule_points = self._epsilon_schedule(epochs)

        for epoch in range(epochs):
            if not self._running:
                break

            epsilon = self._epsilon_at(epoch, schedule_points)

            b_x, b_y = field_w / 2, field_h / 2
            r_x = random.uniform(0.2, field_w - 0.2)
            r_y = random.uniform(0.2, field_h - 0.2)

            if random.random() < 0.6:
                abs_angle = math.degrees(math.atan2(b_x - r_x, b_y - r_y))
                offset = random.uniform(90, 270)
                r_w = (abs_angle + offset) % 360
                if r_w > 180:
                    r_w -= 360
            else:
                r_w = random.uniform(-180, 180)

            episode_reward = 0.0

            for step in range(300):
                rel_w, dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                state = [math.sin(math.radians(rel_w)), math.cos(math.radians(rel_w)), dist / max_dist]

                if random.random() < epsilon:
                    action = random.randint(0, ANZAHL_AKTIONEN - 1)
                else:
                    with torch.no_grad():
                        action = int(torch.argmax(model(torch.tensor([state], dtype=torch.float32))).item())

                target_rel_rad = math.radians(action * WINKEL_SCHRITT)
                global_rad = math.radians(r_w) + target_rel_rad
                r_x += 0.02 * math.sin(global_rad)
                r_y += 0.02 * math.cos(global_rad)

                next_rel_w, next_dist = berechne_zustand(r_x, r_y, r_w, b_x, b_y)
                next_state = [math.sin(math.radians(next_rel_w)), math.cos(math.radians(next_rel_w)), next_dist / max_dist]

                reward = -1.0
                done = False

                if next_dist <= (robot_radius + 2):
                    if abs(next_rel_w) <= tolerance:
                        reward = 100.0
                        hit_history.append(1)
                    else:
                        reward = -10.0
                        hit_history.append(0)
                    done = True
                elif r_x < 0 or r_x > field_w or r_y < 0 or r_y > field_h:
                    reward = -5.0
                    hit_history.append(0)
                    done = True
                else:
                    reward += (dist - next_dist) * 2.0

                episode_reward += reward
                memory.append((state, action, reward, next_state, done))

                if len(memory) >= batch_size and step % 4 == 0:
                    batch = random.sample(memory, batch_size)
                    z_batch = torch.tensor([x[0] for x in batch], dtype=torch.float32)
                    a_batch = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
                    r_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                    nz_batch = torch.tensor([x[3] for x in batch], dtype=torch.float32)
                    d_batch = torch.tensor([x[4] for x in batch], dtype=torch.float32)

                    q_values = model(z_batch).gather(1, a_batch).squeeze()
                    next_q = target_model(nz_batch).max(1)[0]
                    expected_q = r_batch + gamma * next_q * (1 - d_batch)

                    loss = criterion(q_values, expected_q.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    break

            reward_window.append(episode_reward)

            if epoch % 50 == 0:
                target_model.load_state_dict(model.state_dict())

            if epoch % 25 == 0:
                avg_reward = sum(reward_window) / max(1, len(reward_window))
                hit_rate = (sum(hit_history) / len(hit_history)) * 100 if hit_history else 0.0
                self.log_signal.emit(
                    f"Training: Epoche {epoch}/{epochs} | ε={epsilon:.3f} | Reward={avg_reward:.2f} | HitRate={hit_rate:.1f}%"
                )
                reward_window.clear()

            if epoch > 0 and epoch % 2000 == 0:
                torch.save(model.state_dict(), model_path)
                self.log_signal.emit(f"Zwischenspeicher bei Epoche {epoch}: {model_path}")

        torch.save(model.state_dict(), model_path)
        return {
            "model_path": model_path,
            "epochs": epochs,
            "neurons": neurons,
            "threads": cpu_threads,
        }

    def _run_generate_lut(self, model_path: str, lut_path: str):
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "generate_lut.py"),
            "--modell",
            model_path,
            "--neuronen",
            str(self.config["neurons"]),
            "--out",
            lut_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"LUT-Erstellung fehlgeschlagen:\n{proc.stderr or proc.stdout}")

    def _run_simplify_lut(self, source_lut: str, target_lut: str):
        if not os.path.exists(source_lut):
            raise FileNotFoundError(f"LUT-Datei nicht gefunden: {source_lut}")

        side = self.config["smooth_side"]
        radius = self.config["smooth_radius"]

        lut = parse_lut_header(source_lut)
        sym = apply_symmetry(lut, side)
        smooth = smooth_lut(sym, radius)
        write_lut_header(smooth, source_lut, target_lut, side, radius)

    def _run_model_analysis(self):
        result_holder = {"data": None, "error": None}
        worker = AnalyseWorker(
            neuronen=self.config["neurons"],
            feld_w=self.config["analysis_field_w"],
            feld_h=self.config["analysis_field_h"],
            b_x=self.config["analysis_ball_x"],
            b_y=self.config["analysis_ball_y"],
            schritt_cm=self.config["analysis_step_cm"],
            n_orientierungen=self.config["analysis_orientations"],
            max_schritte=self.config["analysis_max_steps"],
        )
        worker.ergebnis_signal.connect(lambda d: result_holder.update(data=d))
        worker.fehler_signal.connect(lambda e: result_holder.update(error=e))
        worker.fortschritt_signal.connect(lambda p, t: self.log_signal.emit(f"Analyser {p}%: {t}"))
        worker.run()
        if result_holder["error"]:
            raise RuntimeError(result_holder["error"])
        return result_holder["data"]

    def _run_lut_analysis(self, lut_path: str):
        result_holder = {"data": None, "error": None}
        worker = LUTAnalyseWorker(
            lut_datei=lut_path,
            feld_w=self.config["analysis_field_w"],
            feld_h=self.config["analysis_field_h"],
            b_x=self.config["analysis_ball_x"],
            b_y=self.config["analysis_ball_y"],
            schritt_cm=self.config["analysis_step_cm"],
            n_orientierungen=self.config["analysis_orientations"],
            max_schritte=self.config["analysis_max_steps"],
        )
        worker.ergebnis_signal.connect(lambda d: result_holder.update(data=d))
        worker.fehler_signal.connect(lambda e: result_holder.update(error=e))
        worker.fortschritt_signal.connect(lambda p, t: self.log_signal.emit(f"LUT-Analyser {p}%: {t}"))
        worker.run()
        if result_holder["error"]:
            raise RuntimeError(result_holder["error"])
        return result_holder["data"]


class UnifiedStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BAI Studio – Gesamtprogramm")
        self.setMinimumSize(1300, 820)
        self.worker = None
        self.child_windows = []
        self.latest_results = {}
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        title = QLabel("BAI Studio · Trainer / Analyser / LUT Workflow")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        main.addWidget(title)

        self.tabs = QTabWidget()
        main.addWidget(self.tabs)

        self.tabs.addTab(self._build_workflow_tab(), "Workflow")
        self.tabs.addTab(self._build_tools_tab(), "Tools")
        self.tabs.addTab(self._build_result_tab(), "Ergebnisse")

    def _build_workflow_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left = QVBoxLayout()
        right = QVBoxLayout()

        grp_steps = QGroupBox("Workflow-Schritte")
        gs = QVBoxLayout(grp_steps)
        self.step_list = QListWidget()
        self.step_list.setDragDropMode(QListWidget.InternalMove)
        for name in ["Training", "LUT erstellen", "LUT glätten", "Analyser", "LUT Analyser"]:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked)
            self.step_list.addItem(item)
        gs.addWidget(self.step_list)

        h_move = QHBoxLayout()
        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_up.clicked.connect(self._move_step_up)
        btn_down.clicked.connect(self._move_step_down)
        h_move.addWidget(btn_up)
        h_move.addWidget(btn_down)
        gs.addLayout(h_move)
        left.addWidget(grp_steps)

        grp_train = QGroupBox("Training")
        ft = QFormLayout(grp_train)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(100, 2_000_000)
        self.spin_epochs.setValue(20_000)
        self.combo_neurons = QComboBox()
        self.combo_neurons.addItems(["64", "128", "256", "400"])
        self.combo_neurons.setCurrentText("128")
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(32, 2048)
        self.spin_batch.setValue(256)
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, 64)
        self.spin_threads.setValue(0)
        self.spin_threads.setToolTip("0 = Auto")
        self.model_name = QComboBox()
        self.model_name.setEditable(True)
        self.model_name.addItems([
            os.path.join(REPO_ROOT, "roboter_rl_modell_64.pth"),
            os.path.join(REPO_ROOT, "roboter_rl_modell_128.pth"),
            os.path.join(REPO_ROOT, "roboter_rl_modell_256.pth"),
            os.path.join(REPO_ROOT, "roboter_rl_modell_400.pth"),
        ])
        ft.addRow("Epochen", self.spin_epochs)
        ft.addRow("Neuronen", self.combo_neurons)
        ft.addRow("Batch-Größe", self.spin_batch)
        ft.addRow("CPU Threads", self.spin_threads)
        ft.addRow("Modelldatei", self.model_name)
        left.addWidget(grp_train)

        grp_sched = QGroupBox("Variabler Zufallsverlauf ε (Prozent:Wert)")
        ls = QVBoxLayout(grp_sched)
        self.txt_schedule = QTextEdit()
        self.txt_schedule.setPlainText("0:1.0\n40:0.35\n75:0.10\n100:0.02")
        self.txt_schedule.setFixedHeight(120)
        ls.addWidget(self.txt_schedule)
        left.addWidget(grp_sched)

        grp_lut = QGroupBox("LUT")
        fl = QFormLayout(grp_lut)
        self.lut_path = QComboBox(); self.lut_path.setEditable(True)
        self.lut_path.addItem(os.path.join(REPO_ROOT, "robot_lut.h"))
        self.lut_opt_path = QComboBox(); self.lut_opt_path.setEditable(True)
        self.lut_opt_path.addItem(os.path.join(REPO_ROOT, "robot_lut_optimiert.h"))
        self.combo_side = QComboBox(); self.combo_side.addItems(["rechts", "links"])
        self.spin_smooth = QSpinBox(); self.spin_smooth.setRange(0, 20); self.spin_smooth.setValue(3)
        fl.addRow("LUT Pfad", self.lut_path)
        fl.addRow("Optimierte LUT", self.lut_opt_path)
        fl.addRow("Symmetrie", self.combo_side)
        fl.addRow("Glätt-Radius", self.spin_smooth)
        right.addWidget(grp_lut)

        grp_analysis = QGroupBox("Analyse")
        fa = QFormLayout(grp_analysis)
        self.analysis_fw = QDoubleSpinBox(); self.analysis_fw.setRange(1.0, 5.0); self.analysis_fw.setValue(3.0)
        self.analysis_fh = QDoubleSpinBox(); self.analysis_fh.setRange(1.0, 5.0); self.analysis_fh.setValue(3.0)
        self.analysis_bx = QDoubleSpinBox(); self.analysis_bx.setRange(0.0, 5.0); self.analysis_bx.setValue(1.5)
        self.analysis_by = QDoubleSpinBox(); self.analysis_by.setRange(0.0, 5.0); self.analysis_by.setValue(1.5)
        self.analysis_step = QSpinBox(); self.analysis_step.setRange(1, 30); self.analysis_step.setValue(5)
        self.analysis_ori = QSpinBox(); self.analysis_ori.setRange(1, 36); self.analysis_ori.setValue(8)
        self.analysis_steps = QSpinBox(); self.analysis_steps.setRange(50, 500); self.analysis_steps.setValue(200)
        fa.addRow("Feld Breite (m)", self.analysis_fw)
        fa.addRow("Feld Höhe (m)", self.analysis_fh)
        fa.addRow("Ball X (m)", self.analysis_bx)
        fa.addRow("Ball Y (m)", self.analysis_by)
        fa.addRow("Grid Schritt (cm)", self.analysis_step)
        fa.addRow("Orientierungen", self.analysis_ori)
        fa.addRow("Max Schritte", self.analysis_steps)
        right.addWidget(grp_analysis)

        grp_run = QGroupBox("Ausführung")
        gr = QVBoxLayout(grp_run)
        hb = QHBoxLayout()
        self.btn_run = QPushButton("Workflow starten")
        self.btn_stop = QPushButton("Stopp")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._start_workflow)
        self.btn_stop.clicked.connect(self._stop_workflow)
        hb.addWidget(self.btn_run)
        hb.addWidget(self.btn_stop)
        gr.addLayout(hb)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        gr.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        gr.addWidget(self.log)
        right.addWidget(grp_run)

        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        return tab

    def _build_tools_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel("Alle bisherigen GUIs sind direkt aus dieser Gesamt-App startbar:")
        layout.addWidget(info)

        grid = QGridLayout()
        tools = [
            ("Trainer", "Trainer_V2", "MainWindow"),
            ("Analyser", "Analyser", "AnalyserWindow"),
            ("LUT Analyser", "LUT_Analyser", "LUTAnalyserWindow"),
            ("LUT Simplifier", "LUT_Simplifier", "LutSimplifierWindow"),
            ("Tester", "Tester", "TesterWindow"),
            ("LUT Tester", "LUT_Tester", "LUTTesterWindow"),
        ]

        for i, (label, module_name, class_name) in enumerate(tools):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, m=module_name, c=class_name: self._open_tool_window(m, c))
            grid.addWidget(btn, i // 2, i % 2)

        layout.addLayout(grid)
        layout.addStretch()
        return tab

    def _build_result_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.result_summary = QLabel("Noch keine Workflow-Ergebnisse vorhanden.")
        self.result_summary.setWordWrap(True)
        layout.addWidget(self.result_summary)

        self.result_fig = Figure(figsize=(10, 5))
        self.result_canvas = FigureCanvas(self.result_fig)
        layout.addWidget(self.result_canvas)
        return tab

    def _move_step_up(self):
        row = self.step_list.currentRow()
        if row > 0:
            item = self.step_list.takeItem(row)
            self.step_list.insertItem(row - 1, item)
            self.step_list.setCurrentRow(row - 1)

    def _move_step_down(self):
        row = self.step_list.currentRow()
        if 0 <= row < self.step_list.count() - 1:
            item = self.step_list.takeItem(row)
            self.step_list.insertItem(row + 1, item)
            self.step_list.setCurrentRow(row + 1)

    def _collect_config(self):
        steps = []
        for i in range(self.step_list.count()):
            it = self.step_list.item(i)
            steps.append({"name": it.text(), "enabled": it.checkState() == Qt.Checked})

        return {
            "steps": steps,
            "epochs": self.spin_epochs.value(),
            "neurons": int(self.combo_neurons.currentText()),
            "batch_size": self.spin_batch.value(),
            "cpu_threads": self.spin_threads.value(),
            "model_path": self.model_name.currentText().strip(),
            "epsilon_schedule": self.txt_schedule.toPlainText(),
            "lut_path": self.lut_path.currentText().strip(),
            "lut_optimized_path": self.lut_opt_path.currentText().strip(),
            "smooth_side": self.combo_side.currentText(),
            "smooth_radius": self.spin_smooth.value(),
            "analysis_field_w": self.analysis_fw.value(),
            "analysis_field_h": self.analysis_fh.value(),
            "analysis_ball_x": self.analysis_bx.value(),
            "analysis_ball_y": self.analysis_by.value(),
            "analysis_step_cm": self.analysis_step.value(),
            "analysis_orientations": self.analysis_ori.value(),
            "analysis_max_steps": self.analysis_steps.value(),
        }

    def _start_workflow(self):
        if self.worker and self.worker.isRunning():
            return

        config = self._collect_config()
        self.worker = WorkflowWorker(config)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)

        self.progress.setValue(0)
        self.log.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._on_log("Workflow gestartet.")
        self.worker.start()

    def _stop_workflow(self):
        if self.worker:
            self.worker.stop()
            self._on_log("Stop-Signal gesendet …")

    def _on_progress(self, pct: int, text: str):
        self.progress.setValue(pct)
        self._on_log(text)

    def _on_log(self, text: str):
        self.log.appendPlainText(text)

    def _on_finished(self, results: dict):
        self.latest_results = results
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._render_results(results)
        self.tabs.setCurrentIndex(2)
        self._on_log("Workflow abgeschlossen.")

    def _on_error(self, error: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Workflow-Fehler", error)
        self._on_log(f"Fehler: {error}")

    def _render_results(self, results: dict):
        lines = []
        if "training" in results:
            tr = results["training"]
            lines.append(
                f"Training: {tr['epochs']} Epochen, {tr['neurons']} Neuronen, Threads={tr['threads']}, Modell={tr['model_path']}"
            )
        if "lut_path" in results:
            lines.append(f"LUT erstellt: {results['lut_path']}")
        if "lut_optimized_path" in results:
            lines.append(f"LUT optimiert: {results['lut_optimized_path']}")

        self.result_summary.setText("\n".join(lines) if lines else "Workflow wurde ohne Ausgabe beendet.")

        self.result_fig.clear()
        axes = self.result_fig.subplots(1, 2)

        drawn = 0
        if results.get("analysis") is not None:
            a = results["analysis"]
            im = axes[drawn].imshow(a["erfolg_map"], origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
            axes[drawn].set_title("Model-Analyse: Erfolgsrate")
            self.result_fig.colorbar(im, ax=axes[drawn], fraction=0.046, pad=0.04)
            drawn += 1

        if results.get("lut_analysis") is not None and drawn < 2:
            a = results["lut_analysis"]
            im = axes[drawn].imshow(a["erfolg_map"], origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
            axes[drawn].set_title("LUT-Analyse: Erfolgsrate")
            self.result_fig.colorbar(im, ax=axes[drawn], fraction=0.046, pad=0.04)
            drawn += 1

        while drawn < 2:
            axes[drawn].axis("off")
            axes[drawn].set_title("Keine Daten")
            drawn += 1

        self.result_fig.tight_layout()
        self.result_canvas.draw()

    def _open_tool_window(self, module_name: str, class_name: str):
        module = __import__(module_name)
        cls = getattr(module, class_name)
        win = cls()
        win.show()
        self.child_windows.append(win)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = UnifiedStudio()
    w.show()
    sys.exit(app.exec_())
