#!/usr/bin/env python3
"""
Pipe Tracker GUI - DDS Tabanli
Kamera goruntulerini DDS FrameChunk'tan, SAM3 maskesini DDS SegmentationMask'tan alir.
Algoritma: pipe_algorithm.py (ayni dosya, headless main.py ile ortak)
"""

import sys
import os
import time
import json
import threading

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QGroupBox, QLineEdit, QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from cyclonedds.qos import Qos, Policy

BEST_EFFORT_QOS = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(depth=1))
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, uint8
from dataclasses import dataclass

from pipe_algorithm import MaskProcessor, PipeController, ProcessResult


# DDS types (inline to avoid path issues when running standalone)
@dataclass
class FrameChunk(IdlStruct):
    timestamp: float
    chunk_buffer: sequence[uint8]
    chunk_id_in_frame: int
    total_chunks_in_frame: int
    width: int
    height: int
    encoding: str


@dataclass
class SegmentationMask(IdlStruct):
    timestamp: float
    camera: str
    width: int
    height: int
    mask_data: sequence[uint8]


@dataclass
class StreamCommand(IdlStruct):
    command_type: str
    command_data: str
    timestamp: int
    client_id: str


# --- DDS Frame Assembler ---

class FrameAssembler:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.expected_chunks = None
        self.received_count = 0
        self.width = 0
        self.height = 0
        self.encoding = ""
        self.buffers = []

    def push(self, chunk):
        with self._lock:
            if chunk.chunk_id_in_frame == 0:
                total = int(chunk.total_chunks_in_frame)
                if total <= 0:
                    self.reset()
                    return None
                self.expected_chunks = total
                self.received_count = 0
                self.width = int(chunk.width)
                self.height = int(chunk.height)
                self.encoding = str(chunk.encoding)
                self.buffers = [None] * total

            if self.expected_chunks is None:
                return None
            if (int(chunk.total_chunks_in_frame) != self.expected_chunks
                    or int(chunk.width) != self.width
                    or int(chunk.height) != self.height):
                self.reset()
                return None

            idx = int(chunk.chunk_id_in_frame)
            if idx < 0 or idx >= self.expected_chunks:
                self.reset()
                return None

            if self.buffers[idx] is None:
                self.buffers[idx] = bytes(chunk.chunk_buffer)
                self.received_count += 1

            if self.received_count != self.expected_chunks:
                return None
            if any(part is None for part in self.buffers):
                self.reset()
                return None

            payload = b"".join(self.buffers)
            enc = self.encoding
            self.reset()
            return payload, enc


# --- DDS Camera Reader ---

class DDSCameraReader:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, FrameChunk, qos=BEST_EFFORT_QOS)
        self._reader = DataReader(participant, self._topic, qos=BEST_EFFORT_QOS)
        self._assembler = FrameAssembler()
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                samples = self._reader.take(32)
                for chunk in samples:
                    result = self._assembler.push(chunk)
                    if result is None:
                        continue
                    payload, encoding = result
                    data = np.frombuffer(payload, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    
                    with self._lock:
                        self._frame = img
            except Exception:
                pass
            time.sleep(0.005)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# --- DDS Mask Reader ---

class DDSMaskReader:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, SegmentationMask, qos=BEST_EFFORT_QOS)
        self._reader = DataReader(participant, self._topic, qos=BEST_EFFORT_QOS)
        self._mask = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                samples = self._reader.take(32)
                for sample in samples:
                    raw = np.frombuffer(bytes(sample.mask_data), dtype=np.uint8)
                    mask = raw.reshape(sample.height, sample.width)
                    with self._lock:
                        self._mask = mask
            except Exception:
                pass
            time.sleep(0.005)

    def get_mask(self):
        with self._lock:
            return self._mask.copy() if self._mask is not None else None


# --- DDS Command Publisher ---

class DDSCommandPublisher:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, StreamCommand, qos=BEST_EFFORT_QOS)
        self._writer = DataWriter(participant, self._topic, qos=BEST_EFFORT_QOS)

    def send(self, rc: dict):
        self._writer.write(StreamCommand(
            command_type="motor_rc",
            command_data=json.dumps(rc),
            timestamp=int(time.time() * 1000),
        ))


# --- Main GUI ---

class PipeTrackerWindow(QMainWindow):
    def __init__(self, participant):
        super().__init__()
        self._participant = participant
        self.tracking = False
        self.heading_deg = 0.0
        self.last_cmd = {}

        self.processor = MaskProcessor(
            num_slices=8,
            slice_weights=[0.25, 0.20, 0.16, 0.12, 0.09, 0.07, 0.06, 0.05],
        )
        self.controller = PipeController()

        self.front_cam = DDSCameraReader(participant, "camera/front/frame")
        self.bottom_cam = DDSCameraReader(participant, "camera/bottom/frame")
        self.front_mask_reader = DDSMaskReader(participant, "sam3/front/segmentation_mask")
        self.bottom_mask_reader = DDSMaskReader(participant, "sam3/bottom/segmentation_mask")
        self.cmd_pub = DDSCommandPublisher(participant, "embedded/control/stream_command")

        self.front_cam.start()
        self.bottom_cam.start()
        self.front_mask_reader.start()
        self.bottom_mask_reader.start()

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

    def _init_ui(self):
        self.setWindowTitle("Pipe Tracker - DDS (tauv-pipe)")
        self.setMinimumSize(1250, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QGroupBox { color: #e94560; font-weight: bold; border: 1px solid #533483;
                        border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #eee; }
            QPushButton { background: #16213e; color: white; border: 1px solid #533483;
                          border-radius: 5px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background: #1a1a40; border-color: #e94560; }
            QPushButton:pressed { background: #533483; }
            QTextEdit { background: #0f0f1a; color: #aaa; border: 1px solid #333;
                        font-family: monospace; font-size: 11px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setSpacing(4)

        # Top: Mission controls
        top = QHBoxLayout()
        tg = QGroupBox("Gorev")
        tl = QHBoxLayout(tg)
        self.btn_start = QPushButton("BASLAT")
        self.btn_start.setStyleSheet("background: #1565C0; font-size: 15px; padding: 10px 40px;")
        self.btn_start.clicked.connect(self.on_start)
        tl.addWidget(self.btn_start)
        self.btn_stop = QPushButton("DURDUR")
        self.btn_stop.setStyleSheet("background: #c62828; font-size: 15px; padding: 10px 40px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)
        tl.addWidget(self.btn_stop)
        self.state_label = QLabel("IDLE")
        self.state_label.setStyleSheet("color: #e94560; font-size: 20px; font-weight: bold;")
        tl.addWidget(self.state_label)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        tl.addWidget(self.info_label)
        top.addWidget(tg)
        lay.addLayout(top)

        # Cameras
        cam_row = QHBoxLayout()
        for name, attr in [("On Kamera (Front)", "front_view"), ("Alt Kamera + Mask (Bottom)", "bottom_view")]:
            g = QGroupBox(name)
            v = QVBoxLayout(g)
            lbl = QLabel("DDS bekleniyor...")
            lbl.setFixedSize(560, 380)
            lbl.setStyleSheet("background: #111; border: 1px solid #333;")
            lbl.setAlignment(Qt.AlignCenter)
            setattr(self, attr, lbl)
            v.addWidget(lbl)
            cam_row.addWidget(g)
        lay.addLayout(cam_row)

        # SAM3 Prompt
        sam3_group = QGroupBox("SAM3 Prompt")
        sam3_lay = QHBoxLayout(sam3_group)
        sam3_lay.addWidget(QLabel("Front:"))
        self.front_prompt_input = QLineEdit("pipe")
        self.front_prompt_input.setMaximumWidth(120)
        self.front_prompt_input.setStyleSheet("background: #16213e; color: white; border: 1px solid #533483; padding: 3px;")
        sam3_lay.addWidget(self.front_prompt_input)
        sam3_lay.addWidget(QLabel("Bottom:"))
        self.bottom_prompt_input = QLineEdit("pipe")
        self.bottom_prompt_input.setMaximumWidth(120)
        self.bottom_prompt_input.setStyleSheet("background: #16213e; color: white; border: 1px solid #533483; padding: 3px;")
        sam3_lay.addWidget(self.bottom_prompt_input)
        sam3_lay.addWidget(QLabel("URL:"))
        self.sam3_url_input = QLineEdit("http://localhost:5003")
        self.sam3_url_input.setMaximumWidth(200)
        self.sam3_url_input.setStyleSheet("background: #16213e; color: white; border: 1px solid #533483; padding: 3px;")
        sam3_lay.addWidget(self.sam3_url_input)
        self.btn_send_prompt = QPushButton("Gonder")
        self.btn_send_prompt.setStyleSheet("background: #e94560; padding: 6px 20px;")
        self.btn_send_prompt.clicked.connect(self._send_prompts)
        sam3_lay.addWidget(self.btn_send_prompt)
        self.sam3_status_label = QLabel("")
        self.sam3_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        sam3_lay.addWidget(self.sam3_status_label)
        sam3_lay.addStretch()
        lay.addWidget(sam3_group)

        # Tuning
        tune_group = QGroupBox("Algoritma Tuning")
        tg_lay = QGridLayout(tune_group)
        self.tune_inputs = {}

        tune_defs = [
            ("kp_yaw",               "Kp Yaw",          "150"),
            ("ki_yaw",               "Ki Yaw",           "200"),
            ("forward_pwm",          "Ileri PWM",        "200"),
            ("max_yaw_pwm",          "Max Yaw PWM",      "200"),
            ("ema_alpha",            "EMA Alpha",        "0.6"),
            ("turn_curvature_thresh","Viraj Esigi",      "0.03"),
            ("turn_forward_pwm",     "Viraj Ileri PWM",  "150"),
            ("turn_yaw_boost",       "Viraj Yaw Boost",  "3.0"),
            ("coast_timeout",        "Coast Timeout (s)", "1.5"),
            ("coast_yaw_decay",      "Coast Decay",      "0.55"),
        ]

        for i, (key, label, default) in enumerate(tune_defs):
            row = i // 5
            col = (i % 5) * 2
            tg_lay.addWidget(QLabel(label), row * 2, col)
            inp = QLineEdit(default)
            inp.setMaximumWidth(70)
            inp.setStyleSheet("background: #16213e; color: white; border: 1px solid #533483; padding: 3px;")
            tg_lay.addWidget(inp, row * 2, col + 1)
            self.tune_inputs[key] = inp

        self.btn_apply_tune = QPushButton("Uygula")
        self.btn_apply_tune.setStyleSheet("background: #e94560; padding: 6px 20px;")
        self.btn_apply_tune.clicked.connect(self._apply_tune)
        tg_lay.addWidget(self.btn_apply_tune, 2, 8, 1, 2)
        lay.addWidget(tune_group)

        # Log
        lg = QGroupBox("Log")
        ll = QVBoxLayout(lg)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        ll.addWidget(self.log_text)
        lay.addWidget(lg)

    def _send_prompts(self):
        import requests
        url = self.sam3_url_input.text().strip().rstrip("/")
        results = []
        for camera, inp in [("front", self.front_prompt_input), ("bottom", self.bottom_prompt_input)]:
            prompt = inp.text().strip()
            if not prompt:
                continue
            try:
                resp = requests.post(
                    f"{url}/api/prompt",
                    json={"camera": camera, "prompt": prompt},
                    timeout=5,
                )
                if resp.ok:
                    results.append(f"{camera}=OK")
                else:
                    results.append(f"{camera}=HATA({resp.status_code})")
            except Exception as e:
                results.append(f"{camera}=BAGLANTI HATASI")
        status = "  ".join(results) if results else "Prompt bos"
        self.sam3_status_label.setText(status)
        self._log(f"SAM3 prompt: {status}")

    def _apply_tune(self):
        try:
            s = self.tune_inputs
            self.controller.update_params(
                kp_yaw=float(s["kp_yaw"].text()),
                ki_yaw=float(s["ki_yaw"].text()),
                forward_pwm=int(s["forward_pwm"].text()),
                max_yaw_pwm=int(s["max_yaw_pwm"].text()),
                ema_alpha=float(s["ema_alpha"].text()),
                turn_curvature_thresh=float(s["turn_curvature_thresh"].text()),
                turn_forward_pwm=int(s["turn_forward_pwm"].text()),
                turn_yaw_boost=float(s["turn_yaw_boost"].text()),
                coast_timeout=float(s["coast_timeout"].text()),
                coast_yaw_decay=float(s["coast_yaw_decay"].text()),
            )
            self._log("Tuning uygulandi")
        except ValueError as e:
            self._log(f"Tuning hatasi: {e}")

    def _overlay_mask(self, frame, raw_mask):
        """Kamera frame uzerine SAM3 maskesini yesil overlay olarak ciz."""
        h_cam, w_cam = frame.shape[:2]
        h_mask, w_mask = raw_mask.shape[:2]
        if (h_mask, w_mask) != (h_cam, w_cam):
            binary = cv2.resize(raw_mask, (w_cam, h_cam), interpolation=cv2.INTER_NEAREST)
        else:
            binary = raw_mask.copy()
        _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)
        annotated = frame.copy()
        overlay = annotated.copy()
        overlay[binary > 0] = [0, 255, 0]
        annotated = cv2.addWeighted(annotated, 0.65, overlay, 0.35, 0)
        return annotated, binary

    def _tick(self):
        front = self.front_cam.get_frame()
        front_mask = self.front_mask_reader.get_mask()

        if front is not None and front_mask is not None:
            annotated_front, _ = self._overlay_mask(front, front_mask)
            self._show(self.front_view, annotated_front)
        elif front is not None:
            self._show(self.front_view, front)

        bottom = self.bottom_cam.get_frame()
        bottom_mask = self.bottom_mask_reader.get_mask()
        bottom_result = None

        if bottom is not None and bottom_mask is not None:
            annotated_bottom, binary = self._overlay_mask(bottom, bottom_mask)
            bottom_result = self.processor.process(binary)
            self._draw_slices(annotated_bottom, bottom_result)
            self._show(self.bottom_view, annotated_bottom)
        elif bottom is not None:
            self._show(self.bottom_view, bottom)

        if not self.tracking:
            return

        cmd = self.controller.compute(bottom_result, heading_deg=self.heading_deg)
        self.last_cmd = cmd
        st = self.controller.state
        self.state_label.setText(st)

        if st == PipeController.STATE_COMPLETE:
            self.state_label.setText("GOREV TAMAMLANDI")
            self.state_label.setStyleSheet("color: #4CAF50; font-size: 20px; font-weight: bold;")
            self.tracking = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("GOREV TAMAMLANDI - 2 gecis bitti")
            self.cmd_pub.send(cmd)
            return

        info_parts = [f"Gecis: {self.controller.pass_count}/2"]
        if st == PipeController.STATE_REVERSE:
            target = self.controller._reverse_target_hdg
            diff = (target - self.heading_deg + 180) % 360 - 180
            info_parts.append(f"| 180 DONUS: hedef={target:.0f} fark={diff:+.0f}")
        elif st == PipeController.STATE_REACQUIRE:
            elapsed = time.time() - self.controller._reacquire_start
            info_parts.append(f"| BORU ARANIYOR ({elapsed:.1f}s/5.0s)")
        elif bottom_result and bottom_result.error is not None:
            info_parts.append(f"err={bottom_result.error:+.2f}")
            info_parts.append(f"ang={bottom_result.pipe_angle_deg:+.0f}")
            cont = "DEVAM" if bottom_result.pipe_continues else "SON"
            info_parts.append(f"scan={bottom_result.scan_hit_count}({cont})")
        else:
            info_parts.append("boru yok")
        info_parts.append(f"| CMD yaw={cmd['yaw']} fwd={cmd['forward']}")
        self.info_label.setText("  ".join(info_parts))

        self.cmd_pub.send(cmd)

    def _draw_slices(self, frame, result):
        if result is None or result.error is None or len(result.slice_centroids) < 2:
            return
        h, w = frame.shape[:2]
        num = result.num_slices
        slice_h = h // num

        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        CYAN = (255, 255, 0)
        MAGENTA = (255, 0, 255)
        WHITE = (255, 255, 255)
        GRAY = (100, 100, 100)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for i in range(1, num):
            y = i * slice_h
            cv2.line(bgr, (0, y), (w, y), GRAY, 1)

        weights = self.processor.slice_weights
        for idx, cx, cy, area in result.slice_centroids:
            weight = weights[idx] if idx < len(weights) else 0
            radius = max(4, int(weight * 40))
            cv2.circle(bgr, (int(cx), cy), radius, CYAN, -1)
            cv2.circle(bgr, (int(cx), cy), radius, YELLOW, 2)
            cv2.putText(bgr, f"w={weight:.2f}", (int(cx) + radius + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, YELLOW, 1)

        center_x = int(result.center_x)
        cv2.line(bgr, (center_x, 0), (center_x, h), RED, 1)

        wcx = int(result.weighted_cx)
        cv2.line(bgr, (wcx, 0), (wcx, h), GREEN, 2)

        if len(result.slice_centroids) >= 2:
            sorted_c = sorted(result.slice_centroids, key=lambda c: c[0])
            pts = [(int(c[1]), c[2]) for c in sorted_c]
            for j in range(len(pts) - 1):
                cv2.line(bgr, pts[j], pts[j+1], MAGENTA, 2)

        for x1, y1, x2, y2, hit in result.scan_rays:
            color = WHITE if hit else RED
            cv2.line(bgr, (x1, y1), (x2, y2), color, 1)
            cv2.circle(bgr, (x2, y2), 3, color, -1)

        cont_str = "DEVAM" if result.pipe_continues else "SON"
        cont_color = GREEN if result.pipe_continues else RED
        cv2.putText(bgr, f"err={result.error:+.2f}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
        cv2.putText(bgr, f"angle={result.pipe_angle_deg:+.0f}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1)
        cv2.putText(bgr, f"curv={result.curvature:+.2f}", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
        cv2.putText(bgr, f"scan={result.scan_hit_count} {cont_str}", (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cont_color, 1)

        rgb_back = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame[:] = rgb_back

    def _show(self, label, frame_rgb):
        h, w = frame_rgb.shape[:2]
        dw, dh = label.width(), label.height()
        sc = min(dw / w, dh / h)
        nw, nh = int(w * sc), int(h * sc)
        r = cv2.resize(frame_rgb, (nw, nh))
        qi = QImage(r.data, nw, nh, nw * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi))

    def on_start(self):
        self.controller.reset()
        self.tracking = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.state_label.setText("FOLLOW")
        self.state_label.setStyleSheet("color: #e94560; font-size: 20px; font-weight: bold;")
        self._log("Takip baslatildi (DDS kamera + SAM3 mask)")

    def on_stop(self):
        self.tracking = False
        self.cmd_pub.send(self.controller.stop_cmd())
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.state_label.setText("IDLE")
        self.info_label.setText("")
        self._log("Takip durduruldu")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.tracking = False
        self.front_cam.stop()
        self.bottom_cam.stop()
        self.front_mask_reader.stop()
        self.bottom_mask_reader.stop()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipe Tracker GUI (DDS)")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID")
    args = parser.parse_args()

    print("=" * 60)
    print("PIPE TRACKER GUI - DDS (tauv-pipe)")
    print("=" * 60)
    print("Kameralar: DDS FrameChunk (camera/front|bottom/frame/rgb8)")
    print("Maske: DDS SegmentationMask (sam3/bottom/segmentation_mask)")
    print("Komut: DDS StreamCommand (embedded/control/stream_command)")
    print("Algoritma: pipe_algorithm.py")
    print("=" * 60)

    participant = DomainParticipant(domain_id=args.domain_id)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(26, 26, 46))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(22, 33, 62))
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.Button, QColor(22, 33, 62))
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.Highlight, QColor(233, 69, 96))
    p.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(p)

    w = PipeTrackerWindow(participant)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
