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
import math
import threading
from collections import deque

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# tauv-client'i Python path'e al -- monorepo icinde kardes submodule.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tauv-client", "src"))

import cv2
import numpy as np

try:
    from tauv_client import Vehicle
except Exception as _exc:
    Vehicle = None
    _vehicle_import_error = _exc
else:
    _vehicle_import_error = None

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
    # Aligned with tauv-client / tauv-core camera FrameChunk (CDR layout must match).
    source_id: str
    feed: str
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
        self.timestamp = 0.0
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
                self.timestamp = float(chunk.timestamp)
                self.buffers = [None] * total

            if self.expected_chunks is None:
                return None
            if (
                int(chunk.total_chunks_in_frame) != self.expected_chunks
                or int(chunk.width) != self.width
                or int(chunk.height) != self.height
                or str(chunk.encoding) != self.encoding
            ):
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
            ts = self.timestamp
            self.reset()
            return payload, enc, ts


# --- DDS Camera Reader ---

class DDSCameraReader:
    """Camera frame reader. Son N frame'i (ts_ms, rgb) olarak rolling buffer'da
    tutar; SAM3 maskesi geldiginde GUI ayni timestamp'li frame'i buradan ister
    ve overlay senkron olur (live frame + late mask lag'i ortadan kalkar)."""

    BUFFER_SIZE = 120

    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, FrameChunk, qos=BEST_EFFORT_QOS)
        self._reader = DataReader(participant, self._topic, qos=BEST_EFFORT_QOS)
        self._assembler = FrameAssembler()
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
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
                    payload, encoding, ts_ms = result
                    data = np.frombuffer(payload, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if encoding != "rgb8":
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._buffer.append((float(ts_ms), img))
            except Exception:
                pass
            time.sleep(0.005)

    def get_frame(self):
        """En son frame (geriye uyumluluk icin)."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1][1].copy()

    def get_frame_at(self, ts_ms: float, max_age_ms: float = 2000.0):
        """Verilen timestamp'e en yakin frame'i dondur. Eslesme yas farki
        max_age_ms'in disindaysa None."""
        with self._lock:
            if not self._buffer:
                return None
            best = min(self._buffer, key=lambda item: abs(item[0] - ts_ms))
        if abs(best[0] - ts_ms) > max_age_ms:
            return None
        return best[1].copy()


# --- DDS Mask Reader ---

class DDSMaskReader:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, SegmentationMask, qos=BEST_EFFORT_QOS)
        self._reader = DataReader(participant, self._topic, qos=BEST_EFFORT_QOS)
        self._mask = None
        self._mask_ts = 0.0
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
                        self._mask_ts = float(sample.timestamp)
            except Exception:
                pass
            time.sleep(0.005)

    def get_mask(self):
        with self._lock:
            return self._mask.copy() if self._mask is not None else None

    def get_mask_with_meta(self):
        """En son mask'i (mask, ts_ms) olarak dondur. SAM3 mask'in olustugu
        kaynak frame'in timestamp'ini tasiyor (ms cinsinden)."""
        with self._lock:
            if self._mask is None:
                return None
            return self._mask.copy(), self._mask_ts


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
            client_id="pipe_tracker_gui",
        ))  


# --- Main GUI ---

class PipeTrackerWindow(QMainWindow):
    def __init__(self, participant):
        super().__init__()
        self._participant = participant
        self.tracking = False
        self.last_cmd = {}

        self._heading_lock = threading.Lock()
        self._heading_deg = 0.0
        self._heading_ok = False
        self._vehicle = None
        self._heading_stop = threading.Event()

        self.processor = MaskProcessor(
            num_slices=8,
            slice_weights=[0.25, 0.20, 0.16, 0.12, 0.09, 0.07, 0.06, 0.05],
        )
        self.controller = PipeController()

        self.bottom_cam = DDSCameraReader(participant, "camera/bottom/frame")
        self.bottom_mask_reader = DDSMaskReader(participant, "sam3/bottom/segmentation_mask")
        self.cmd_pub = DDSCommandPublisher(participant, "embedded/control/stream_command")

        self.bottom_cam.start()
        self.bottom_mask_reader.start()

        self._init_ui()

        # tauv-client Vehicle ile attitude.yaw -> heading_deg beslemesi.
        # 180 donus reverse'i ve compute() icin gerekli.
        if Vehicle is None:
            self._log(f"UYARI: tauv-client import edilemedi ({_vehicle_import_error!r}); heading=0 sabit kalacak, REVERSE state takilir.")
        else:
            try:
                self._vehicle = Vehicle()
                threading.Thread(target=self._heading_loop, daemon=True, name="pipe-heading").start()
                self._log("Heading thread'i basladi (tauv-client Vehicle.attitude)")
            except Exception as exc:
                self._vehicle = None
                self._log(f"UYARI: Vehicle baslatilamadi: {exc}")

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

    def _heading_loop(self):
        """Vehicle.attitude'u poll ederek heading_deg'i guncelle.
        attitude.yaw radian; 0..360 dereceye normalize edilir."""
        while not self._heading_stop.is_set():
            try:
                att = self._vehicle.attitude
                hdg = math.degrees(att.yaw) % 360.0
                with self._heading_lock:
                    self._heading_deg = hdg
                    self._heading_ok = True
            except Exception:
                with self._heading_lock:
                    self._heading_ok = False
            time.sleep(0.05)

    @property
    def heading_deg(self) -> float:
        with self._heading_lock:
            return self._heading_deg

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
        self.heading_label = QLabel("hdg=---")
        self.heading_label.setStyleSheet("color: #7c8aff; font-size: 13px; font-weight: bold;")
        tl.addWidget(self.heading_label)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        tl.addWidget(self.info_label)
        top.addWidget(tg)
        lay.addLayout(top)

        # Cameras: tek panel -- alt kamera + senkron SAM3 mask overlay.
        # SAM3 sadece bottom kamerada calistigi icin front paneli yok.
        cam_row = QHBoxLayout()
        bottom_group = QGroupBox("Alt Kamera + Mask (senkron) + Algoritma")
        bv = QVBoxLayout(bottom_group)
        self.bottom_view = QLabel("DDS bekleniyor...")
        self.bottom_view.setMinimumSize(960, 600)
        self.bottom_view.setStyleSheet("background: #111; border: 1px solid #333;")
        self.bottom_view.setAlignment(Qt.AlignCenter)
        bv.addWidget(self.bottom_view)
        cam_row.addWidget(bottom_group)
        lay.addLayout(cam_row)

        # SAM3 Prompt (sadece bottom)
        sam3_group = QGroupBox("SAM3 Prompt (bottom)")
        sam3_lay = QHBoxLayout(sam3_group)
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
        prompt = self.bottom_prompt_input.text().strip()
        if not prompt:
            self.sam3_status_label.setText("Prompt bos")
            self._log("SAM3 prompt: bos")
            return
        try:
            resp = requests.post(
                f"{url}/api/prompt",
                json={"camera": "bottom", "prompt": prompt},
                timeout=5,
            )
            status = "bottom=OK" if resp.ok else f"bottom=HATA({resp.status_code})"
        except Exception:
            status = "bottom=BAGLANTI HATASI"
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
        # Heading label'i her tick guncelle -- REVERSE state'i icin
        # gercek heading geliyor mu acikca gor.
        with self._heading_lock:
            hdg_now = self._heading_deg
            hdg_ok = self._heading_ok
        if hdg_ok:
            self.heading_label.setText(f"hdg={hdg_now:5.1f}")
            self.heading_label.setStyleSheet("color: #7c8aff; font-size: 13px; font-weight: bold;")
        else:
            self.heading_label.setText("hdg=YOK")
            self.heading_label.setStyleSheet("color: #c62828; font-size: 13px; font-weight: bold;")

        # Mask-driven senkron overlay: SAM3 mask'i timestamp tasiyor (kaynak
        # frame'in zamani). O frame'i camera buffer'indan al, ikisini overlay
        # et -- bu sayede live frame + late mask lag'i ortadan kalkar.
        bottom_result = None
        sample = self.bottom_mask_reader.get_mask_with_meta()

        if sample is not None:
            bottom_mask, mask_ts = sample
            bottom = self.bottom_cam.get_frame_at(mask_ts, max_age_ms=2000.0)
            if bottom is None:
                # Buffer'da yeterince eski frame yok (ornegin GUI yeni acildi)
                # -- en son frame'le fallback yap.
                bottom = self.bottom_cam.get_frame()
            if bottom is not None:
                annotated_bottom, binary = self._overlay_mask(bottom, bottom_mask)
                bottom_result = self.processor.process(binary)
                self._draw_slices(annotated_bottom, bottom_result)
                self._show(self.bottom_view, annotated_bottom)
        else:
            bottom = self.bottom_cam.get_frame()
            if bottom is not None:
                self._show(self.bottom_view, bottom)

        if not self.tracking:
            return

        cmd = self.controller.compute(bottom_result, heading_deg=self.heading_deg)
        self.last_cmd = cmd
        st = self.controller.state
        self.state_label.setText(st)

        info_parts = []
        if st == PipeController.STATE_REVERSE:
            target = self.controller._reverse_target_hdg
            diff = (target - hdg_now + 180) % 360 - 180
            info_parts.append(f"180 DONUS: hedef={target:.0f} fark={diff:+.0f}")
        elif st == PipeController.STATE_REACQUIRE:
            elapsed = time.time() - self.controller._reacquire_start
            dur = self.controller.reacquire_duration_s
            info_parts.append(f"BORU ARANIYOR ({elapsed:.1f}s/{dur:.1f}s) fwd={self.controller.reacquire_forward_pwm}")
        elif bottom_result and bottom_result.error is not None:
            info_parts.append(f"err={bottom_result.error:+.2f}")
            info_parts.append(f"ang={bottom_result.pipe_angle_deg:+.0f}")
            cont = "DEVAM" if bottom_result.pipe_continues else "SON"
            info_parts.append(f"scan={bottom_result.scan_hit_count}({cont})")
        else:
            info_parts.append("boru yok")
        info_parts.append(f"CMD yaw={cmd['yaw']} fwd={cmd['forward']}")
        self.info_label.setText("  |  ".join(info_parts))

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
        self._heading_stop.set()
        self.bottom_cam.stop()
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
    print("Kamera: DDS FrameChunk (camera/bottom/frame, MJPEG)")
    print("Maske : DDS SegmentationMask (sam3/bottom/segmentation_mask)")
    print("Komut : DDS StreamCommand (embedded/control/stream_command)")
    print("Algo  : pipe_algorithm.py  -- mask senkron overlay (mask ts ile frame eslesmesi)")
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
