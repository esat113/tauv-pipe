#!/usr/bin/env python3
"""
TAUV Sekmeli GUI
Sekme 1: Arac Kontrolu (MAVLink direkt - Pixhawk)
Sekme 2: Boru Takip (DDS - kamera + SAM3 maske)

Kullanim:
  python3 tauv_gui.py
  python3 tauv_gui.py --mavlink udpin:127.0.0.1:14553
  python3 tauv_gui.py --mavlink /dev/ttyACM0 --baud 115200
"""

import sys
import os
import time
import json
import math
import threading
import glob as glob_module
import argparse

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QGroupBox, QLineEdit, QTextEdit,
    QTabWidget, QComboBox, QMessageBox, QFrame, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont

try:
    from pymavlink import mavutil
except ImportError:
    print("pymavlink kurulu degil! pip3 install pymavlink")
    sys.exit(1)

from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, uint8
from dataclasses import dataclass

from pipe_algorithm import MaskProcessor, PipeController, ProcessResult

# ═══════════════════════════════════════════════════════════
# Ortak stiller
# ═══════════════════════════════════════════════════════════

# Renk paleti (tutarli, sade)
C_BG       = "#1e1e2e"  # ana arkaplan
C_SURFACE  = "#2a2a3e"  # kart/grup arkaplan
C_BORDER   = "#3a3a5a"  # kenarlik
C_TEXT     = "#ddd"      # normal metin
C_DIM      = "#888"      # soluk metin
C_ACCENT   = "#7c8aff"  # vurgu (mavi-mor)
C_OK       = "#5cb85c"  # basarili / aktif (yesil)
C_WARN     = "#f0ad4e"  # uyari (turuncu)
C_DANGER   = "#d9534f"  # tehlike / hata (kirmizi)
C_NEUTRAL  = "#aaa"     # notr deger

DARK_STYLE = f"""
    QMainWindow {{ background-color: {C_BG}; }}
    QTabWidget::pane {{ border: 1px solid {C_BORDER}; }}
    QTabBar::tab {{ background: {C_SURFACE}; color: {C_TEXT}; padding: 10px 20px; border: 1px solid {C_BORDER}; }}
    QTabBar::tab:selected {{ background: {C_ACCENT}; color: white; }}
    QGroupBox {{ color: {C_ACCENT}; font-weight: bold; border: 1px solid {C_BORDER};
                border-radius: 5px; margin-top: 10px; padding-top: 10px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
    QLabel {{ color: {C_TEXT}; }}
    QPushButton {{ background: {C_SURFACE}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                  border-radius: 5px; padding: 8px 16px; font-weight: bold; }}
    QPushButton:hover {{ background: {C_BG}; border-color: {C_ACCENT}; }}
    QPushButton:pressed {{ background: {C_ACCENT}; }}
    QTextEdit {{ background: #161622; color: {C_DIM}; border: 1px solid {C_BORDER};
                font-family: monospace; font-size: 11px; }}
    QLineEdit {{ background: {C_SURFACE}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 3px; padding: 5px; }}
    QComboBox {{ background: {C_SURFACE}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 3px; padding: 5px; }}
    QComboBox QAbstractItemView {{ background: {C_SURFACE}; color: {C_TEXT};
                                   selection-background-color: {C_ACCENT}; }}
"""

# ═══════════════════════════════════════════════════════════
# DDS Tipleri (inline)
# ═══════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════
# DDS Yardimci Siniflar
# ═══════════════════════════════════════════════════════════

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


class DDSCameraReader:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, FrameChunk)
        self._reader = DataReader(participant, self._topic)
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
                    if img is not None:
                        if encoding != "rgb8":
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with self._lock:
                            self._frame = img
            except Exception:
                pass
            time.sleep(0.005)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


class DDSMaskReader:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, SegmentationMask)
        self._reader = DataReader(participant, self._topic)
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


class DDSCommandPublisher:
    def __init__(self, participant, topic_name):
        self._topic = Topic(participant, topic_name, StreamCommand)
        self._writer = DataWriter(participant, self._topic)

    def send(self, rc: dict):
        self._writer.write(StreamCommand(
            command_type="RC_OVERRIDE",
            command_data=json.dumps(rc),
            timestamp=int(time.time() * 1000),
        ))


# ═══════════════════════════════════════════════════════════
# SEKME 1: Arac Kontrolu (MAVLink)
# ═══════════════════════════════════════════════════════════

ARDUSUB_MODES = {
    'MANUAL': 19, 'STABILIZE': 0, 'ALT_HOLD': 2,
    'POSHOLD': 16, 'ACRO': 1, 'SURFACE': 9,
}
ARDUSUB_MODE_NAMES = {v: k for k, v in ARDUSUB_MODES.items()}

RC_CHANNELS = {
    'pitch': 0, 'roll': 1, 'throttle': 2, 'yaw': 3,
    'forward': 4, 'lateral': 5, 'camera_pan': 6, 'camera_tilt': 7,
}


def detect_serial_ports():
    patterns = ['/dev/ttyACM*', '/dev/ttyUSB*', '/dev/serial/by-id/*']
    ports = []
    for pattern in patterns:
        ports.extend(glob_module.glob(pattern))
    return sorted(set(ports)) or ['/dev/ttyACM0']


class ControlButton(QPushButton):
    def __init__(self, text, channel, direction, controller_ref, parent=None):
        super().__init__(text, parent)
        self.channel = channel
        self.direction = direction
        self._ctrl = controller_ref
        self.pwm_value = 1700
        self.neutral = 1500
        self.setMinimumSize(80, 60)
        self.setFont(QFont('Arial', 10, QFont.Bold))

    def set_pwm_values(self, pwm, neutral):
        self.pwm_value = pwm
        self.neutral = neutral

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        pwm = self.pwm_value if self.direction == 'positive' else self.neutral - (self.pwm_value - self.neutral)
        self._ctrl["set_channel"](self.channel, pwm)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._ctrl["set_channel"](self.channel, self.neutral)


class VehicleControlTab(QWidget):
    """Sekme 1: MAVLink ile dogrudan Pixhawk kontrolu."""

    def __init__(self, shared_state, parent=None):
        super().__init__(parent)
        self._shared = shared_state
        self.master = None
        self.connected = False
        self.armed = False
        self.current_mode = ""
        self.current_mode_id = -1
        self.pwm_values = [1500] * 8
        self.running = False
        self.boot_time = 0.0
        self.depth_m = 0.0
        self.heading_deg = 0.0
        self.active_keys = set()
        self.control_buttons = []
        self.rc_channels = [0] * 8
        self.servo_outputs = [0] * 8

        self._depth_target = None
        self._depth_tolerance = 0.3
        self._depth_target_active = False
        self._heading_target = None
        self._heading_active = False
        self._pending_mode = None
        self._pending_mode_id = None
        self._pending_mode_time = 0
        self._pending_mode_retry = 0
        self._pending_arm = None
        self._pending_arm_time = 0
        self._pending_arm_retry = 0
        self._lock = threading.Lock()

        self._ctrl_ref = {
            "set_channel": self._set_channel,
        }

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_ui)
        self._timer.start(100)

    def _set_channel(self, channel_name, pwm_value):
        if channel_name in RC_CHANNELS:
            with self._lock:
                self.pwm_values[RC_CHANNELS[channel_name]] = pwm_value

    def _stop_all(self):
        with self._lock:
            self.pwm_values = [1500] * 8
        self.active_keys.clear()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Baglanti
        conn_group = QGroupBox("Baglanti - Pixhawk (MAVLink)")
        conn_grid = QGridLayout(conn_group)
        self.conn_label = QLabel("Bagli Degil")
        self.conn_label.setStyleSheet(f"color: {C_DANGER}; font-size: 14px; font-weight: bold;")
        conn_grid.addWidget(self.conn_label, 0, 0, 1, 2)

        conn_grid.addWidget(QLabel("Tip:"), 1, 0)
        self.conn_type_combo = QComboBox()
        self.conn_type_combo.addItems(["MAVProxy UDP", "Seri Port", "Ozel Adres"])
        self.conn_type_combo.currentIndexChanged.connect(self._on_conn_type)
        conn_grid.addWidget(self.conn_type_combo, 1, 1, 1, 2)

        conn_grid.addWidget(QLabel("Port/Adres:"), 2, 0)
        self.conn_port_combo = QComboBox()
        self.conn_port_combo.setEditable(True)
        self.conn_port_combo.addItem("udpin:127.0.0.1:14553")
        conn_grid.addWidget(self.conn_port_combo, 2, 1, 1, 2)

        conn_grid.addWidget(QLabel("Baudrate:"), 3, 0)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '57600', '921600'])
        conn_grid.addWidget(self.baud_combo, 3, 1)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Baglan")
        self.btn_connect.setStyleSheet(f"background: {C_OK};")
        self.btn_connect.clicked.connect(self._on_connect)
        btn_row.addWidget(self.btn_connect)
        self.btn_disconnect = QPushButton("Kes")
        self.btn_disconnect.setStyleSheet(f"background: {C_DANGER};")
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        btn_row.addWidget(self.btn_disconnect)
        conn_grid.addLayout(btn_row, 4, 0, 1, 3)
        layout.addWidget(conn_group)

        # ARM + Mod
        arm_mode = QHBoxLayout()

        arm_group = QGroupBox("Arm")
        arm_lay = QHBoxLayout(arm_group)
        self.arm_label = QLabel("DISARMED")
        self.arm_label.setStyleSheet(f"color: {C_WARN}; font-size: 16px; font-weight: bold;")
        arm_lay.addWidget(self.arm_label)
        self.btn_arm = QPushButton("ARM")
        self.btn_arm.setStyleSheet(f"background: {C_OK};")
        self.btn_arm.setEnabled(False)
        self.btn_arm.clicked.connect(self._on_arm)
        arm_lay.addWidget(self.btn_arm)
        self.btn_disarm = QPushButton("DISARM")
        self.btn_disarm.setStyleSheet(f"background: {C_DANGER};")
        self.btn_disarm.setEnabled(False)
        self.btn_disarm.clicked.connect(self._on_disarm)
        arm_lay.addWidget(self.btn_disarm)
        arm_mode.addWidget(arm_group)

        mode_group = QGroupBox("Mod")
        mode_lay = QHBoxLayout(mode_group)
        self.mode_buttons = {}
        for name in ["MANUAL", "STABILIZE", "ALT_HOLD", "POSHOLD", "SURFACE"]:
            btn = QPushButton(name)
            btn.setMinimumWidth(80)
            btn.clicked.connect(lambda _, m=name: self._on_mode(m))
            self.mode_buttons[name] = btn
            mode_lay.addWidget(btn)
        self.mode_label = QLabel("?")
        self.mode_label.setStyleSheet(f"color: {C_ACCENT}; font-weight: bold;")
        mode_lay.addWidget(self.mode_label)
        arm_mode.addWidget(mode_group)
        layout.addLayout(arm_mode)

        # Derinlik + Heading
        info_row = QHBoxLayout()
        depth_group = QGroupBox("Derinlik")
        dg = QGridLayout(depth_group)
        self.depth_label = QLabel("0.00 m")
        self.depth_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 20px; font-weight: bold;")
        dg.addWidget(self.depth_label, 0, 0)
        dg.addWidget(QLabel("Hedef (m):"), 1, 0)
        self.depth_input = QLineEdit("1.5")
        self.depth_input.setMaximumWidth(60)
        dg.addWidget(self.depth_input, 1, 1)
        dg.addWidget(QLabel("Tolerans (m):"), 1, 2)
        self.depth_tolerance_input = QLineEdit("0.3")
        self.depth_tolerance_input.setMaximumWidth(50)
        dg.addWidget(self.depth_tolerance_input, 1, 3)
        self.btn_go_depth = QPushButton("Hedefe In")
        self.btn_go_depth.setStyleSheet(f"background: {C_ACCENT};")
        self.btn_go_depth.setEnabled(False)
        self.btn_go_depth.clicked.connect(self._on_go_depth)
        dg.addWidget(self.btn_go_depth, 2, 0)
        self.btn_cancel_depth = QPushButton("Iptal")
        self.btn_cancel_depth.setStyleSheet(f"background: {C_DANGER};")
        self.btn_cancel_depth.setEnabled(False)
        self.btn_cancel_depth.clicked.connect(self._on_cancel_depth)
        dg.addWidget(self.btn_cancel_depth, 2, 1)
        self.depth_status_label = QLabel("")
        self.depth_status_label.setStyleSheet(f"color: {C_DIM}; font-size: 10px;")
        dg.addWidget(self.depth_status_label, 3, 0, 1, 2)
        info_row.addWidget(depth_group)

        hdg_group = QGroupBox("Heading")
        hg = QGridLayout(hdg_group)
        self.heading_label = QLabel("---")
        self.heading_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 24px; font-weight: bold;")
        self.heading_label.setAlignment(Qt.AlignCenter)
        hg.addWidget(self.heading_label, 0, 0, 1, 6)
        hg.addWidget(QLabel("Hedef:"), 1, 0)
        self.heading_input = QLineEdit("0")
        self.heading_input.setMaximumWidth(50)
        hg.addWidget(self.heading_input, 1, 1)
        hg.addWidget(QLabel("Tol:"), 1, 2)
        self.heading_tolerance_input = QLineEdit("3")
        self.heading_tolerance_input.setMaximumWidth(30)
        hg.addWidget(self.heading_tolerance_input, 1, 3)
        self.btn_go_heading = QPushButton("Don")
        self.btn_go_heading.setStyleSheet(f"background: {C_ACCENT};")
        self.btn_go_heading.setEnabled(False)
        self.btn_go_heading.clicked.connect(self._on_go_heading)
        hg.addWidget(self.btn_go_heading, 1, 4)
        self.btn_cancel_heading = QPushButton("Iptal")
        self.btn_cancel_heading.setStyleSheet(f"background: {C_DANGER};")
        self.btn_cancel_heading.setEnabled(False)
        self.btn_cancel_heading.clicked.connect(self._on_cancel_heading)
        hg.addWidget(self.btn_cancel_heading, 1, 5)
        self.heading_status_label = QLabel("")
        self.heading_status_label.setStyleSheet(f"color: {C_DIM}; font-size: 10px;")
        hg.addWidget(self.heading_status_label, 2, 0, 1, 6)
        info_row.addWidget(hdg_group)
        layout.addLayout(info_row)

        # Motor kontrol
        motor_group = QGroupBox("Motor Kontrolu (Basili Tut / WASD+QE+RF)")
        motor_lay = QHBoxLayout(motor_group)

        for label_text, buttons_def in [
            ("Throttle", [("YUKARI (R)", 'throttle', 'positive', C_ACCENT),
                          ("ASAGI (F)", 'throttle', 'negative', C_ACCENT)]),
            ("Forward", [("ILERI (W)", 'forward', 'positive', C_OK),
                         ("GERI (S)", 'forward', 'negative', C_OK)]),
            ("Lateral", [("SOL (A)", 'lateral', 'negative', C_WARN),
                         ("SAG (D)", 'lateral', 'positive', C_WARN)]),
            ("Yaw", [("SOL (Q)", 'yaw', 'negative', C_ACCENT),
                     ("SAG (E)", 'yaw', 'positive', C_ACCENT)]),
        ]:
            frame = QFrame()
            fl = QVBoxLayout(frame)
            fl.addWidget(QLabel(label_text))
            for btn_text, channel, direction, color in buttons_def:
                btn = ControlButton(btn_text, channel, direction, self._ctrl_ref)
                btn.setStyleSheet(f"background: {color};")
                fl.addWidget(btn)
                self.control_buttons.append(btn)
            motor_lay.addWidget(frame)

        layout.addWidget(motor_group)

        # E-STOP
        self.btn_stop = QPushButton("TUM MOTORLARI DURDUR (SPACE)")
        self.btn_stop.setStyleSheet(f"background: {C_DANGER}; font-size: 16px; padding: 12px;")
        self.btn_stop.clicked.connect(self._stop_all)
        layout.addWidget(self.btn_stop)

        # Telemetri: RC Girdi + Servo Cikti + Gonderilen PWM
        telem_row = QHBoxLayout()
        ch_names = ['Pitch', 'Roll', 'Thr', 'Yaw', 'Fwd', 'Lat', 'Pan', 'Tilt']

        rc_group = QGroupBox("RC Girdi")
        rc_lay = QHBoxLayout(rc_group)
        self.rc_labels = []
        for name in ch_names:
            lbl = QLabel(f"{name}\n---")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px; font-weight: bold;")
            lbl.setMinimumWidth(45)
            rc_lay.addWidget(lbl)
            self.rc_labels.append(lbl)
        telem_row.addWidget(rc_group)

        servo_group = QGroupBox("Motor Cikti")
        servo_lay = QHBoxLayout(servo_group)
        self.servo_labels = []
        motor_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
        for name in motor_names:
            lbl = QLabel(f"{name}\n---")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {C_WARN}; font-size: 11px; font-weight: bold;")
            lbl.setMinimumWidth(45)
            servo_lay.addWidget(lbl)
            self.servo_labels.append(lbl)
        telem_row.addWidget(servo_group)

        sent_group = QGroupBox("Gonderilen PWM")
        sent_lay = QHBoxLayout(sent_group)
        self.channel_labels = []
        for name in ch_names:
            lbl = QLabel(f"{name}\n1500")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {C_NEUTRAL}; font-size: 11px; font-weight: bold;")
            lbl.setMinimumWidth(45)
            sent_lay.addWidget(lbl)
            self.channel_labels.append(lbl)
        telem_row.addWidget(sent_group)

        layout.addLayout(telem_row)

        # Log
        log_group = QGroupBox("MAVLink Log")
        log_lay = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        log_lay.addWidget(self.log_text)
        layout.addWidget(log_group)

    def _populate_ports(self):
        self.conn_port_combo.clear()
        self.conn_port_combo.addItems(detect_serial_ports())

    def _on_conn_type(self, index):
        if index == 0:
            self.conn_port_combo.clear()
            self.conn_port_combo.addItem("udpin:127.0.0.1:14553")
            self.baud_combo.setEnabled(False)
        elif index == 1:
            self._populate_ports()
            self.baud_combo.setEnabled(True)
        else:
            self.conn_port_combo.clear()
            self.baud_combo.setEnabled(True)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _on_connect(self):
        conn_str = self.conn_port_combo.currentText().strip()
        if not conn_str:
            return
        try:
            kwargs = {}
            if not conn_str.startswith('udp'):
                kwargs['baud'] = int(self.baud_combo.currentText())
            self._log(f"Baglaniliyor: {conn_str}...")
            self.master = mavutil.mavlink_connection(conn_str, **kwargs)
            self.boot_time = time.time()
            self.master.wait_heartbeat(timeout=15)
            self.connected = True
            self.running = True
            self.conn_label.setText(f"Bagli (SYS:{self.master.target_system})")
            self.conn_label.setStyleSheet(f"color: {C_OK}; font-size: 14px; font-weight: bold;")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.conn_type_combo.setEnabled(False)
            self.conn_port_combo.setEnabled(False)
            self.baud_combo.setEnabled(False)
            self.btn_arm.setEnabled(True)
            self.btn_disarm.setEnabled(True)
            self.btn_go_depth.setEnabled(True)
            self.btn_go_heading.setEnabled(True)
            threading.Thread(target=self._comm_loop, daemon=True).start()
            self._log("Baglanti kuruldu")
        except Exception as e:
            self._log(f"Baglanti hatasi: {e}")

    def _on_disconnect(self):
        self.running = False
        self.connected = False
        if self.master:
            try:
                neutral = [1500] * 8 + [65535] * 10
                for _ in range(5):
                    self.master.mav.rc_channels_override_send(
                        self.master.target_system, self.master.target_component, *neutral)
                    time.sleep(0.05)
            except:
                pass
            self.master.close()
            self.master = None
        self._pending_mode = None
        self._pending_arm = None
        self._depth_target = None
        self._depth_target_active = False
        self._heading_target = None
        self._heading_active = False
        self.conn_label.setText("Bagli Degil")
        self.conn_label.setStyleSheet(f"color: {C_DANGER}; font-size: 14px; font-weight: bold;")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.conn_type_combo.setEnabled(True)
        self.conn_port_combo.setEnabled(True)
        self._on_conn_type(self.conn_type_combo.currentIndex())
        self.btn_arm.setEnabled(False)
        self.btn_disarm.setEnabled(False)
        self.btn_go_depth.setEnabled(False)
        self.btn_cancel_depth.setEnabled(False)
        self.depth_status_label.setText("")
        self.btn_go_heading.setEnabled(False)
        self.btn_cancel_heading.setEnabled(False)
        self.heading_status_label.setText("")
        self._log("Baglanti kesildi")

    def _on_arm(self):
        if not self.master:
            return
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 21196, 0, 0, 0, 0, 0)
        self._pending_arm = True
        self._pending_arm_time = time.time()
        self._pending_arm_retry = time.time()
        self._log("ARM gonderildi...")

    def _on_disarm(self):
        if not self.master:
            return
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0, 0, 0, 0, 0)
        self._pending_arm = False
        self._pending_arm_time = time.time()
        self._pending_arm_retry = time.time()
        self._log("DISARM gonderildi...")

    def _on_mode(self, mode_name):
        if not self.master:
            return
        mode_id = ARDUSUB_MODES.get(mode_name)
        if mode_id is None:
            return
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        self._pending_mode = mode_name
        self._pending_mode_id = mode_id
        self._pending_mode_time = time.time()
        self._pending_mode_retry = time.time()
        self._log(f"Mod: {mode_name} gonderildi...")

    def _on_go_depth(self):
        if not self.master:
            return
        try:
            depth = float(self.depth_input.text())
            tolerance = float(self.depth_tolerance_input.text() or "0.3")
        except ValueError:
            self._log("Gecersiz deger")
            return
        if depth <= 0:
            self._log("Derinlik pozitif olmali")
            return
        self._depth_target = -abs(depth)
        self._depth_tolerance = tolerance
        self._depth_target_active = True
        self.btn_go_depth.setEnabled(False)
        self.btn_cancel_depth.setEnabled(True)
        if self.current_mode_id != ARDUSUB_MODES['ALT_HOLD']:
            self._on_mode('ALT_HOLD')
        self._send_depth_target()
        self._log(f"Depth hedef: {depth:.1f}m")

    def _on_cancel_depth(self):
        self._depth_target = None
        self._depth_target_active = False
        self.btn_go_depth.setEnabled(True)
        self.btn_cancel_depth.setEnabled(False)
        self.depth_status_label.setText("")
        self._log("Depth hedef iptal")

    def _on_go_heading(self):
        if not self.master:
            return
        try:
            target_deg = float(self.heading_input.text())
        except ValueError:
            self._log("Gecersiz heading degeri")
            return
        self._heading_target = target_deg % 360.0
        self._heading_active = True
        self.btn_go_heading.setEnabled(False)
        self.btn_cancel_heading.setEnabled(True)
        self._send_heading_target()
        self._log(f"Heading hedef: {self._heading_target:.0f} derece")

    def _on_cancel_heading(self):
        self._heading_target = None
        self._heading_active = False
        self.btn_go_heading.setEnabled(True)
        self.btn_cancel_heading.setEnabled(False)
        self.heading_status_label.setText("")
        self._log("Heading hedef iptal")

    def _send_heading_target(self):
        """Heading hedefi: eger depth de aktifse tek mesajda birlestirir, degilse set_attitude_target kullanir."""
        if not self.master or self._heading_target is None:
            return
        yaw_rad = math.radians(self._heading_target)

        if self._depth_target_active and self._depth_target is not None:
            self._send_depth_and_heading(yaw_rad)
        else:
            from pymavlink.quaternion import QuaternionBase
            self.master.mav.set_attitude_target_send(
                int(1e3 * (time.time() - self.boot_time)),
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE,
                QuaternionBase([math.radians(a) for a in (0, 0, self._heading_target)]),
                0, 0, 0, 0)

    def _send_depth_target(self):
        if not self.master or self._depth_target is None:
            return
        yaw_rad = 0.0
        yaw_mask = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        if self._heading_active and self._heading_target is not None:
            yaw_rad = math.radians(self._heading_target)
            yaw_mask = 0
        self.master.mav.set_position_target_global_int_send(
            int(1e3 * (time.time() - self.boot_time)),
            self.master.target_system, self.master.target_component,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            type_mask=(
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                yaw_mask |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            ),
            lat_int=0, lon_int=0, alt=self._depth_target,
            vx=0, vy=0, vz=0, afx=0, afy=0, afz=0, yaw=yaw_rad, yaw_rate=0)

    def _send_depth_and_heading(self, yaw_rad):
        """Depth + heading tek mesajda."""
        if not self.master or self._depth_target is None:
            return
        self.master.mav.set_position_target_global_int_send(
            int(1e3 * (time.time() - self.boot_time)),
            self.master.target_system, self.master.target_component,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            type_mask=(
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            ),
            lat_int=0, lon_int=0, alt=self._depth_target,
            vx=0, vy=0, vz=0, afx=0, afy=0, afz=0, yaw=yaw_rad, yaw_rate=0)

    def _comm_loop(self):
        last_hb = 0.0
        last_rc = 0.0
        last_depth_send = 0.0
        last_heading_send = 0.0

        while self.running and self.master:
            now = time.time()
            try:
                if now - last_hb >= 1.0:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    last_hb = now

                if now - last_rc >= 0.05:
                    with self._lock:
                        vals = self.pwm_values.copy()
                    rc_all = vals + [65535] * 10
                    self.master.mav.rc_channels_override_send(
                        self.master.target_system, self.master.target_component, *rc_all)
                    last_rc = now

                if (self._depth_target_active and self._depth_target is not None
                        and self.current_mode_id == ARDUSUB_MODES['ALT_HOLD']
                        and now - last_depth_send >= 1.0):
                    self._send_depth_target()
                    last_depth_send = now

                if (self._heading_active and self._heading_target is not None
                        and now - last_heading_send >= 0.1):
                    self._send_heading_target()
                    last_heading_send = now

                if (self._pending_mode is not None
                        and now - self._pending_mode_time < 5.0
                        and now - self._pending_mode_retry >= 1.0):
                    self.master.mav.set_mode_send(
                        self.master.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        self._pending_mode_id)
                    self._pending_mode_retry = now

                if (self._pending_mode is not None
                        and now - self._pending_mode_time >= 5.0):
                    self._pending_mode = None

                if (self._pending_arm is not None
                        and now - self._pending_arm_time < 5.0
                        and now - self._pending_arm_retry >= 1.0):
                    param1 = 1 if self._pending_arm else 0
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, param1, 21196, 0, 0, 0, 0, 0)
                    self._pending_arm_retry = now

                if (self._pending_arm is not None
                        and now - self._pending_arm_time >= 5.0):
                    self._pending_arm = None

                while True:
                    msg = self.master.recv_match(blocking=False)
                    if msg is None:
                        break
                    mtype = msg.get_type()

                    if mtype == 'HEARTBEAT':
                        if (msg.type != mavutil.mavlink.MAV_TYPE_GCS
                                and msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID):
                            new_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                            new_mode_id = msg.custom_mode

                            if self._pending_arm is not None:
                                if self._pending_arm and new_armed:
                                    self._pending_arm = None
                                elif not self._pending_arm and not new_armed:
                                    self._pending_arm = None

                            if (self._pending_mode is not None
                                    and new_mode_id == self._pending_mode_id):
                                self._pending_mode = None

                            self.armed = new_armed
                            self.current_mode_id = new_mode_id
                            self.current_mode = ARDUSUB_MODE_NAMES.get(new_mode_id, "?")

                    elif mtype == 'GLOBAL_POSITION_INT':
                        self.depth_m = abs(msg.relative_alt / 1000.0)
                        self.heading_deg = msg.hdg / 100.0
                        self._shared["heading_deg"] = self.heading_deg
                        self._shared["depth_m"] = self.depth_m

                        if self._depth_target_active and self._depth_target is not None:
                            target_m = abs(self._depth_target)
                            error = abs(self.depth_m - target_m)
                            if error <= self._depth_tolerance:
                                self._depth_target_active = False

                        

                    elif mtype == 'RC_CHANNELS':
                        self.rc_channels = [
                            msg.chan1_raw, msg.chan2_raw, msg.chan3_raw, msg.chan4_raw,
                            msg.chan5_raw, msg.chan6_raw, msg.chan7_raw, msg.chan8_raw,
                        ]
                    elif mtype == 'RC_CHANNELS_RAW':
                        self.rc_channels = [
                            msg.chan1_raw, msg.chan2_raw, msg.chan3_raw, msg.chan4_raw,
                            msg.chan5_raw, msg.chan6_raw, msg.chan7_raw, msg.chan8_raw,
                        ]
                    elif mtype == 'SERVO_OUTPUT_RAW':
                        self.servo_outputs = [
                            msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw,
                            msg.servo5_raw, msg.servo6_raw, msg.servo7_raw, msg.servo8_raw,
                        ]

                    self._shared["rc_channels"] = list(self.rc_channels)
                    self._shared["servo_outputs"] = list(self.servo_outputs)

            except Exception:
                pass
            time.sleep(0.01)

    def _update_ui(self):
        if self.connected:
            self.arm_label.setText("ARMED" if self.armed else "DISARMED")
            self.arm_label.setStyleSheet(f"color: {C_DANGER if self.armed else C_WARN}; font-size: 16px; font-weight: bold;")
            self.mode_label.setText(self.current_mode)
            for m, btn in self.mode_buttons.items():
                btn.setStyleSheet(f"background: {C_ACCENT};" if m == self.current_mode else "")
            self.depth_label.setText(f"{self.depth_m:.2f} m")
            self.heading_label.setText(f"{self.heading_deg:.1f}")

            if self._depth_target_active and self._depth_target is not None:
                target_m = abs(self._depth_target)
                error = abs(self.depth_m - target_m)
                self.depth_status_label.setText(f"Hedef: {target_m:.1f}m | Mevcut: {self.depth_m:.2f}m | Fark: {error:.2f}m")
                self.depth_status_label.setStyleSheet(f"color: {C_WARN}; font-size: 10px;")
            elif not self._depth_target_active and self.btn_cancel_depth.isEnabled():
                self.btn_go_depth.setEnabled(True)
                self.btn_cancel_depth.setEnabled(False)
                self.depth_status_label.setText(f"Hedefe ulasildi: {self.depth_m:.2f}m")
                self.depth_status_label.setStyleSheet(f"color: {C_OK}; font-size: 10px;")

            if self._heading_active and self._heading_target is not None:
                hdg_diff = (self._heading_target - self.heading_deg + 180) % 360 - 180
                try:
                    hdg_tol = float(self.heading_tolerance_input.text() or "3")
                except ValueError:
                    hdg_tol = 3.0
                color = C_OK if abs(hdg_diff) < hdg_tol else C_WARN
                status = "TUTUYOR" if abs(hdg_diff) < hdg_tol else "DONUYOR"
                self.heading_status_label.setText(f"{status} | Hedef: {self._heading_target:.0f} | Mevcut: {self.heading_deg:.0f} | Fark: {hdg_diff:+.0f}")
                self.heading_status_label.setStyleSheet(f"color: {color}; font-size: 10px;")

        ch_names = ['Pitch', 'Roll', 'Thr', 'Yaw', 'Fwd', 'Lat', 'Pan', 'Tilt']

        for i, lbl in enumerate(self.rc_labels):
            val = self.rc_channels[i] if i < len(self.rc_channels) else 0
            lbl.setText(f"{ch_names[i]}\n{val}")
            color = C_OK if val and abs(val - 1500) > 20 else C_ACCENT
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

        motor_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
        for i, lbl in enumerate(self.servo_labels):
            val = self.servo_outputs[i] if i < len(self.servo_outputs) else 0
            lbl.setText(f"{motor_names[i]}\n{val}")
            color = C_OK if val and abs(val - 1500) > 20 else C_WARN
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

        for i, lbl in enumerate(self.channel_labels):
            val = self.pwm_values[i]
            lbl.setText(f"{ch_names[i]}\n{val}")
            color = C_OK if val != 1500 else C_NEUTRAL
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

        for btn in self.control_buttons:
            btn.set_pwm_values(1700, 1500)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key_map = {
            Qt.Key_W: ('forward', 'positive'), Qt.Key_S: ('forward', 'negative'),
            Qt.Key_A: ('lateral', 'negative'), Qt.Key_D: ('lateral', 'positive'),
            Qt.Key_Q: ('yaw', 'negative'), Qt.Key_E: ('yaw', 'positive'),
            Qt.Key_R: ('throttle', 'positive'), Qt.Key_F: ('throttle', 'negative'),
        }
        if event.key() == Qt.Key_Space:
            self._stop_all()
            return
        if event.key() in key_map and event.key() not in self.active_keys:
            self.active_keys.add(event.key())
            ch, direction = key_map[event.key()]
            value = 1700 if direction == 'positive' else 1300
            self._set_channel(ch, value)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key_map = {
            Qt.Key_W: 'forward', Qt.Key_S: 'forward',
            Qt.Key_A: 'lateral', Qt.Key_D: 'lateral',
            Qt.Key_Q: 'yaw', Qt.Key_E: 'yaw',
            Qt.Key_R: 'throttle', Qt.Key_F: 'throttle',
        }
        if event.key() in key_map:
            self.active_keys.discard(event.key())
            self._set_channel(key_map[event.key()], 1500)

    def cleanup(self):
        self._on_disconnect()


# ═══════════════════════════════════════════════════════════
# SEKME 2: Boru Takip (DDS)
# ═══════════════════════════════════════════════════════════

class PipeTrackingTab(QWidget):
    """Sekme 2: DDS kamera + SAM3 mask ile boru takip."""

    def __init__(self, participant, shared_state, parent=None):
        super().__init__(parent)
        self._shared = shared_state
        self.tracking = False
        self.last_cmd = {}

        self.processor = MaskProcessor(
            num_slices=8,
            slice_weights=[0.25, 0.20, 0.16, 0.12, 0.09, 0.07, 0.06, 0.05],
        )
        self.controller = PipeController()

        self.front_cam = DDSCameraReader(participant, "camera/front/frame/rgb8")
        self.bottom_cam = DDSCameraReader(participant, "camera/bottom/frame/rgb8")
        self.front_mask_reader = DDSMaskReader(participant, "sam3/front/segmentation_mask")
        self.bottom_mask_reader = DDSMaskReader(participant, "sam3/bottom/segmentation_mask")
        self.cmd_pub = DDSCommandPublisher(participant, "embedded/control/stream_command")

        self.front_cam.start()
        self.bottom_cam.start()
        self.front_mask_reader.start()
        self.bottom_mask_reader.start()

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(4)

        # Gorev
        top = QHBoxLayout()
        tg = QGroupBox("Boru Takip Gorevi")
        tl = QHBoxLayout(tg)
        self.btn_start = QPushButton("BASLAT")
        self.btn_start.setStyleSheet(f"background: {C_ACCENT}; font-size: 15px; padding: 10px 40px;")
        self.btn_start.clicked.connect(self._on_start)
        tl.addWidget(self.btn_start)
        self.btn_stop = QPushButton("DURDUR")
        self.btn_stop.setStyleSheet(f"background: {C_DANGER}; font-size: 15px; padding: 10px 40px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        tl.addWidget(self.btn_stop)
        self.state_label = QLabel("IDLE")
        self.state_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 20px; font-weight: bold;")
        tl.addWidget(self.state_label)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"color: {C_DIM}; font-size: 11px;")
        tl.addWidget(self.info_label)
        top.addWidget(tg)
        lay.addLayout(top)

        # Kameralar
        cam_row = QHBoxLayout()
        for name, attr in [("On Kamera + SAM3", "front_view"), ("Alt Kamera + Algoritma", "bottom_view")]:
            g = QGroupBox(name)
            v = QVBoxLayout(g)
            lbl = QLabel("DDS bekleniyor...")
            lbl.setFixedSize(560, 360)
            lbl.setStyleSheet(f"background: #161622; border: 1px solid {C_BORDER};")
            lbl.setAlignment(Qt.AlignCenter)
            setattr(self, attr, lbl)
            v.addWidget(lbl)
            cam_row.addWidget(g)
        lay.addLayout(cam_row)

        # SAM3 Prompt
        sam3_group = QGroupBox("SAM3 Prompt")
        sam3_lay = QHBoxLayout(sam3_group)
        sam3_lay.addWidget(QLabel("Front:"))
        self.front_prompt = QLineEdit("pipe")
        self.front_prompt.setMaximumWidth(100)
        sam3_lay.addWidget(self.front_prompt)
        sam3_lay.addWidget(QLabel("Bottom:"))
        self.bottom_prompt = QLineEdit("pipe")
        self.bottom_prompt.setMaximumWidth(100)
        sam3_lay.addWidget(self.bottom_prompt)
        sam3_lay.addWidget(QLabel("URL:"))
        self.sam3_url = QLineEdit("http://localhost:5003")
        self.sam3_url.setMaximumWidth(180)
        sam3_lay.addWidget(self.sam3_url)
        btn = QPushButton("Gonder")
        btn.setStyleSheet(f"background: {C_ACCENT};")
        btn.clicked.connect(self._send_prompts)
        sam3_lay.addWidget(btn)
        self.sam3_status = QLabel("")
        self.sam3_status.setStyleSheet(f"color: {C_DIM}; font-size: 11px;")
        sam3_lay.addWidget(self.sam3_status)
        sam3_lay.addStretch()
        lay.addWidget(sam3_group)

        # Tuning
        tune_group = QGroupBox("Algoritma Tuning")
        tg_lay = QGridLayout(tune_group)
        self.tune_inputs = {}
        for i, (key, label, default) in enumerate([
            ("kp_yaw", "Kp", "150"), ("ki_yaw", "Ki", "200"),
            ("forward_pwm", "Fwd PWM", "200"), ("max_yaw_pwm", "Max Yaw", "200"),
            ("ema_alpha", "EMA", "0.6"), ("turn_curvature_thresh", "Viraj", "0.03"),
            ("turn_forward_pwm", "Viraj Fwd", "150"), ("turn_yaw_boost", "Boost", "3.0"),
            ("coast_timeout", "Coast T", "1.5"), ("coast_yaw_decay", "Decay", "0.55"),
        ]):
            row, col = i // 5, (i % 5) * 2
            tg_lay.addWidget(QLabel(label), row * 2, col)
            inp = QLineEdit(default)
            inp.setMaximumWidth(60)
            tg_lay.addWidget(inp, row * 2, col + 1)
            self.tune_inputs[key] = inp
        btn_tune = QPushButton("Uygula")
        btn_tune.setStyleSheet(f"background: {C_ACCENT};")
        btn_tune.clicked.connect(self._apply_tune)
        tg_lay.addWidget(btn_tune, 2, 8, 1, 2)
        lay.addWidget(tune_group)

        # RC + Servo telemetri (MAVLink'ten, shared_state uzerinden)
        telem_row = QHBoxLayout()
        ch_names = ['Pitch', 'Roll', 'Thr', 'Yaw', 'Fwd', 'Lat', 'Pan', 'Tilt']

        rc_group = QGroupBox("RC Girdi")
        rc_lay = QHBoxLayout(rc_group)
        self.pipe_rc_labels = []
        for name in ch_names:
            lbl = QLabel(f"{name}\n---")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px; font-weight: bold;")
            lbl.setMinimumWidth(42)
            rc_lay.addWidget(lbl)
            self.pipe_rc_labels.append(lbl)
        telem_row.addWidget(rc_group)

        servo_group = QGroupBox("Motor Cikti")
        servo_lay = QHBoxLayout(servo_group)
        self.pipe_servo_labels = []
        for name in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']:
            lbl = QLabel(f"{name}\n---")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {C_WARN}; font-size: 11px; font-weight: bold;")
            lbl.setMinimumWidth(42)
            servo_lay.addWidget(lbl)
            self.pipe_servo_labels.append(lbl)
        telem_row.addWidget(servo_group)

        lay.addLayout(telem_row)

        # Log
        lg = QGroupBox("Pipe Log")
        ll = QVBoxLayout(lg)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        ll.addWidget(self.log_text)
        lay.addWidget(lg)

    def _send_prompts(self):
        import requests
        url = self.sam3_url.text().strip().rstrip("/")
        results = []
        for camera, inp in [("front", self.front_prompt), ("bottom", self.bottom_prompt)]:
            prompt = inp.text().strip()
            if not prompt:
                continue
            try:
                resp = requests.post(f"{url}/api/prompt", json={"camera": camera, "prompt": prompt}, timeout=5)
                results.append(f"{camera}={'OK' if resp.ok else 'HATA'}")
            except Exception:
                results.append(f"{camera}=HATA")
        self.sam3_status.setText("  ".join(results))

    def _apply_tune(self):
        try:
            s = self.tune_inputs
            self.controller.update_params(
                kp_yaw=float(s["kp_yaw"].text()), ki_yaw=float(s["ki_yaw"].text()),
                forward_pwm=int(s["forward_pwm"].text()), max_yaw_pwm=int(s["max_yaw_pwm"].text()),
                ema_alpha=float(s["ema_alpha"].text()),
                turn_curvature_thresh=float(s["turn_curvature_thresh"].text()),
                turn_forward_pwm=int(s["turn_forward_pwm"].text()),
                turn_yaw_boost=float(s["turn_yaw_boost"].text()),
                coast_timeout=float(s["coast_timeout"].text()),
                coast_yaw_decay=float(s["coast_yaw_decay"].text()),
            )
            self._log("Tuning uygulandi")
        except ValueError as e:
            self._log(f"Hata: {e}")

    def _overlay_mask(self, frame, raw_mask):
        h_cam, w_cam = frame.shape[:2]
        h_mask, w_mask = raw_mask.shape[:2]
        binary = cv2.resize(raw_mask, (w_cam, h_cam), interpolation=cv2.INTER_NEAREST) if (h_mask, w_mask) != (h_cam, w_cam) else raw_mask.copy()
        _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)
        annotated = frame.copy()
        overlay = annotated.copy()
        overlay[binary > 0] = [0, 255, 0]
        return cv2.addWeighted(annotated, 0.65, overlay, 0.35, 0), binary

    def _tick(self):
        ch_names = ['Pitch', 'Roll', 'Thr', 'Yaw', 'Fwd', 'Lat', 'Pan', 'Tilt']
        rc = self._shared.get("rc_channels", [0] * 8)
        servo = self._shared.get("servo_outputs", [0] * 8)
        for i, lbl in enumerate(self.pipe_rc_labels):
            val = rc[i] if i < len(rc) else 0
            lbl.setText(f"{ch_names[i]}\n{val}")
            lbl.setStyleSheet(f"color: {C_OK if val and abs(val - 1500) > 20 else C_ACCENT}; font-size: 11px; font-weight: bold;")
        motor_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
        for i, lbl in enumerate(self.pipe_servo_labels):
            val = servo[i] if i < len(servo) else 0
            lbl.setText(f"{motor_names[i]}\n{val}")
            lbl.setStyleSheet(f"color: {C_OK if val and abs(val - 1500) > 20 else C_WARN}; font-size: 11px; font-weight: bold;")

        front = self.front_cam.get_frame()
        front_mask = self.front_mask_reader.get_mask()
        if front is not None and front_mask is not None:
            ann, _ = self._overlay_mask(front, front_mask)
            self._show(self.front_view, ann)
        elif front is not None:
            self._show(self.front_view, front)

        bottom = self.bottom_cam.get_frame()
        bottom_mask = self.bottom_mask_reader.get_mask()
        bottom_result = None
        if bottom is not None and bottom_mask is not None:
            ann, binary = self._overlay_mask(bottom, bottom_mask)
            bottom_result = self.processor.process(binary)
            self._draw_slices(ann, bottom_result)
            self._show(self.bottom_view, ann)
        elif bottom is not None:
            self._show(self.bottom_view, bottom)

        if not self.tracking:
            return

        heading = self._shared.get("heading_deg", 0.0)
        cmd = self.controller.compute(bottom_result, heading_deg=heading)
        self.last_cmd = cmd
        st = self.controller.state
        self.state_label.setText(st)

        if st == PipeController.STATE_COMPLETE:
            self.state_label.setText("GOREV TAMAMLANDI")
            self.state_label.setStyleSheet(f"color: {C_OK}; font-size: 20px; font-weight: bold;")
            self.tracking = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("GOREV TAMAMLANDI")
            self.cmd_pub.send(cmd)
            return

        info = [f"Gecis:{self.controller.pass_count}/2"]
        if st == PipeController.STATE_REVERSE:
            diff = (self.controller._reverse_target_hdg - heading + 180) % 360 - 180
            info.append(f"180DON fark={diff:+.0f}")
        elif bottom_result and bottom_result.error is not None:
            cont = "DEVAM" if bottom_result.pipe_continues else "SON"
            info.append(f"err={bottom_result.error:+.2f} scan={bottom_result.scan_hit_count}({cont})")
        info.append(f"yaw={cmd['yaw']} fwd={cmd['forward']} hdg={heading:.0f}")
        self.info_label.setText("  ".join(info))
        self.cmd_pub.send(cmd)

    def _draw_slices(self, frame, result):
        if result is None or result.error is None or len(result.slice_centroids) < 2:
            return
        h, w = frame.shape[:2]
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        num = result.num_slices
        slice_h = h // num
        for i in range(1, num):
            cv2.line(bgr, (0, i * slice_h), (w, i * slice_h), (100, 100, 100), 1)
        for idx, cx, cy, area in result.slice_centroids:
            wt = self.processor.slice_weights[idx] if idx < len(self.processor.slice_weights) else 0
            r = max(4, int(wt * 40))
            cv2.circle(bgr, (int(cx), cy), r, (255, 255, 0), -1)
            cv2.circle(bgr, (int(cx), cy), r, (0, 255, 255), 2)
            cv2.putText(bgr, f"w={wt:.2f}", (int(cx) + r + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.line(bgr, (int(result.center_x), 0), (int(result.center_x), h), (0, 0, 255), 1)
        cv2.line(bgr, (int(result.weighted_cx), 0), (int(result.weighted_cx), h), (0, 255, 0), 2)
        sorted_c = sorted(result.slice_centroids, key=lambda c: c[0])
        pts = [(int(c[1]), c[2]) for c in sorted_c]
        for j in range(len(pts) - 1):
            cv2.line(bgr, pts[j], pts[j + 1], (255, 0, 255), 2)
        for x1, y1, x2, y2, hit in result.scan_rays:
            color = (255, 255, 255) if hit else (0, 0, 255)
            cv2.line(bgr, (x1, y1), (x2, y2), color, 1)
        cont_str = "DEVAM" if result.pipe_continues else "SON"
        cv2.putText(bgr, f"err={result.error:+.2f} ang={result.pipe_angle_deg:+.0f} {cont_str}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
        frame[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _show(self, label, frame_rgb):
        h, w = frame_rgb.shape[:2]
        dw, dh = label.width(), label.height()
        sc = min(dw / w, dh / h)
        nw, nh = int(w * sc), int(h * sc)
        r = cv2.resize(frame_rgb, (nw, nh))
        qi = QImage(r.data, nw, nh, nw * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi))

    def _on_start(self):
        self.controller.reset()
        self.tracking = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.state_label.setText("FOLLOW")
        self.state_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 20px; font-weight: bold;")
        self._log("Takip baslatildi")

    def _on_stop(self):
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

    def cleanup(self):
        self.tracking = False
        self.front_cam.stop()
        self.bottom_cam.stop()
        self.front_mask_reader.stop()
        self.bottom_mask_reader.stop()


# ═══════════════════════════════════════════════════════════
# ANA PENCERE
# ═══════════════════════════════════════════════════════════

class TauvMainWindow(QMainWindow):
    def __init__(self, dds_participant):
        super().__init__()
        self.setWindowTitle("TAUV Control Suite")
        self.setMinimumSize(1250, 950)
        self.setStyleSheet(DARK_STYLE)

        self._shared_state = {"heading_deg": 0.0}

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._vehicle_tab = VehicleControlTab(self._shared_state)
        self._pipe_tab = PipeTrackingTab(dds_participant, self._shared_state)

        self._tabs.addTab(self._vehicle_tab, "Arac Kontrolu")
        self._tabs.addTab(self._pipe_tab, "Boru Takip")

    def keyPressEvent(self, event):
        if self._tabs.currentIndex() == 0:
            self._vehicle_tab.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._tabs.currentIndex() == 0:
            self._vehicle_tab.keyReleaseEvent(event)
        else:
            super().keyReleaseEvent(event)

    def closeEvent(self, event):
        self._vehicle_tab.cleanup()
        self._pipe_tab.cleanup()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="TAUV Control Suite")
    parser.add_argument("--domain-id", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("  TAUV CONTROL SUITE")
    print("  Sekme 1: Arac Kontrolu (MAVLink)")
    print("  Sekme 2: Boru Takip (DDS)")
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

    w = TauvMainWindow(participant)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
