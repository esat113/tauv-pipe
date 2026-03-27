#!/usr/bin/env python3
"""
TAUV Sekmeli GUI v2 -- Tamamen DDS (Vehicle API)
Sekme 1: Arac Kontrolu (DDS uzerinden bridge'e)
Sekme 2: Boru Takip (DDS kamera + SAM3 maske)

MAVLink yok, bridge ile cakilma yok.

Kullanim:
  pip install -e ../tauv-client
  export CYCLONEDDS_URI=file:///tmp/dds_gui.xml
  python3 tauv_gui2.py
"""

import sys
import os
import time
import json
import math
import threading
import argparse

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QGroupBox, QLineEdit, QTextEdit,
    QTabWidget, QFrame,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont

from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from cyclonedds.qos import Qos, Policy
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, uint8
from dataclasses import dataclass

from pipe_algorithm import MaskProcessor, PipeController, ProcessResult

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tauv-client", "src"))
from tauv_client import Vehicle
from tauv_client.core.errors import SensorTimeoutError

C_BG       = "#1e1e2e"
C_SURFACE  = "#2a2a3e"
C_BORDER   = "#3a3a5a"
C_TEXT     = "#ddd"
C_DIM      = "#888"
C_ACCENT   = "#7c8aff"
C_OK       = "#5cb85c"
C_WARN     = "#f0ad4e"
C_DANGER   = "#d9534f"
C_NEUTRAL  = "#aaa"

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
"""

BEST_EFFORT_QOS = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(depth=1))

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


class FrameAssembler:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()
    def reset(self):
        self.expected_chunks = None
        self.received_count = 0
        self.width = self.height = 0
        self.encoding = ""
        self.buffers = []
    def push(self, chunk):
        with self._lock:
            if chunk.chunk_id_in_frame == 0:
                total = int(chunk.total_chunks_in_frame)
                if total <= 0:
                    self.reset(); return None
                self.expected_chunks = total
                self.received_count = 0
                self.width = int(chunk.width)
                self.height = int(chunk.height)
                self.encoding = str(chunk.encoding)
                self.buffers = [None] * total
            if self.expected_chunks is None: return None
            if int(chunk.total_chunks_in_frame) != self.expected_chunks: self.reset(); return None
            idx = int(chunk.chunk_id_in_frame)
            if idx < 0 or idx >= self.expected_chunks: self.reset(); return None
            if self.buffers[idx] is None:
                self.buffers[idx] = bytes(chunk.chunk_buffer)
                self.received_count += 1
            if self.received_count != self.expected_chunks: return None
            if any(p is None for p in self.buffers): self.reset(); return None
            payload = b"".join(self.buffers)
            enc = self.encoding
            self.reset()
            return payload, enc


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
                for chunk in self._reader.take(32):
                    result = self._assembler.push(chunk)
                    if result is None: continue
                    payload, enc = result
                    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        if enc != "rgb8": img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with self._lock: self._frame = img
            except Exception: pass
            time.sleep(0.005)
    def get_frame(self):
        with self._lock: return self._frame.copy() if self._frame is not None else None


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
                for s in self._reader.take(32):
                    raw = np.frombuffer(bytes(s.mask_data), np.uint8).reshape(s.height, s.width)
                    with self._lock: self._mask = raw
            except Exception: pass
            time.sleep(0.005)
    def get_mask(self):
        with self._lock: return self._mask.copy() if self._mask is not None else None


# ═══════════════════════════════════════════════════════════
# SEKME 1: Arac Kontrolu (Vehicle API / DDS)
# ═══════════════════════════════════════════════════════════

class VehicleControlTab(QWidget):
    def __init__(self, vehicle: Vehicle, shared_state: dict, parent=None):
        super().__init__(parent)
        self._v = vehicle
        self._shared = shared_state
        self.active_keys = set()
        self._rc_active = True
        self._heading_active = False
        self._heading_target = None

        self._telem = {
            "armed": False, "mode": "?", "depth": 0.0, "heading": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw_rad": 0.0,
        }
        self._telem_lock = threading.Lock()
        self._stop = threading.Event()

        threading.Thread(target=self._sensor_loop, daemon=True, name="v-sensors").start()

        self._build_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_ui)
        self._timer.start(100)

    def _sensor_loop(self):
        while not self._stop.is_set():
            try:
                vm = self._v.vehicle_mode
                with self._telem_lock:
                    self._telem["armed"] = vm.armed
                    self._telem["mode"] = vm.mode
            except Exception: pass
            try:
                d = self._v.depth
                with self._telem_lock:
                    self._telem["depth"] = d.depth
            except Exception: pass
            try:
                att = self._v.attitude
                hdg = math.degrees(att.yaw) % 360
                with self._telem_lock:
                    self._telem["heading"] = hdg
                    self._telem["roll"] = att.roll
                    self._telem["pitch"] = att.pitch
                    self._telem["yaw_rad"] = att.yaw
                self._shared["heading_deg"] = hdg
            except Exception: pass

            if self._heading_active and self._heading_target is not None:
                try:
                    self._v.set_target_attitude(yaw=self._heading_target)
                except Exception: pass

            time.sleep(0.05)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        status_group = QGroupBox("Durum (DDS)")
        sg = QHBoxLayout(status_group)
        self.conn_label = QLabel("DDS Aktif")
        self.conn_label.setStyleSheet(f"color: {C_OK}; font-size: 14px; font-weight: bold;")
        sg.addWidget(self.conn_label)
        self.arm_label = QLabel("DISARMED")
        self.arm_label.setStyleSheet(f"color: {C_WARN}; font-size: 16px; font-weight: bold;")
        sg.addWidget(self.arm_label)
        self.mode_label = QLabel("?")
        self.mode_label.setStyleSheet(f"color: {C_ACCENT}; font-weight: bold; font-size: 14px;")
        sg.addWidget(self.mode_label)
        layout.addWidget(status_group)

        arm_mode = QHBoxLayout()
        arm_group = QGroupBox("Arm")
        al = QHBoxLayout(arm_group)
        btn_arm = QPushButton("ARM")
        btn_arm.setStyleSheet(f"background: {C_OK};")
        btn_arm.clicked.connect(lambda: self._do_cmd(self._v.arm, force=True))
        al.addWidget(btn_arm)
        btn_disarm = QPushButton("DISARM")
        btn_disarm.setStyleSheet(f"background: {C_DANGER};")
        btn_disarm.clicked.connect(lambda: self._do_cmd(self._v.disarm, force=True))
        al.addWidget(btn_disarm)
        arm_mode.addWidget(arm_group)

        mode_group = QGroupBox("Mod")
        ml = QHBoxLayout(mode_group)
        for name in ["manuel", "stabilize", "hold_altitude"]:
            btn = QPushButton(name.upper())
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, m=name: self._do_cmd(self._v.set_mode, m))
            ml.addWidget(btn)
        arm_mode.addWidget(mode_group)
        layout.addLayout(arm_mode)

        info_row = QHBoxLayout()
        depth_group = QGroupBox("Derinlik")
        dg = QGridLayout(depth_group)
        self.depth_label = QLabel("0.00 m")
        self.depth_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 20px; font-weight: bold;")
        dg.addWidget(self.depth_label, 0, 0)
        dg.addWidget(QLabel("Hedef:"), 1, 0)
        self.depth_input = QLineEdit("1.5")
        self.depth_input.setMaximumWidth(60)
        dg.addWidget(self.depth_input, 1, 1)
        btn_depth = QPushButton("In")
        btn_depth.setStyleSheet(f"background: {C_ACCENT};")
        btn_depth.clicked.connect(self._on_go_depth)
        dg.addWidget(btn_depth, 1, 2)
        info_row.addWidget(depth_group)

        hdg_group = QGroupBox("Heading")
        hg = QGridLayout(hdg_group)
        self.heading_label = QLabel("---")
        self.heading_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 24px; font-weight: bold;")
        self.heading_label.setAlignment(Qt.AlignCenter)
        hg.addWidget(self.heading_label, 0, 0, 1, 3)
        hg.addWidget(QLabel("Hedef:"), 1, 0)
        self.heading_input = QLineEdit("0")
        self.heading_input.setMaximumWidth(50)
        hg.addWidget(self.heading_input, 1, 1)
        btn_hdg = QPushButton("Don")
        btn_hdg.setStyleSheet(f"background: {C_ACCENT};")
        btn_hdg.clicked.connect(self._on_go_heading)
        hg.addWidget(btn_hdg, 1, 2)
        btn_hdg_cancel = QPushButton("Iptal")
        btn_hdg_cancel.setStyleSheet(f"background: {C_DANGER};")
        btn_hdg_cancel.clicked.connect(self._on_cancel_heading)
        hg.addWidget(btn_hdg_cancel, 1, 3)
        self.heading_status = QLabel("")
        self.heading_status.setStyleSheet(f"color: {C_DIM}; font-size: 10px;")
        hg.addWidget(self.heading_status, 2, 0, 1, 4)
        info_row.addWidget(hdg_group)
        layout.addLayout(info_row)

        motor_group = QGroupBox("Motor (WASD+QE+RF, SPACE=durdur)")
        motor_lay = QHBoxLayout(motor_group)
        for label, keys in [
            ("Thr: R/F", [("R UP", Qt.Key_R), ("F DN", Qt.Key_F)]),
            ("Fwd: W/S", [("W FWD", Qt.Key_W), ("S BCK", Qt.Key_S)]),
            ("Lat: A/D", [("A L", Qt.Key_A), ("D R", Qt.Key_D)]),
            ("Yaw: Q/E", [("Q L", Qt.Key_Q), ("E R", Qt.Key_E)]),
        ]:
            f = QFrame()
            fl = QVBoxLayout(f)
            fl.addWidget(QLabel(label))
            motor_lay.addWidget(f)
        layout.addWidget(motor_group)

        estop_row = QHBoxLayout()
        btn_stop = QPushButton("TUM MOTORLARI DURDUR (SPACE)")
        btn_stop.setStyleSheet(f"background: {C_DANGER}; font-size: 16px; padding: 12px;")
        btn_stop.clicked.connect(self._stop_motors)
        estop_row.addWidget(btn_stop)
        self.btn_rc_toggle = QPushButton("RC: AKTIF")
        self.btn_rc_toggle.setStyleSheet(f"background: {C_OK}; font-size: 13px; padding: 12px;")
        self.btn_rc_toggle.clicked.connect(self._toggle_rc)
        estop_row.addWidget(self.btn_rc_toggle)
        layout.addLayout(estop_row)

        lg = QGroupBox("Log")
        ll = QVBoxLayout(lg)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        ll.addWidget(self.log_text)
        layout.addWidget(lg)

    def _do_cmd(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            self._log(f"OK: {fn.__name__}")
        except Exception as e:
            self._log(f"HATA: {fn.__name__} -> {e}")

    def _on_go_depth(self):
        try:
            depth = float(self.depth_input.text())
        except ValueError:
            return
        self._do_cmd(self._v.set_depth_target, depth)

    def _on_go_heading(self):
        try:
            target = float(self.heading_input.text()) % 360
        except ValueError:
            return
        self._heading_target = target
        self._heading_active = True
        self._log(f"Heading hedef: {target:.0f}")

    def _on_cancel_heading(self):
        self._heading_active = False
        self._heading_target = None
        self.heading_status.setText("")
        self._log("Heading iptal")

    def _stop_motors(self):
        self._v.motor_rc_stop()
        self.active_keys.clear()

    def _toggle_rc(self):
        self._rc_active = not self._rc_active
        if self._rc_active:
            self.btn_rc_toggle.setText("RC: AKTIF")
            self.btn_rc_toggle.setStyleSheet(f"background: {C_OK}; font-size: 13px; padding: 12px;")
        else:
            self._stop_motors()
            self.btn_rc_toggle.setText("RC: KAPALI")
            self.btn_rc_toggle.setStyleSheet(f"background: {C_DIM}; font-size: 13px; padding: 12px;")

    KEY_MAP = {
        Qt.Key_W: ("forward", 1700), Qt.Key_S: ("forward", 1300),
        Qt.Key_A: ("lateral", 1300), Qt.Key_D: ("lateral", 1700),
        Qt.Key_Q: ("yaw", 1300), Qt.Key_E: ("yaw", 1700),
        Qt.Key_R: ("throttle", 1700), Qt.Key_F: ("throttle", 1300),
    }

    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() == Qt.Key_Space:
            self._stop_motors(); return
        if not self._rc_active: return
        if event.key() in self.KEY_MAP and event.key() not in self.active_keys:
            self.active_keys.add(event.key())
            ch, val = self.KEY_MAP[event.key()]
            self._v.motor_rc(**{ch: val})

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in self.KEY_MAP:
            self.active_keys.discard(event.key())
            ch, _ = self.KEY_MAP[event.key()]
            self._v.motor_rc(**{ch: 1500})

    def _update_ui(self):
        with self._telem_lock:
            t = dict(self._telem)
        self.arm_label.setText("ARMED" if t["armed"] else "DISARMED")
        self.arm_label.setStyleSheet(f"color: {C_DANGER if t['armed'] else C_WARN}; font-size: 16px; font-weight: bold;")
        self.mode_label.setText(t["mode"])
        self.depth_label.setText(f"{t['depth']:.2f} m")
        self.heading_label.setText(f"{t['heading']:.1f}")

        if self._heading_active and self._heading_target is not None:
            diff = (self._heading_target - t["heading"] + 180) % 360 - 180
            status = "TUTUYOR" if abs(diff) < 3 else "DONUYOR"
            color = C_OK if abs(diff) < 3 else C_WARN
            self.heading_status.setText(f"{status} | Hedef:{self._heading_target:.0f} Mevcut:{t['heading']:.0f} Fark:{diff:+.0f}")
            self.heading_status.setStyleSheet(f"color: {color}; font-size: 10px;")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def cleanup(self):
        self._stop.set()
        self._heading_active = False
        try: self._v.motor_rc_stop()
        except: pass


# ═══════════════════════════════════════════════════════════
# SEKME 2: Boru Takip (DDS) -- ayni tauv_gui.py mantigi
# ═══════════════════════════════════════════════════════════

class PipeTrackingTab(QWidget):
    def __init__(self, participant, vehicle: Vehicle, shared_state: dict, parent=None):
        super().__init__(parent)
        self._v = vehicle
        self._shared = shared_state
        self._pipe_state = {"tracking": False, "state": "IDLE", "cmd": {}, "result": None, "info": ""}
        self._pipe_lock = threading.Lock()

        self.processor = MaskProcessor(num_slices=8, slice_weights=[0.25,0.20,0.16,0.12,0.09,0.07,0.06,0.05])
        self.controller = PipeController()

        self.front_cam = DDSCameraReader(participant, "camera/front/frame")
        self.bottom_cam = DDSCameraReader(participant, "camera/bottom/frame")
        self.bottom_mask_gui = DDSMaskReader(participant, "sam3/bottom/segmentation_mask")
        self.front_mask_gui = DDSMaskReader(participant, "sam3/front/segmentation_mask")
        self._algo_mask = DDSMaskReader(participant, "sam3/bottom/segmentation_mask")

        self.front_cam.start()
        self.bottom_cam.start()
        self.bottom_mask_gui.start()
        self.front_mask_gui.start()
        self._algo_mask.start()

        self._stop_event = threading.Event()
        self._algo_thread = None

        self._build_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)

    def _algo_loop(self):
        while not self._stop_event.is_set():
            with self._pipe_lock:
                tracking = self._pipe_state["tracking"]
            if not tracking:
                time.sleep(0.05); continue

            sample = self._algo_mask.get_mask()
            heading = self._shared.get("heading_deg", 0.0)

            if sample is None:
                cmd = self.controller.compute(None, heading_deg=heading)
                self._v.motor_rc(yaw=cmd["yaw"], forward=cmd["forward"], lateral=cmd["lateral"])
                with self._pipe_lock:
                    self._pipe_state.update(cmd=cmd, state=self.controller.state, result=None, info="MASKE YOK")
                continue

            _, binary = cv2.threshold(sample, 128, 255, cv2.THRESH_BINARY)
            result = self.processor.process(binary)
            cmd = self.controller.compute(result, heading_deg=heading)
            self._v.motor_rc(yaw=cmd["yaw"], forward=cmd["forward"], lateral=cmd["lateral"])

            st = self.controller.state
            info_parts = [f"Gecis:{self.controller.pass_count}/2"]
            if st == PipeController.STATE_REVERSE:
                diff = (self.controller._reverse_target_hdg - heading + 180) % 360 - 180
                info_parts.append(f"180DON:{diff:+.0f}")
            elif result.error is not None:
                cont = "OK" if result.pipe_continues else "SON"
                info_parts.append(f"err={result.error:+.2f} scan={result.scan_hit_count}({cont})")
            info_parts.append(f"yaw={cmd['yaw']} fwd={cmd['forward']}")

            with self._pipe_lock:
                self._pipe_state.update(cmd=cmd, state=st, result=result, info="  ".join(info_parts))
                if st == PipeController.STATE_COMPLETE:
                    self._pipe_state["tracking"] = False

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(4)

        top = QHBoxLayout()
        tg = QGroupBox("Boru Takip")
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
        sam3_lay.addWidget(self.sam3_status)
        sam3_lay.addStretch()
        lay.addWidget(sam3_group)

        tune_group = QGroupBox("Tuning")
        tg_lay = QGridLayout(tune_group)
        self.tune_inputs = {}
        for i, (key, label, default) in enumerate([
            ("kp_yaw","Kp","150"),("ki_yaw","Ki","200"),("forward_pwm","Fwd","200"),
            ("max_yaw_pwm","MaxYaw","200"),("ema_alpha","EMA","0.6"),
            ("turn_curvature_thresh","Viraj","0.03"),("turn_forward_pwm","VFwd","150"),
            ("turn_yaw_boost","Boost","3.0"),("coast_timeout","CoastT","1.5"),
            ("coast_yaw_decay","Decay","0.55"),
        ]):
            row, col = i // 5, (i % 5) * 2
            tg_lay.addWidget(QLabel(label), row*2, col)
            inp = QLineEdit(default); inp.setMaximumWidth(55)
            tg_lay.addWidget(inp, row*2, col+1)
            self.tune_inputs[key] = inp
        btn_tune = QPushButton("Uygula")
        btn_tune.setStyleSheet(f"background: {C_ACCENT};")
        btn_tune.clicked.connect(self._apply_tune)
        tg_lay.addWidget(btn_tune, 2, 8, 1, 2)
        lay.addWidget(tune_group)

        lg = QGroupBox("Log")
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
            if not prompt: continue
            try:
                resp = requests.post(f"{url}/api/prompt", json={"camera": camera, "prompt": prompt}, timeout=5)
                results.append(f"{camera}={'OK' if resp.ok else 'HATA'}")
            except: results.append(f"{camera}=HATA")
        self.sam3_status.setText("  ".join(results))

    def _apply_tune(self):
        try:
            s = self.tune_inputs
            self.controller.update_params(
                kp_yaw=float(s["kp_yaw"].text()), ki_yaw=float(s["ki_yaw"].text()),
                forward_pwm=int(s["forward_pwm"].text()), max_yaw_pwm=int(s["max_yaw_pwm"].text()),
                ema_alpha=float(s["ema_alpha"].text()), turn_curvature_thresh=float(s["turn_curvature_thresh"].text()),
                turn_forward_pwm=int(s["turn_forward_pwm"].text()), turn_yaw_boost=float(s["turn_yaw_boost"].text()),
                coast_timeout=float(s["coast_timeout"].text()), coast_yaw_decay=float(s["coast_yaw_decay"].text()),
            )
            self._log("Tuning OK")
        except ValueError as e:
            self._log(f"Hata: {e}")

    def _overlay_mask(self, frame, raw_mask):
        h, w = frame.shape[:2]
        mh, mw = raw_mask.shape[:2]
        binary = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST) if (mh, mw) != (h, w) else raw_mask.copy()
        _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)
        ann = frame.copy()
        ov = ann.copy(); ov[binary > 0] = [0, 255, 0]
        return cv2.addWeighted(ann, 0.65, ov, 0.35, 0), binary

    def _tick(self):
        front = self.front_cam.get_frame()
        front_mask = self.front_mask_gui.get_mask()
        if front is not None and front_mask is not None:
            ann, _ = self._overlay_mask(front, front_mask)
            self._show(self.front_view, ann)
        elif front is not None:
            self._show(self.front_view, front)

        bottom = self.bottom_cam.get_frame()
        bottom_mask = self.bottom_mask_gui.get_mask()
        if bottom is not None and bottom_mask is not None:
            ann, binary = self._overlay_mask(bottom, bottom_mask)
            viz = self.processor.process(binary)
            if viz: self._draw_slices(ann, viz)
            self._show(self.bottom_view, ann)
        elif bottom is not None:
            self._show(self.bottom_view, bottom)

        with self._pipe_lock:
            st = self._pipe_state["state"]
            info = self._pipe_state["info"]
        self.state_label.setText(st)
        self.info_label.setText(info)
        if st == PipeController.STATE_COMPLETE and self.btn_stop.isEnabled():
            self.state_label.setStyleSheet(f"color: {C_OK}; font-size: 20px; font-weight: bold;")
            self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
            self._log("GOREV TAMAMLANDI")

    def _draw_slices(self, frame, result):
        if result is None or result.error is None or len(result.slice_centroids) < 2: return
        h, w = frame.shape[:2]
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        num = result.num_slices; sh = h // num
        for i in range(1, num): cv2.line(bgr, (0, i*sh), (w, i*sh), (100,100,100), 1)
        for idx, cx, cy, area in result.slice_centroids:
            wt = self.processor.slice_weights[idx] if idx < len(self.processor.slice_weights) else 0
            r = max(4, int(wt*40))
            cv2.circle(bgr, (int(cx), cy), r, (255,255,0), -1)
            cv2.circle(bgr, (int(cx), cy), r, (0,255,255), 2)
            cv2.putText(bgr, f"w={wt:.2f}", (int(cx)+r+2, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
        cv2.line(bgr, (int(result.center_x),0), (int(result.center_x),h), (0,0,255), 1)
        cv2.line(bgr, (int(result.weighted_cx),0), (int(result.weighted_cx),h), (0,255,0), 2)
        sc = sorted(result.slice_centroids, key=lambda c: c[0])
        pts = [(int(c[1]), c[2]) for c in sc]
        for j in range(len(pts)-1): cv2.line(bgr, pts[j], pts[j+1], (255,0,255), 2)
        for x1,y1,x2,y2,hit in result.scan_rays:
            cv2.line(bgr, (x1,y1), (x2,y2), (255,255,255) if hit else (0,0,255), 1)
        cont = "OK" if result.pipe_continues else "SON"
        cv2.putText(bgr, f"err={result.error:+.2f} ang={result.pipe_angle_deg:+.0f} {cont}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,200), 1)
        frame[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _show(self, label, frame_rgb):
        h, w = frame_rgb.shape[:2]
        dw, dh = label.width(), label.height()
        sc = min(dw/w, dh/h)
        nw, nh = int(w*sc), int(h*sc)
        r = cv2.resize(frame_rgb, (nw, nh))
        label.setPixmap(QPixmap.fromImage(QImage(r.data, nw, nh, nw*3, QImage.Format_RGB888)))

    def _on_start(self):
        self.controller.reset()
        with self._pipe_lock:
            self._pipe_state.update(tracking=True, state="FOLLOW", info="")
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.state_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 20px; font-weight: bold;")
        if self._algo_thread is None or not self._algo_thread.is_alive():
            self._stop_event.clear()
            self._algo_thread = threading.Thread(target=self._algo_loop, daemon=True, name="pipe-algo")
            self._algo_thread.start()
        self._log("Takip baslatildi")

    def _on_stop(self):
        with self._pipe_lock: self._pipe_state["tracking"] = False
        self._v.motor_rc_stop()
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.state_label.setText("IDLE"); self.info_label.setText("")
        self._log("Durduruldu")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def cleanup(self):
        with self._pipe_lock: self._pipe_state["tracking"] = False
        self._stop_event.set()
        if self._algo_thread and self._algo_thread.is_alive(): self._algo_thread.join(timeout=3)
        self.front_cam.stop(); self.bottom_cam.stop()
        self.bottom_mask_gui.stop(); self.front_mask_gui.stop(); self._algo_mask.stop()


# ═══════════════════════════════════════════════════════════

class TauvMainWindow2(QMainWindow):
    def __init__(self, vehicle, participant):
        super().__init__()
        self.setWindowTitle("TAUV Control Suite v2 (DDS)")
        self.setMinimumSize(1250, 950)
        self.setStyleSheet(DARK_STYLE)
        shared = {"heading_deg": 0.0}
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)
        self._vtab = VehicleControlTab(vehicle, shared)
        self._ptab = PipeTrackingTab(participant, vehicle, shared)
        self._tabs.addTab(self._vtab, "Arac Kontrolu (DDS)")
        self._tabs.addTab(self._ptab, "Boru Takip")

    def keyPressEvent(self, event):
        if self._tabs.currentIndex() == 0: self._vtab.keyPressEvent(event)
        else: super().keyPressEvent(event)
    def keyReleaseEvent(self, event):
        if self._tabs.currentIndex() == 0: self._vtab.keyReleaseEvent(event)
        else: super().keyReleaseEvent(event)
    def closeEvent(self, event):
        self._vtab.cleanup(); self._ptab.cleanup(); event.accept()


def main():
    parser = argparse.ArgumentParser(description="TAUV Control Suite v2 (DDS)")
    parser.add_argument("--domain-id", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("  TAUV CONTROL SUITE v2 -- Tamamen DDS")
    print("  Sekme 1: Arac Kontrolu (Vehicle API)")
    print("  Sekme 2: Boru Takip (DDS + Algoritma)")
    print("=" * 60)

    vehicle = Vehicle()
    participant = vehicle._participant

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 30, 46))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(42, 42, 62))
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.Button, QColor(42, 42, 62))
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.Highlight, QColor(124, 138, 255))
    p.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(p)

    w = TauvMainWindow2(vehicle, participant)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
