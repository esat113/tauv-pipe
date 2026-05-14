#!/usr/bin/env python3
"""
Pipe Tracker GUI v4 - decoupled visual servo.

SAM3 mask -> PipeGeometryProcessor -> PipeVisualServoController -> motor_rc

Fark:
- Boru merkezi hatasi `lateral` kanalina gider.
- Boru acisi/lookahead hatasi `yaw` kanalina gider.
- Maske gecikmesi ve ayni maskeyi tekrar isleme kontrol edilir.
"""

import os
from pathlib import Path
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
_local_dds_config = Path(__file__).resolve().parent / "dds_config.xml"
if _local_dds_config.exists() and os.environ.get("TAUV_PIPE_USE_ENV_DDS") != "1":
    os.environ["CYCLONEDDS_URI"] = f"file://{_local_dds_config}"
elif "CYCLONEDDS_URI" not in os.environ and _local_dds_config.exists():
    os.environ["CYCLONEDDS_URI"] = f"file://{_local_dds_config}"

_tauv_client_src = Path(__file__).resolve().parent.parent / "tauv-client" / "src"
if _tauv_client_src.exists():
    sys.path.insert(0, str(_tauv_client_src))

import cv2
import numpy as np
import requests

try:
    from tauv_client import Vehicle
    from tauv_client.guidance.path_tracking import ned_attitude_yaw_deg
except Exception as _vehicle_exc:
    Vehicle = None
    ned_attitude_yaw_deg = None
    _vehicle_import_error = _vehicle_exc
else:
    _vehicle_import_error = None

from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import QColor, QBrush, QImage, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from cyclonedds.domain import DomainParticipant

from pipe_algorithm import (
    PipeGeometryProcessor,
    PipeGeometryResult,
    PipeVisualServoController,
)
from pipe_tracker_aruco import ArucoMarkerDetector, ArucoResult
from pipe_tracker_dds import (
    DDSCameraReader,
    DDSMaskReader,
    DDSMotorPublisher,
)
from pipe_reacquire import (
    LastPipeObservation,
    PoseHistory,
    PoseSample,
    ReturnCommand,
    ReturnToLastSeenController,
    observation_from_pose_history,
    select_return_observation,
)


@dataclass(frozen=True)
class MappedPipePoint:
    x: float
    y: float
    vehicle_x: float
    vehicle_y: float
    yaw: float
    lateral_error: float
    heading_error_deg: float
    confidence: float
    wall_time_s: float


class PipeMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(960, 600)
        self._pose_trail: list[PoseSample] = []
        self._pipe_trail: list[MappedPipePoint] = []
        self._current_pose: PoseSample | None = None
        self._return_observation: LastPipeObservation | None = None
        self._return_cmd: ReturnCommand | None = None
        self._state = "IDLE"
        self._pose_ok = False

    def set_data(
        self,
        pose_trail: list[PoseSample],
        pipe_trail: list[MappedPipePoint],
        current_pose: PoseSample | None,
        return_observation: LastPipeObservation | None,
        return_cmd: ReturnCommand | None,
        state: str,
        pose_ok: bool,
    ) -> None:
        self._pose_trail = list(pose_trail)
        self._pipe_trail = list(pipe_trail)
        self._current_pose = current_pose
        self._return_observation = return_observation
        self._return_cmd = return_cmd
        self._state = str(state or "IDLE")
        self._pose_ok = bool(pose_ok)
        self.update()

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(7, 9, 12))

        w = max(1, self.width())
        h = max(1, self.height())
        margin = 44
        points = [(s.x, s.y) for s in self._pose_trail if s.usable]
        points.extend((pt.x, pt.y) for pt in self._pipe_trail)
        if self._current_pose is not None and self._current_pose.usable:
            points.append((self._current_pose.x, self._current_pose.y))
        if self._return_observation is not None and self._return_observation.pose.usable:
            points.append((self._return_observation.pose.x, self._return_observation.pose.y))
        if self._return_cmd is not None:
            if self._return_cmd.target_x is not None and self._return_cmd.target_y is not None:
                points.append((self._return_cmd.target_x, self._return_cmd.target_y))

        if points:
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            span_x = max(max(xs) - min(xs), 1.0)
            span_y = max(max(ys) - min(ys), 1.0)
        else:
            cx = cy = 0.0
            span_x = span_y = 2.0

        sx = max(1.0, (w - 2 * margin) / span_x)
        sy = max(1.0, (h - 2 * margin) / span_y)
        scale = max(18.0, min(150.0, min(sx, sy)))

        def screen(x: float, y: float) -> QPointF:
            return QPointF(w * 0.5 + (x - cx) * scale, h * 0.5 - (y - cy) * scale)

        self._draw_grid(p, screen, cx, cy, scale)
        self._draw_polyline(p, [screen(s.x, s.y) for s in self._pose_trail if s.usable], QColor(70, 145, 255), 2)
        self._draw_polyline(
            p,
            [screen(pt.x, pt.y) for pt in self._pipe_trail],
            QColor(236, 190, 70),
            3,
        )

        for pipe_pt in self._pipe_trail[-160:]:
            c = QColor(236, 190, 70)
            c.setAlpha(70 + int(120 * max(0.0, min(1.0, pipe_pt.confidence))))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(c))
            pt = screen(pipe_pt.x, pipe_pt.y)
            p.drawEllipse(pt, 3.0, 3.0)
            self._draw_pipe_heading_tick(p, pt, pipe_pt.yaw)

        if self._return_observation is not None and self._return_observation.pose.usable:
            pt = screen(self._return_observation.pose.x, self._return_observation.pose.y)
            p.setPen(QPen(QColor(235, 80, 80), 2))
            p.setBrush(QBrush(QColor(235, 80, 80)))
            p.drawEllipse(pt, 7.0, 7.0)
            p.drawLine(QPointF(pt.x() - 12, pt.y()), QPointF(pt.x() + 12, pt.y()))
            p.drawLine(QPointF(pt.x(), pt.y() - 12), QPointF(pt.x(), pt.y() + 12))

        if self._return_cmd is not None:
            self._draw_return_area(p, screen, scale)

        if self._current_pose is not None and self._current_pose.usable:
            self._draw_vehicle(p, screen(self._current_pose.x, self._current_pose.y), self._current_pose.yaw)

        p.setPen(QPen(QColor(220, 228, 236), 1))
        p.drawText(14, 24, f"state={self._state}  ekf={'OK' if self._pose_ok else 'NO'}")
        p.drawText(14, 44, f"vehicle_pts={len(self._pose_trail)}  pipe_pts={len(self._pipe_trail)}  scale={scale:.0f}px/m")
        p.end()

    def _draw_grid(self, p: QPainter, screen, cx: float, cy: float, scale: float) -> None:
        del cx, cy
        w = self.width()
        h = self.height()
        step_m = 1.0
        if scale < 35:
            step_m = 2.0
        elif scale > 95:
            step_m = 0.5
        origin = screen(0.0, 0.0)
        grid_color = QColor(42, 50, 60)
        axis_color = QColor(92, 106, 122)
        p.setPen(QPen(grid_color, 1))

        x = origin.x()
        while x < w:
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            x += step_m * scale
        x = origin.x() - step_m * scale
        while x >= 0:
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            x -= step_m * scale

        y = origin.y()
        while y < h:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y += step_m * scale
        y = origin.y() - step_m * scale
        while y >= 0:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y -= step_m * scale

        p.setPen(QPen(axis_color, 2))
        p.drawLine(QPointF(origin.x(), 0), QPointF(origin.x(), h))
        p.drawLine(QPointF(0, origin.y()), QPointF(w, origin.y()))

    @staticmethod
    def _draw_polyline(p: QPainter, pts: list[QPointF], color: QColor, width: int) -> None:
        if len(pts) < 2:
            return
        p.setPen(QPen(color, width))
        for a, b in zip(pts[:-1], pts[1:]):
            p.drawLine(a, b)

    def _draw_return_area(self, p: QPainter, screen, scale: float) -> None:
        cmd = self._return_cmd
        if cmd is None:
            return
        if cmd.target_x is not None and cmd.target_y is not None:
            pt = screen(cmd.target_x, cmd.target_y)
            p.setPen(QPen(QColor(120, 230, 255), 2))
            p.setBrush(QBrush(QColor(120, 230, 255)))
            p.drawEllipse(pt, 5.0, 5.0)

    @staticmethod
    def _draw_vehicle(p: QPainter, pos: QPointF, yaw: float) -> None:
        length = 20.0
        width = 10.0
        c = np.cos(float(yaw))
        s = np.sin(float(yaw))

        def rot(dx: float, dy: float) -> QPointF:
            return QPointF(pos.x() + dx * c - dy * s, pos.y() - (dx * s + dy * c))

        poly = QPolygonF([
            rot(length, 0.0),
            rot(-length * 0.55, width),
            rot(-length * 0.25, 0.0),
            rot(-length * 0.55, -width),
        ])
        p.setPen(QPen(QColor(40, 255, 150), 2))
        p.setBrush(QBrush(QColor(40, 170, 110)))
        p.drawPolygon(poly)

    @staticmethod
    def _draw_pipe_heading_tick(p: QPainter, pos: QPointF, yaw: float) -> None:
        half = 9.0
        c = np.cos(float(yaw))
        s = np.sin(float(yaw))
        a = QPointF(pos.x() - half * c, pos.y() + half * s)
        b = QPointF(pos.x() + half * c, pos.y() - half * s)
        p.setPen(QPen(QColor(250, 220, 120, 120), 1))
        p.drawLine(a, b)


class PipeTrackerServoWindow(QMainWindow):
    def __init__(self, participant: DomainParticipant):
        super().__init__()
        self.tracking = False
        self.last_cmd = {}

        self.processor = PipeGeometryProcessor()
        self.controller = PipeVisualServoController()
        self.return_controller = ReturnToLastSeenController()
        self.aruco_detector = ArucoMarkerDetector()
        self._mask_stale_s = 0.35
        self._pose_match_max_delta_ms = 600.0
        self._return_target_lookback_s = 1.2
        self._return_min_target_distance_m = 0.45
        self._pipe_map_lateral_m_per_unit = 0.35
        self._pipe_map_lateral_sign = 1.0
        self._pipe_map_heading_sign = 1.0

        self.bottom_cam = DDSCameraReader(participant, "camera/bottom/frame")
        self.bottom_mask_reader = DDSMaskReader(participant, "sam3/bottom/segmentation_mask")
        self.cmd_pub = DDSMotorPublisher(participant, "embedded/control/stream_command")

        self._vehicle = None
        self._pose_history = PoseHistory(maxlen=160)
        self._pose_stop = threading.Event()
        self._pose_thread: threading.Thread | None = None
        self._pose_ok = False
        self._pose_error = ""
        self._last_pipe_observation: LastPipeObservation | None = None
        self._pipe_observation_history: deque[LastPipeObservation] = deque(maxlen=120)
        self._map_pose_trail: deque[PoseSample] = deque(maxlen=2500)
        self._map_pipe_trail: deque[MappedPipePoint] = deque(maxlen=2500)
        self._selected_return_observation: LastPipeObservation | None = None
        self._last_map_pose_ts_ms: float | None = None
        self._last_valid_pipe_wall_t = 0.0
        self._last_return_cmd: ReturnCommand | None = None
        self._last_return_state = ""

        self._last_processed_mask_ts_ms = None
        self._last_geom: PipeGeometryResult | None = None
        self._last_geom_wall_t = 0.0
        self._sync_pair_cache = None
        self._last_mask_wall_t = None
        self._mask_fps_ema = 0.0
        self._last_mask_diag = "mask=YOK"
        self._last_aruco_result: ArucoResult = self.aruco_detector.status()

        self.bottom_cam.start()
        self.bottom_mask_reader.start()
        self._init_ui()
        self._update_aruco_info()
        self._start_pose_reader()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def _init_ui(self):
        self.setWindowTitle("Pipe Tracker v4 - Decoupled Visual Servo")
        self.setMinimumSize(1250, 900)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #15171b; }
            QGroupBox {
                color: #d77a61; font-weight: bold; border: 1px solid #39414d;
                border-radius: 5px; margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #e8edf2; }
            QPushButton {
                background: #222a33; color: white; border: 1px solid #46515f;
                border-radius: 5px; padding: 8px 16px; font-weight: bold;
            }
            QPushButton:hover { background: #2b3540; border-color: #d77a61; }
            QLineEdit {
                background: #20262e; color: white; border: 1px solid #46515f;
                padding: 3px;
            }
            QTextEdit {
                background: #101216; color: #b8c0ca; border: 1px solid #303842;
                font-family: monospace; font-size: 11px;
            }
            QTabWidget::pane { border: 1px solid #303842; }
            QTabBar::tab {
                background: #20262e; color: #d6dde6; padding: 8px 16px;
                border: 1px solid #303842;
            }
            QTabBar::tab:selected { background: #2b3540; color: white; }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        central_lay.addWidget(self.tabs)

        tracker_tab = QWidget()
        root = QVBoxLayout(tracker_tab)
        root.setSpacing(4)

        mission_group = QGroupBox("Gorev")
        mission = QHBoxLayout(mission_group)
        self.btn_start = QPushButton("BASLAT")
        self.btn_start.setStyleSheet("background: #1565C0; font-size: 15px; padding: 10px 40px;")
        self.btn_start.clicked.connect(self.on_start)
        mission.addWidget(self.btn_start)
        self.btn_stop = QPushButton("DURDUR")
        self.btn_stop.setStyleSheet("background: #b3261e; font-size: 15px; padding: 10px 40px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)
        mission.addWidget(self.btn_stop)
        self.state_label = QLabel("IDLE")
        self.state_label.setStyleSheet("color: #d77a61; font-size: 20px; font-weight: bold;")
        mission.addWidget(self.state_label)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #b8c0ca; font-size: 11px;")
        mission.addWidget(self.info_label, stretch=1)
        root.addWidget(mission_group)

        cam_group = QGroupBox("Alt Kamera + SAM3 Mask + Geometri")
        cam_lay = QVBoxLayout(cam_group)
        self.bottom_view = QLabel("DDS bekleniyor...")
        self.bottom_view.setMinimumSize(960, 600)
        self.bottom_view.setStyleSheet("background: #050608; border: 1px solid #303842;")
        self.bottom_view.setAlignment(Qt.AlignCenter)
        cam_lay.addWidget(self.bottom_view)
        root.addWidget(cam_group)

        aruco_group = QGroupBox("ArUco Marker")
        aruco_lay = QHBoxLayout(aruco_group)
        self.aruco_label = QLabel("")
        self.aruco_label.setStyleSheet("color: #b8c0ca; font-size: 12px;")
        aruco_lay.addWidget(self.aruco_label, stretch=1)
        root.addWidget(aruco_group)

        prompt_group = QGroupBox("SAM3 Prompt")
        prompt_lay = QHBoxLayout(prompt_group)
        prompt_lay.addWidget(QLabel("Bottom:"))
        self.prompt_input = QLineEdit("pipe")
        self.prompt_input.setMaximumWidth(140)
        prompt_lay.addWidget(self.prompt_input)
        prompt_lay.addWidget(QLabel("URL:"))
        self.sam3_url_input = QLineEdit("http://localhost:5003")
        self.sam3_url_input.setMaximumWidth(220)
        prompt_lay.addWidget(self.sam3_url_input)
        self.btn_prompt = QPushButton("Gonder")
        self.btn_prompt.clicked.connect(self._send_prompt)
        prompt_lay.addWidget(self.btn_prompt)
        self.sam3_status_label = QLabel("")
        self.sam3_status_label.setStyleSheet("color: #b8c0ca; font-size: 11px;")
        prompt_lay.addWidget(self.sam3_status_label)
        prompt_lay.addStretch()
        root.addWidget(prompt_group)

        tune_group = QGroupBox("Tuning")
        tune = QGridLayout(tune_group)
        self.tune_inputs = {}
        tune_defs = [
            ("kp_lateral", "Kp lateral", "95"),
            ("kp_yaw_angle", "Kp yaw angle", "2.0"),
            ("kp_yaw_lookahead", "Kp yaw look", "55"),
            ("base_forward_pwm", "Forward", "105"),
            ("acquire_forward_pwm", "Acquire fwd", "35"),
            ("max_lateral_pwm", "Max lateral", "120"),
            ("max_yaw_pwm", "Max yaw", "120"),
            ("min_confidence", "Min conf", "0.25"),
            ("ema_alpha", "EMA", "0.45"),
            ("yaw_sign", "Yaw sign", "1"),
            ("lateral_sign", "Lat sign", "1"),
            ("mask_stale_s", "Mask stale s", "0.35"),
            ("lookahead_y_ratio", "Look y", "0.25"),
            ("center_y_ratio", "Center y", "0.55"),
            ("min_area_ratio", "Min area", "0.002"),
            ("pose_match_ms", "Pose match ms", "600"),
            ("return_target_lookback_s", "Ret lookback", "1.2"),
            ("return_min_target_distance_m", "Ret min dist", "0.45"),
            ("pipe_map_lateral_m_per_unit", "Map lat m/u", "0.35"),
            ("pipe_map_lateral_sign", "Map lat sign", "1"),
            ("pipe_map_heading_sign", "Map ang sign", "1"),
            ("return_delay_s", "Return delay", "0.7"),
            ("return_accept_radius_m", "Return radius", "0.35"),
            ("return_timeout_s", "Return timeout", "10.0"),
            ("max_last_seen_age_s", "Seen max age", "8.0"),
            ("return_speed_m_s", "Return speed", "0.15"),
            ("return_fwd_kp", "Ret fwd Kp", "1.0"),
            ("return_fwd_ki", "Ret fwd Ki", "0.0"),
            ("return_fwd_kd", "Ret fwd Kd", "0.0"),
            ("return_forward_pwm_min_offset", "Ret fwd min", "35"),
            ("return_forward_heading_gate_deg", "Ret fwd gate", "70"),
            ("return_heading_soft_gate_deg", "Ret soft gate", "25"),
            ("return_forward_min_scale", "Ret fwd scale", "0.35"),
            ("return_lateral_kp_pwm_per_m", "Ret lat Kp", "80"),
            ("return_lateral_kd_pwm_s_per_m", "Ret lat Kd", "4"),
            ("return_pwm_min", "Ret PWM min", "1450"),
            ("return_pwm_max", "Ret PWM max", "1550"),
            ("max_fwd_pwm_step", "Ret slew", "20"),
            ("align_heading_tol_deg", "Ret align", "15"),
            ("return_yaw_kp_pwm_per_rad", "Ret yaw Kp", "90"),
            ("return_yaw_pwm_max", "Ret yaw max", "80"),
            ("return_yaw_sign", "Ret yaw sign", "1"),
            ("local_search_yaw_pwm", "Local yaw", "35"),
        ]
        for i, (key, label, default) in enumerate(tune_defs):
            row = i // 5
            col = (i % 5) * 2
            tune.addWidget(QLabel(label), row, col)
            inp = QLineEdit(default)
            inp.setMaximumWidth(78)
            tune.addWidget(inp, row, col + 1)
            self.tune_inputs[key] = inp

        self.btn_apply = QPushButton("Uygula")
        self.btn_apply.clicked.connect(self._apply_tune)
        button_row = (len(tune_defs) + 4) // 5
        tune.addWidget(self.btn_apply, button_row, 8, 1, 2)
        root.addWidget(tune_group)

        log_group = QGroupBox("Log")
        log_lay = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_lay.addWidget(self.log_text)
        root.addWidget(log_group)
        self.tabs.addTab(tracker_tab, "Tracker")

        map_tab = QWidget()
        map_root = QVBoxLayout(map_tab)
        map_group = QGroupBox("EKF 2D Map + Pipe Trail")
        map_lay = QVBoxLayout(map_group)
        self.map_widget = PipeMapWidget()
        map_lay.addWidget(self.map_widget)
        map_footer = QHBoxLayout()
        self.map_info_label = QLabel(
            "Mavi: arac izi  |  Sari: boru gorulen pozlar  |  Kirmizi: return target"
        )
        self.map_info_label.setStyleSheet("color: #b8c0ca; font-size: 12px;")
        map_footer.addWidget(self.map_info_label, stretch=1)
        self.btn_clear_map = QPushButton("Haritayi Temizle")
        self.btn_clear_map.clicked.connect(self._clear_map)
        map_footer.addWidget(self.btn_clear_map)
        map_lay.addLayout(map_footer)
        map_root.addWidget(map_group)
        self.tabs.addTab(map_tab, "2D Map")

    def _send_prompt(self):
        url = self.sam3_url_input.text().strip().rstrip("/")
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.sam3_status_label.setText("Prompt bos")
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
                kp_lateral=float(s["kp_lateral"].text()),
                kp_yaw_angle=float(s["kp_yaw_angle"].text()),
                kp_yaw_lookahead=float(s["kp_yaw_lookahead"].text()),
                base_forward_pwm=int(s["base_forward_pwm"].text()),
                acquire_forward_pwm=int(s["acquire_forward_pwm"].text()),
                max_lateral_pwm=int(s["max_lateral_pwm"].text()),
                max_yaw_pwm=int(s["max_yaw_pwm"].text()),
                min_confidence=float(s["min_confidence"].text()),
                ema_alpha=float(s["ema_alpha"].text()),
                yaw_sign=float(s["yaw_sign"].text()),
                lateral_sign=float(s["lateral_sign"].text()),
            )
            self.processor.lookahead_y_ratio = float(s["lookahead_y_ratio"].text())
            self.processor.center_y_ratio = float(s["center_y_ratio"].text())
            self.processor.min_area_ratio = float(s["min_area_ratio"].text())
            self._mask_stale_s = float(s["mask_stale_s"].text())
            self._pose_match_max_delta_ms = float(s["pose_match_ms"].text())
            self._return_target_lookback_s = float(s["return_target_lookback_s"].text())
            self._return_min_target_distance_m = float(s["return_min_target_distance_m"].text())
            self._pipe_map_lateral_m_per_unit = float(s["pipe_map_lateral_m_per_unit"].text())
            self._pipe_map_lateral_sign = float(s["pipe_map_lateral_sign"].text())
            self._pipe_map_heading_sign = float(s["pipe_map_heading_sign"].text())
            self.return_controller.update_params(
                return_delay_s=float(s["return_delay_s"].text()),
                return_accept_radius_m=float(s["return_accept_radius_m"].text()),
                return_timeout_s=float(s["return_timeout_s"].text()),
                max_last_seen_age_s=float(s["max_last_seen_age_s"].text()),
                return_speed_m_s=float(s["return_speed_m_s"].text()),
                return_fwd_kp=float(s["return_fwd_kp"].text()),
                return_fwd_ki=float(s["return_fwd_ki"].text()),
                return_fwd_kd=float(s["return_fwd_kd"].text()),
                return_forward_pwm_min_offset=int(s["return_forward_pwm_min_offset"].text()),
                return_forward_heading_gate_deg=float(s["return_forward_heading_gate_deg"].text()),
                return_heading_soft_gate_deg=float(s["return_heading_soft_gate_deg"].text()),
                return_forward_min_scale=float(s["return_forward_min_scale"].text()),
                return_lateral_kp_pwm_per_m=float(s["return_lateral_kp_pwm_per_m"].text()),
                return_lateral_kd_pwm_s_per_m=float(s["return_lateral_kd_pwm_s_per_m"].text()),
                return_pwm_min=int(s["return_pwm_min"].text()),
                return_pwm_max=int(s["return_pwm_max"].text()),
                max_fwd_pwm_step=int(s["max_fwd_pwm_step"].text()),
                align_heading_tol_deg=float(s["align_heading_tol_deg"].text()),
                return_yaw_kp_pwm_per_rad=float(s["return_yaw_kp_pwm_per_rad"].text()),
                return_yaw_pwm_max=int(s["return_yaw_pwm_max"].text()),
                return_yaw_sign=float(s["return_yaw_sign"].text()),
                local_search_yaw_pwm=int(s["local_search_yaw_pwm"].text()),
            )
            self._log("Tuning uygulandi")
        except ValueError as exc:
            self._log(f"Tuning hatasi: {exc}")

    def _start_pose_reader(self) -> None:
        if Vehicle is None:
            self._pose_error = f"tauv-client import edilemedi: {_vehicle_import_error!r}"
            self._log(f"UYARI: {self._pose_error}; EKF reacquire kapali")
            return
        try:
            self._vehicle = Vehicle()
        except Exception as exc:
            self._pose_error = f"Vehicle baslatilamadi: {exc}"
            self._log(f"UYARI: {self._pose_error}; EKF reacquire kapali")
            return
        self._pose_stop.clear()
        self._pose_thread = threading.Thread(target=self._pose_loop, daemon=True, name="pipe-ekf-pose")
        self._pose_thread.start()
        self._log("EKF pose thread basladi (Vehicle.state.snapshot)")

    def _pose_loop(self) -> None:
        while not self._pose_stop.is_set():
            try:
                snap = self._vehicle.state.snapshot
                sample = self._pose_history.push_snapshot(snap)
                self._pose_ok = sample.usable
                self._pose_error = "" if sample.usable else "pose stale/uninitialized"
            except Exception as exc:
                self._pose_ok = False
                self._pose_error = str(exc)
            self._pose_stop.wait(0.05)

    def _current_pose(self) -> PoseSample | None:
        return self._pose_history.latest(require_usable=True)

    def _record_pose_for_map(self, pose: PoseSample | None) -> None:
        if pose is None or not pose.usable:
            return
        if self._last_map_pose_ts_ms == pose.timestamp_ms:
            return
        self._last_map_pose_ts_ms = pose.timestamp_ms
        self._map_pose_trail.append(pose)

    def _refresh_map(
        self,
        state: str,
        current_pose: PoseSample | None,
        ret_cmd: ReturnCommand | None = None,
    ) -> None:
        return_obs = self.return_controller.observation or self._selected_return_observation
        self.map_widget.set_data(
            pose_trail=list(self._map_pose_trail),
            pipe_trail=list(self._map_pipe_trail),
            current_pose=current_pose,
            return_observation=return_obs,
            return_cmd=ret_cmd,
            state=state,
            pose_ok=self._pose_ok,
        )
        if current_pose is not None and current_pose.usable:
            target_text = "target=YOK"
            if return_obs is not None:
                target_dist = ((return_obs.pose.x - current_pose.x) ** 2 + (return_obs.pose.y - current_pose.y) ** 2) ** 0.5
                target_text = f"target d={target_dist:.2f}m"
            ret_text = ""
            if ret_cmd is not None:
                ret_text = (
                    f"  |  return={ret_cmd.reason} d={ret_cmd.distance_m:.2f}m "
                    f"herr={ret_cmd.heading_error_deg:+.0f}deg"
                )
            self.map_info_label.setText(
                f"x={current_pose.x:+.2f} y={current_pose.y:+.2f} "
                f"yaw={np.degrees(current_pose.yaw):.1f}deg  |  "
                f"pose_pts={len(self._map_pose_trail)} pipe_pts={len(self._map_pipe_trail)}  |  "
                f"{target_text}{ret_text}"
            )
        else:
            self.map_info_label.setText(
                f"EKF pose yok  |  pose_pts={len(self._map_pose_trail)} pipe_pts={len(self._map_pipe_trail)}"
            )

    def _clear_map(self) -> None:
        self._map_pose_trail.clear()
        self._map_pipe_trail.clear()
        self._pipe_observation_history.clear()
        self._last_pipe_observation = None
        self._selected_return_observation = None
        self._last_map_pose_ts_ms = None
        self._refresh_map(self.state_label.text(), self._current_pose(), self._last_return_cmd)
        self._log("2D map temizlendi")

    def _pipe_geom_valid(self, geom: PipeGeometryResult | None) -> bool:
        return (
            geom is not None
            and geom.found
            and geom.confidence >= float(self.controller.min_confidence)
        )

    def _record_last_seen_if_valid(
        self,
        geom: PipeGeometryResult,
        mask_ts_ms: float,
        wall_now: float,
    ) -> None:
        if not self._pipe_geom_valid(geom):
            return
        self._last_valid_pipe_wall_t = float(wall_now)
        obs = observation_from_pose_history(
            self._pose_history,
            mask_ts_ms=mask_ts_ms,
            wall_time_s=wall_now,
            confidence=geom.confidence,
            reason=geom.reason,
            pose_match_max_delta_ms=self._pose_match_max_delta_ms,
        )
        if obs is not None:
            self._last_pipe_observation = obs
            self._pipe_observation_history.append(obs)
            self._map_pipe_trail.append(self._map_pipe_point(obs, geom))

    def _map_pipe_point(self, obs: LastPipeObservation, geom: PipeGeometryResult) -> MappedPipePoint:
        pose = obs.pose
        lateral_m = (
            self._pipe_map_lateral_sign
            * self._pipe_map_lateral_m_per_unit
            * float(geom.lateral_error)
        )
        right_x = -np.sin(pose.yaw)
        right_y = np.cos(pose.yaw)
        pipe_x = pose.x + lateral_m * right_x
        pipe_y = pose.y + lateral_m * right_y
        pipe_yaw = pose.yaw + np.radians(self._pipe_map_heading_sign * float(geom.heading_error_deg))
        return MappedPipePoint(
            x=float(pipe_x),
            y=float(pipe_y),
            vehicle_x=float(pose.x),
            vehicle_y=float(pose.y),
            yaw=float(pipe_yaw),
            lateral_error=float(geom.lateral_error),
            heading_error_deg=float(geom.heading_error_deg),
            confidence=float(obs.confidence),
            wall_time_s=float(obs.wall_time_s),
        )

    def _return_observation_target(
        self,
        now_s: float,
        current_pose: PoseSample | None,
    ) -> LastPipeObservation | None:
        max_age = max(0.0, float(self.return_controller.max_last_seen_age_s))
        while (
            self._pipe_observation_history
            and self._pipe_observation_history[0].age_s(now_s) > max_age
        ):
            self._pipe_observation_history.popleft()
        return select_return_observation(
            self._pipe_observation_history,
            now_s=now_s,
            current_pose=current_pose,
            max_age_s=max_age,
            target_lookback_s=self._return_target_lookback_s,
            min_target_distance_m=max(
                self._return_min_target_distance_m,
                float(self.return_controller.return_accept_radius_m),
            ),
        )

    def _log_return_transition(self, ret: ReturnCommand | None) -> None:
        state = ret.state if ret is not None else ""
        if state == self._last_return_state:
            return
        self._last_return_state = state
        if ret is None or state == ReturnToLastSeenController.STATE_IDLE:
            return
        self._log(
            f"Reacquire {state}: {ret.reason}, "
            f"d={ret.distance_m:.2f}m, last_seen_age={ret.last_seen_age_s:.1f}s"
        )

    def _send_return_yaw(self, ret: ReturnCommand) -> None:
        if ret.target_yaw_odom_rad is None:
            return
        if self._vehicle is None or ned_attitude_yaw_deg is None:
            return
        try:
            yaw_deg = ned_attitude_yaw_deg(ret.target_yaw_odom_rad, self._vehicle.state.tree)
            self._vehicle.set_target_attitude(yaw_deg)
        except Exception as exc:
            self._pose_error = f"set_target_attitude failed: {exc}"

    @staticmethod
    def _normalize_frame_ts_ms(ts: float) -> float:
        if ts <= 0:
            return float(ts)
        t = float(ts)
        if t < 1e11:
            return t * 1000.0
        return t

    @staticmethod
    def _binary_mask_for_frame(frame: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
        h_cam, w_cam = frame.shape[:2]
        h_mask, w_mask = raw_mask.shape[:2]
        if (h_mask, w_mask) != (h_cam, w_cam):
            mask = cv2.resize(raw_mask, (w_cam, h_cam), interpolation=cv2.INTER_NEAREST)
        else:
            mask = raw_mask.copy()
        mask = mask.astype(np.uint8)
        if mask.size and int(mask.max()) <= 1:
            binary = (mask > 0).astype(np.uint8) * 255
        else:
            _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        return binary

    def _annotate(
        self,
        frame: np.ndarray,
        raw_mask: np.ndarray,
        wall_now: float,
    ) -> tuple[np.ndarray, PipeGeometryResult]:
        aruco = self.aruco_detector.detect(frame, now=wall_now)
        self._last_aruco_result = aruco

        binary = self._binary_mask_for_frame(frame, raw_mask)
        geom = self.processor.process(binary)

        annotated = frame.copy()
        overlay = annotated.copy()
        overlay[binary > 0] = [255, 40, 40]
        annotated = cv2.addWeighted(annotated, 0.55, overlay, 0.45, 0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 230, 60), 2)
        self._draw_geometry(annotated, geom)
        annotated = self.aruco_detector.annotate(annotated, aruco)
        return annotated, geom

    def _annotate_mask_only(self, raw_mask: np.ndarray) -> tuple[np.ndarray, PipeGeometryResult]:
        mask = raw_mask.astype(np.uint8)
        if mask.size and int(mask.max()) <= 1:
            binary = (mask > 0).astype(np.uint8) * 255
        else:
            _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        geom = self.processor.process(binary)
        color = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        color[:, :] = [18, 24, 32]
        color[binary > 0] = [255, 40, 40]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color, contours, -1, (255, 230, 60), 2)
        self._draw_geometry(color, geom)
        return color, geom

    def _tick(self):
        wall_now = time.monotonic()
        sample = self.bottom_mask_reader.get_mask_with_meta()
        if sample is not None:
            raw_mask, mask_ts_raw = sample
            mask_ts_ms = self._normalize_frame_ts_ms(mask_ts_raw)
            new_mask = mask_ts_ms != self._last_processed_mask_ts_ms
            nz = int(cv2.countNonZero(raw_mask.astype(np.uint8)))
            maxv = int(raw_mask.max()) if raw_mask.size else 0
            self._last_mask_diag = f"mask={raw_mask.shape[1]}x{raw_mask.shape[0]} nz={nz} max={maxv}"

            if new_mask:
                frame = self.bottom_cam.get_frame_at(mask_ts_ms, max_age_ms=600.0)
                frame_mode = "matched"
                if frame is None:
                    frame = self.bottom_cam.get_frame()
                    frame_mode = "fallback" if frame is not None else "none"
                if frame is not None:
                    annotated, geom = self._annotate(frame, raw_mask, wall_now)
                    self._show(self.bottom_view, annotated)
                    self._sync_pair_cache = (mask_ts_ms, annotated.copy(), geom)
                    self._last_geom = geom
                    self._last_geom_wall_t = wall_now
                    self._record_last_seen_if_valid(geom, mask_ts_ms, wall_now)
                    self._update_mask_fps(mask_ts_ms, wall_now)
                    self._last_processed_mask_ts_ms = mask_ts_ms
                    self._last_mask_diag += f" frame={frame_mode} geom={geom.reason}"
                    self._update_aruco_info()
                else:
                    self._last_aruco_result = self.aruco_detector.no_frame(wall_now)
                    annotated, geom = self._annotate_mask_only(raw_mask)
                    self._show(self.bottom_view, annotated)
                    self._sync_pair_cache = (mask_ts_ms, annotated.copy(), geom)
                    self._last_geom = geom
                    self._last_geom_wall_t = wall_now
                    self._record_last_seen_if_valid(geom, mask_ts_ms, wall_now)
                    self._update_mask_fps(mask_ts_ms, wall_now)
                    self._last_processed_mask_ts_ms = mask_ts_ms
                    self._last_mask_diag += f" frame=none(mask-only) geom={geom.reason}"
                    self._update_aruco_info()
            elif self._sync_pair_cache is not None:
                _, ann_cached, _ = self._sync_pair_cache
                self._show(self.bottom_view, ann_cached)
        else:
            frame = self.bottom_cam.get_frame()
            if frame is not None:
                self._show(self.bottom_view, frame)

        geom_for_control = None
        geom_age = wall_now - self._last_geom_wall_t if self._last_geom_wall_t > 0 else 999.0
        if self._last_geom is not None and geom_age <= self._mask_stale_s:
            geom_for_control = self._last_geom

        current_pose = self._current_pose()
        self._record_pose_for_map(current_pose)

        if not self.tracking:
            self._update_info(geom_for_control, geom_age, None, None)
            self._refresh_map("IDLE", current_pose, None)
            return

        ret_cmd = None
        if self._pipe_geom_valid(geom_for_control):
            self.return_controller.reset()
            self._last_return_state = ""
            self._selected_return_observation = None
            cmd = self.controller.compute(geom_for_control, now=wall_now)
            state = self.controller.state
        else:
            visual_lost_s = (
                wall_now - self._last_valid_pipe_wall_t
                if self._last_valid_pipe_wall_t > 0
                else 999.0
            )
            return_obs = self._return_observation_target(wall_now, current_pose)
            self._selected_return_observation = return_obs
            if self.return_controller.maybe_start(
                return_obs,
                visual_lost_s=visual_lost_s,
                now_s=wall_now,
            ):
                self._selected_return_observation = self.return_controller.observation
                ret_cmd = self.return_controller.update(current_pose, now_s=wall_now)
                self._log_return_transition(ret_cmd)
                if ret_cmd.state == ReturnToLastSeenController.STATE_IDLE:
                    self._selected_return_observation = None
                    cmd = self.controller.compute(None, now=wall_now)
                    state = self.controller.state
                else:
                    self._send_return_yaw(ret_cmd)
                    cmd = ret_cmd.rc
                    state = ret_cmd.state
            else:
                self._log_return_transition(None)
                self._selected_return_observation = None
                cmd = self.controller.compute(None, now=wall_now)
                state = self.controller.state

        self._last_return_cmd = ret_cmd
        self.last_cmd = cmd
        self.cmd_pub.send(cmd)
        self.state_label.setText(state)
        self._update_info(geom_for_control, geom_age, cmd, ret_cmd)
        self._refresh_map(state, current_pose, ret_cmd)

    def _update_mask_fps(self, mask_ts_ms: float, wall_now: float) -> None:
        if self._last_mask_wall_t is not None:
            dt = wall_now - self._last_mask_wall_t
            if dt > 1e-6:
                fps = 1.0 / dt
                self._mask_fps_ema = fps if self._mask_fps_ema <= 0 else 0.15 * fps + 0.85 * self._mask_fps_ema
        self._last_mask_wall_t = wall_now
        del mask_ts_ms

    def _update_info(
        self,
        geom: PipeGeometryResult | None,
        geom_age: float,
        cmd: dict | None,
        ret_cmd: ReturnCommand | None = None,
    ) -> None:
        parts = [self._last_mask_diag]
        if geom is not None and geom.found:
            parts.append(f"lat={geom.lateral_error:+.2f}")
            parts.append(f"look={geom.lookahead_error:+.2f}")
            parts.append(f"ang={geom.heading_error_deg:+.1f}")
            parts.append(f"conf={geom.confidence:.2f}")
        else:
            parts.append("boru yok/stale")
        parts.append(f"age={geom_age:.2f}s")
        parts.append(f"mask_fps={self._mask_fps_ema:.1f}")
        if self._last_pipe_observation is not None:
            obs_age = max(0.0, time.monotonic() - self._last_pipe_observation.wall_time_s)
            parts.append(f"last_seen={obs_age:.1f}s hist={len(self._pipe_observation_history)}")
        if ret_cmd is not None:
            parts.append(
                f"return={ret_cmd.reason} d={ret_cmd.distance_m:.2f}m "
                f"herr={ret_cmd.heading_error_deg:+.0f}deg"
            )
        elif self._pose_error:
            parts.append(f"ekf={self._pose_error}")
        if cmd is not None:
            parts.append(f"yaw={cmd['yaw']} fwd={cmd['forward']} lat={cmd['lateral']}")
        self.info_label.setText("  |  ".join(parts))

    def _update_aruco_info(self) -> None:
        r = self._last_aruco_result
        if not r.available:
            self.aruco_label.setText(f"ArUco: devre disi ({r.reason})")
            return

        visible = ", ".join(str(marker_id) for marker_id in r.ids) if r.ids else "yok"
        order = " -> ".join(str(marker_id) for marker_id in r.read_order[-30:]) if r.read_order else "yok"
        self.aruco_label.setText(
            f"Gorunen IDler: {visible}  |  "
            f"Okuma sirasi: {order}  |  "
            f"aruco_age={r.age_s:.2f}s  |  "
            f"aruco_fps={r.fps:.1f}  |  "
            f"durum={r.reason}"
        )

    def _draw_geometry(self, frame: np.ndarray, geom: PipeGeometryResult) -> None:
        h, w = frame.shape[:2]
        red = (255, 60, 60)
        green = (40, 255, 110)
        yellow = (255, 220, 60)
        cyan = (60, 220, 255)
        blue = (80, 140, 255)
        gray = (130, 140, 150)

        cv2.line(frame, (w // 2, 0), (w // 2, h), red, 1)
        if geom is None:
            return

        if geom.bbox is not None:
            x, y, bw, bh = geom.bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), yellow, 1)
        if geom.centerline is not None:
            cv2.line(frame, geom.centerline[0], geom.centerline[1], green, 3)
        if geom.center_point is not None:
            cv2.circle(frame, geom.center_point, 7, yellow, -1)
        if geom.lookahead_point is not None:
            cv2.circle(frame, geom.lookahead_point, 7, cyan, -1)
            cv2.line(frame, (0, geom.lookahead_point[1]), (w, geom.lookahead_point[1]), gray, 1)
        for _, cx, cy, area in geom.slice_centroids:
            radius = max(4, min(12, int(area ** 0.5 * 0.08)))
            cv2.circle(frame, (int(round(cx)), int(cy)), radius, blue, -1)

        text = (
            f"{geom.reason} lat={geom.lateral_error:+.2f} "
            f"look={geom.lookahead_error:+.2f} ang={geom.heading_error_deg:+.1f} "
            f"conf={geom.confidence:.2f}"
        )
        cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, green, 1)

    def _show(self, label: QLabel, frame_rgb: np.ndarray) -> None:
        h, w = frame_rgb.shape[:2]
        dw, dh = label.width(), label.height()
        scale = min(dw / max(1, w), dh / max(1, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame_rgb, (nw, nh))
        qi = QImage(resized.data, nw, nh, nw * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi))

    def on_start(self):
        self.controller.reset()
        self.return_controller.reset()
        self._last_pipe_observation = None
        self._pipe_observation_history.clear()
        self._map_pose_trail.clear()
        self._map_pipe_trail.clear()
        self._selected_return_observation = None
        self._last_map_pose_ts_ms = None
        self._last_valid_pipe_wall_t = 0.0
        self._last_return_cmd = None
        self._last_return_state = ""
        self._last_geom = None
        self._last_geom_wall_t = 0.0
        self._last_processed_mask_ts_ms = None
        self._sync_pair_cache = None
        self._last_mask_wall_t = None
        self._mask_fps_ema = 0.0
        self.tracking = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.state_label.setText(self.controller.state)
        self._log("Takip baslatildi (v4: lateral + yaw ayrik kontrol)")

    def on_stop(self):
        self.tracking = False
        self.return_controller.reset()
        self._last_return_cmd = None
        self._last_return_state = ""
        self._selected_return_observation = None
        self.cmd_pub.send(self.controller.stop_cmd())
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.state_label.setText("IDLE")
        self.info_label.setText("")
        self._log("Takip durduruldu")

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.tracking = False
        self._pose_stop.set()
        if self._pose_thread is not None:
            self._pose_thread.join(timeout=1.0)
        if self._vehicle is not None:
            try:
                self._vehicle.close()
            except Exception:
                pass
        self.bottom_cam.stop()
        self.bottom_mask_reader.stop()
        event.accept()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipe Tracker GUI v4 (decoupled visual servo)")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID")
    args = parser.parse_args()

    print("=" * 60)
    print("PIPE TRACKER GUI v4 - DDS (decoupled visual servo)")
    print("=" * 60)
    print("Kamera: DDS FrameChunk (camera/bottom/frame)")
    print("Maske : DDS SegmentationMask (sam3/bottom/segmentation_mask)")
    print("Komut : motor_rc (lateral + yaw + forward)")
    print(f"DDS   : {os.environ.get('CYCLONEDDS_URI', '-')}")
    print("=" * 60)

    participant = DomainParticipant(domain_id=args.domain_id)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(21, 23, 27))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(32, 38, 46))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(34, 42, 51))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(215, 122, 97))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = PipeTrackerServoWindow(participant)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
