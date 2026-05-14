"""ArUco detection helpers for Pipe Tracker GUI v4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class ArucoMarker:
    marker_id: int
    corners: np.ndarray
    center_px: tuple[float, float]
    area_frac: float


@dataclass
class ArucoResult:
    available: bool
    detected: bool
    ids: list[int] = field(default_factory=list)
    markers: list[ArucoMarker] = field(default_factory=list)
    read_order: list[int] = field(default_factory=list)
    age_s: float = 999.0
    fps: float = 0.0
    reason: str = ""


class ArucoMarkerDetector:
    """Detect DICT_ARUCO_ORIGINAL markers and track first-seen order."""

    def __init__(self, history_size: int = 30):
        self._history: deque[int] = deque(maxlen=int(history_size))
        self._visible_ids: set[int] = set()
        self._last_seen_t = 0.0
        self._last_detect_t = 0.0
        self._fps_ema = 0.0
        self._last_markers: list[ArucoMarker] = []
        self._last_reason = "bekleniyor"

        self.available = False
        self.reason = ""
        self._detector = None

        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            self.reason = "cv2.aruco yok"
            return
        if not hasattr(aruco, "DICT_ARUCO_ORIGINAL"):
            self.reason = "DICT_ARUCO_ORIGINAL yok"
            return
        if not hasattr(aruco, "ArucoDetector") or not hasattr(aruco, "DetectorParameters"):
            self.reason = "ArucoDetector yok"
            return

        dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters()
        self._detector = aruco.ArucoDetector(dictionary, parameters)
        self.available = True
        self.reason = "ok"

    def detect(self, frame_rgb: np.ndarray, now: float | None = None) -> ArucoResult:
        t = time.monotonic() if now is None else float(now)
        self._update_fps(t)

        if not self.available or self._detector is None:
            self._last_markers = []
            self._visible_ids = set()
            self._last_reason = self.reason
            return self._result(t)

        if frame_rgb is None or frame_rgb.size == 0:
            self._last_markers = []
            self._visible_ids = set()
            self._last_reason = "frame_yok"
            return self._result(t)

        try:
            if frame_rgb.ndim == 3:
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame_rgb.astype(np.uint8)
            corners, ids, _ = self._detector.detectMarkers(gray)
        except Exception as exc:
            self._last_markers = []
            self._visible_ids = set()
            self._last_reason = f"hata:{exc}"
            return self._result(t)

        if ids is None or len(ids) == 0:
            self._last_markers = []
            self._visible_ids = set()
            self._last_reason = "yok"
            return self._result(t)

        h, w = frame_rgb.shape[:2]
        img_area = float(max(1, w * h))
        markers: list[ArucoMarker] = []
        current_ids: list[int] = []
        for corner_arr, raw_id in zip(corners, ids.flatten()):
            marker_id = int(raw_id)
            pts = corner_arr.reshape(4, 2).astype(np.float32)
            center = pts.mean(axis=0)
            area_frac = float(cv2.contourArea(pts) / img_area)
            markers.append(
                ArucoMarker(
                    marker_id=marker_id,
                    corners=pts,
                    center_px=(float(center[0]), float(center[1])),
                    area_frac=area_frac,
                )
            )
            current_ids.append(marker_id)

        appended_this_frame: set[int] = set()
        for marker_id in current_ids:
            if marker_id not in self._visible_ids and marker_id not in appended_this_frame:
                self._history.append(marker_id)
                appended_this_frame.add(marker_id)

        self._visible_ids = set(current_ids)
        self._last_markers = markers
        self._last_seen_t = t
        self._last_reason = "ok"
        return self._result(t)

    def no_frame(self, now: float | None = None) -> ArucoResult:
        t = time.monotonic() if now is None else float(now)
        self._last_markers = []
        self._visible_ids = set()
        self._last_reason = "frame_yok"
        return self._result(t)

    def status(self, now: float | None = None) -> ArucoResult:
        t = time.monotonic() if now is None else float(now)
        return self._result(t)

    def annotate(self, frame_rgb: np.ndarray, result: ArucoResult) -> np.ndarray:
        out = frame_rgb.copy()
        if not result.available or not result.markers:
            return out

        for marker in result.markers:
            pts = np.round(marker.corners).astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=(60, 255, 120), thickness=2)
            cx, cy = marker.center_px
            center = (int(round(cx)), int(round(cy)))
            cv2.circle(out, center, 5, (255, 255, 255), -1)
            x0 = int(np.min(pts[:, 0]))
            y0 = int(np.min(pts[:, 1]))
            label = f"ArUco {marker.marker_id}"
            cv2.putText(
                out,
                label,
                (max(0, x0), max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 255, 120),
                2,
            )
        return out

    def _update_fps(self, t: float) -> None:
        if self._last_detect_t > 0:
            dt = t - self._last_detect_t
            if dt > 1e-6:
                fps = 1.0 / dt
                self._fps_ema = fps if self._fps_ema <= 0 else 0.2 * fps + 0.8 * self._fps_ema
        self._last_detect_t = t

    def _result(self, t: float) -> ArucoResult:
        age = t - self._last_seen_t if self._last_seen_t > 0 else 999.0
        ids = [marker.marker_id for marker in self._last_markers]
        return ArucoResult(
            available=self.available,
            detected=bool(self._last_markers),
            ids=ids,
            markers=list(self._last_markers),
            read_order=list(self._history),
            age_s=float(age),
            fps=float(self._fps_ema),
            reason=self._last_reason if self.available else self.reason,
        )
