#!/usr/bin/env python3
"""
Pipe Tracking Algorithm - tauv-pipe based
Sadece algoritma: MaskProcessor + PipeController
GUI ve donanim bagimliligi yok, import edilebilir.

Gercek aracta: SAM3 mask -> process() -> compute() -> RC komut
Simulasyonda: OpenCV mask -> process() -> compute() -> RC komut
"""

import time
import math
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class ProcessResult:
    error: float | None = None
    binary_mask: np.ndarray | None = None
    total_area: int = 0
    slice_centroids: list = field(default_factory=list)
    weighted_cx: float = 0.0
    center_x: float = 64.0
    width: int = 128
    height: int = 128
    num_slices: int = 8
    pipe_angle_deg: float = 0.0
    curvature: float = 0.0
    turn_direction: int = 0
    coverage_ratio: float = 0.0
    pipe_continues: bool = True
    scan_hit_count: int = 0
    scan_rays: list = field(default_factory=list)  # [(x1,y1,x2,y2,hit), ...] for debug viz


class MaskProcessor:
    """Mask -> slice centroid + weighted error + pipe angle + curvature."""

    def __init__(self, num_slices=8, slice_weights=None, min_mask_area=500):
        self.num_slices = num_slices
        self.min_mask_area = min_mask_area
        if slice_weights is None:
            raw = [1.0 / (i + 1) for i in range(num_slices)]
            total = sum(raw)
            self.slice_weights = [w / total for w in raw]
        else:
            self.slice_weights = slice_weights

    def process(self, binary_mask):
        height, width = binary_mask.shape[:2]
        result = ProcessResult(
            width=width, height=height,
            center_x=width / 2.0, num_slices=self.num_slices,
        )
        result.binary_mask = binary_mask
        result.total_area = cv2.countNonZero(binary_mask)
        result.coverage_ratio = result.total_area / max(1, width * height)
        if result.total_area < self.min_mask_area:
            return result

        slice_h = height // self.num_slices
        weighted_sum = 0.0
        weight_sum = 0.0

        for i in range(self.num_slices):
            y_start = i * slice_h
            y_end = (i + 1) * slice_h if i < self.num_slices - 1 else height
            roi = binary_mask[y_start:y_end, :]
            area = cv2.countNonZero(roi)
            if area < 50:
                continue
            moments = cv2.moments(roi)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy_local = moments["m01"] / moments["m00"]
            w = self.slice_weights[i]
            weighted_sum += w * cx
            weight_sum += w
            result.slice_centroids.append((i, cx, int(y_start + cy_local), area))

        if weight_sum == 0:
            return result

        result.weighted_cx = weighted_sum / weight_sum
        error = (result.weighted_cx - result.center_x) / result.center_x
        result.error = max(-1.0, min(1.0, error))
        self._estimate_angle_and_curvature(result)
        self._check_continuation(result)
        return result

    def _check_continuation(self, result):
        """Borunun ileriye devam edip etmedigini tarama isinlari ile kontrol et."""
        centroids = result.slice_centroids
        mask = result.binary_mask
        if mask is None or len(centroids) < 3:
            result.pipe_continues = True
            return

        h, w = mask.shape[:2]
        sorted_c = sorted(centroids, key=lambda c: c[0])
        pts = np.array([(c[1], c[2]) for c in sorted_c], dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        tip_cx, tip_cy = pts[0][0], pts[0][1]
        tail_cx, tail_cy = pts[-1][0], pts[-1][1]

        fwd_dx = tip_cx - tail_cx
        fwd_dy = tip_cy - tail_cy
        base_angle = math.atan2(fwd_dy, fwd_dx)

        scan_distances = [40, 70, 110]
        theta_range = 60
        theta_step = 5
        hit_count = 0
        total_valid = 0
        rays = []

        for delta_deg in range(-theta_range, theta_range + 1, theta_step):
            angle = base_angle + math.radians(delta_deg)
            ray_hit = False
            furthest_x, furthest_y = int(tip_cx), int(tip_cy)

            for dist in scan_distances:
                end_x = tip_cx + dist * math.cos(angle)
                end_y = tip_cy + dist * math.sin(angle)
                cx_i = int(max(0, min(w - 1, end_x)))
                cy_i = int(max(0, min(h - 1, end_y)))
                furthest_x, furthest_y = cx_i, cy_i

                in_bounds = (0 <= end_x < w and 0 <= end_y < h)
                if in_bounds:
                    total_valid += 1

                r = 6
                x1c, x2c = max(0, cx_i - r), min(w, cx_i + r)
                y1c, y2c = max(0, cy_i - r), min(h, cy_i + r)
                if x2c > x1c and y2c > y1c:
                    roi = mask[y1c:y2c, x1c:x2c]
                    if cv2.countNonZero(roi) > 0:
                        ray_hit = True

            if ray_hit:
                hit_count += 1
            rays.append((int(tip_cx), int(tip_cy), furthest_x, furthest_y, ray_hit))

        result.scan_rays = rays
        result.scan_hit_count = hit_count

        oob_rays = sum(1 for _, _, x2, y2, _ in rays
                       if x2 <= 1 or x2 >= w - 2 or y2 <= 1 or y2 >= h - 2)
        oob_ratio = oob_rays / max(1, len(rays))

        if oob_ratio > 0.5:
            result.pipe_continues = True
        else:
            result.pipe_continues = hit_count > 0

    def _estimate_angle_and_curvature(self, result):
        centroids = result.slice_centroids
        if len(centroids) < 2:
            return
        centroids_sorted = sorted(centroids, key=lambda c: c[0])
        points = [(c[1], c[2]) for c in centroids_sorted]
        n = len(points)
        mid = max(1, n // 2)
        bot_cx = sum(p[0] for p in points[mid:]) / len(points[mid:])
        top_cx = sum(p[0] for p in points[:mid]) / len(points[:mid])
        bot_cy = sum(p[1] for p in points[mid:]) / len(points[mid:])
        top_cy = sum(p[1] for p in points[:mid]) / len(points[:mid])
        dx = top_cx - bot_cx
        dy = top_cy - bot_cy
        if abs(dy) > 1e-6:
            result.pipe_angle_deg = math.degrees(math.atan2(dx, -dy))
        else:
            result.pipe_angle_deg = 90.0 if dx > 0 else -90.0
        if len(points) >= 3:
            angles = []
            for j in range(len(points) - 1):
                angles.append(math.atan2(
                    points[j+1][0] - points[j][0],
                    points[j+1][1] - points[j][1],
                ))
            peak = 0.0
            for j in range(len(angles) - 1):
                d = (angles[j+1] - angles[j] + math.pi) % (2 * math.pi) - math.pi
                if abs(d) > abs(peak):
                    peak = d
            result.curvature = peak
            if abs(peak) > 0.05:
                result.turn_direction = 1 if peak > 0 else -1


class PipeController:
    """Alt kamera mask -> yaw + forward komut.
    State machine: FOLLOW / TURNING / COAST
    Throttle'a dokunmaz - depth hold korur.
    """

    STATE_FOLLOW = "FOLLOW"
    STATE_TURNING = "TURNING"
    STATE_COAST = "COAST"
    STATE_REVERSE = "REVERSE"
    STATE_REACQUIRE = "REACQUIRE"
    STATE_COMPLETE = "COMPLETE"

    def __init__(
        self,
        neutral_pwm=1500,
        forward_pwm=200,
        max_yaw_pwm=200,
        kp_yaw=150.0,
        ki_yaw=200.0,
        ema_alpha=0.6,
        turn_curvature_thresh=0.03,
        turn_align_thresh_deg=10.0,
        turn_forward_pwm=150,
        turn_yaw_boost=3.0,
        coast_timeout=1.5,
        coast_yaw_decay=0.55,
        reverse_yaw_pwm=150,
        reacquire_forward_pwm=100,
        reacquire_duration_s=3.0,
    ):
        self.neutral = neutral_pwm
        self.forward_pwm = forward_pwm
        self.max_yaw_pwm = max_yaw_pwm
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.ema_alpha = ema_alpha
        self.turn_curvature_thresh = turn_curvature_thresh
        self.turn_align_thresh_deg = turn_align_thresh_deg
        self.turn_forward_pwm = turn_forward_pwm
        self.turn_yaw_boost = turn_yaw_boost
        self.coast_timeout = coast_timeout
        self.coast_yaw_decay = coast_yaw_decay
        self.reverse_yaw_pwm = reverse_yaw_pwm
        # 180 donus sonrasi sabit ileri sureci. Mask gelmese de
        # reacquire_duration_s boyunca reacquire_forward_pwm offset ile
        # ileri gider, sonra FOLLOW'a doner ki coast tekrar tetiklensin.
        self.reacquire_forward_pwm = reacquire_forward_pwm
        self.reacquire_duration_s = reacquire_duration_s

        self._state = self.STATE_FOLLOW
        self._prev_err = None
        self._integral = 0.0
        self._last_yaw = 0.0
        self._last_fwd = 0
        self._last_valid_t = 0.0
        # pass_count ve COMPLETE state artik kullanilmiyor; geriye uyumluluk
        # icin alanlar duruyor ama hicbir yerde set edilmiyor. Gorev sonsuz
        # dongu olarak akiyor, manuel DURDUR ile sonlandirilir.
        self._pass_count = 0
        self._continues_history = []
        self._reverse_target_hdg = 0.0
        self._reacquire_start = 0.0

    @property
    def state(self):
        return self._state

    @property
    def pass_count(self):
        return self._pass_count

    def reset(self):
        self._state = self.STATE_FOLLOW
        self._prev_err = None
        self._integral = 0.0
        self._last_yaw = 0.0
        self._last_fwd = 0
        self._last_valid_t = time.monotonic()
        self._pass_count = 0
        self._continues_history = []
        self._reverse_target_hdg = 0.0
        self._reacquire_start = 0.0

    def update_params(self, **kwargs):
        """Canli parametre guncelleme (GUI slider'larindan)."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _ema(self, v, alpha=None):
        a = alpha if alpha is not None else self.ema_alpha
        if self._prev_err is None:
            self._prev_err = v
            return v
        s = a * v + (1 - a) * self._prev_err
        self._prev_err = s
        return s

    def _clamp_yaw(self, v):
        return max(-self.max_yaw_pwm, min(self.max_yaw_pwm, v))

    def _make_cmd(self, yaw_offset, fwd_offset):
        return {
            "pitch": self.neutral,
            "roll": self.neutral,
            "throttle": self.neutral,
            "yaw": self.neutral + int(yaw_offset),
            "forward": self.neutral + int(fwd_offset),
            "lateral": self.neutral,
        }

    def stop_cmd(self):
        return self._make_cmd(0, 0)

    def compute(self, result: ProcessResult, heading_deg: float = 0.0) -> dict:
        """ProcessResult + heading'ten RC komut dict'i uret."""
        if self._state == self.STATE_COMPLETE:
            return self.stop_cmd()

        if self._state == self.STATE_REVERSE:
            return self._reverse_cmd(heading_deg)

        if self._state == self.STATE_REACQUIRE:
            return self._reacquire_cmd(result)

        if result is not None and result.error is not None:
            self._continues_history.append(result.pipe_continues)
            if len(self._continues_history) > 10:
                self._continues_history = self._continues_history[-10:]

        if result is None or result.error is None:
            return self._coast(heading_deg)

        self._last_valid_t = time.monotonic()
        ac = abs(result.curvature)

        if self._state == self.STATE_FOLLOW and ac > self.turn_curvature_thresh:
            self._state = self.STATE_TURNING
        elif self._state == self.STATE_TURNING:
            if ac < self.turn_curvature_thresh / 2.0 and abs(result.pipe_angle_deg) < self.turn_align_thresh_deg:
                self._state = self.STATE_FOLLOW
        elif self._state == self.STATE_COAST:
            self._state = self.STATE_TURNING if ac > self.turn_curvature_thresh else self.STATE_FOLLOW

        if self._state == self.STATE_TURNING:
            return self._turning_cmd(result)
        return self._follow_cmd(result)

    def _pi_yaw(self, error):
        """PI controller: P merkezler, I sabit offset'i duzeltir."""
        smoothed = self._ema(error)
        self._integral += smoothed
        max_integral = self.max_yaw_pwm / max(0.1, self.ki_yaw)
        self._integral = max(-max_integral, min(max_integral, self._integral))
        if abs(smoothed) < 0.05:
            self._integral *= 0.9
        return self.kp_yaw * smoothed + self.ki_yaw * self._integral

    def _follow_cmd(self, r):
        yaw = self._clamp_yaw(self._pi_yaw(r.error))
        abs_err = abs(r.error)
        fwd = self.forward_pwm * max(0.2, 1.0 - abs_err * 1.0)
        self._last_yaw = yaw
        self._last_fwd = int(fwd)
        return self._make_cmd(yaw, int(fwd))

    def _turning_cmd(self, r):
        angle_yaw = (r.pipe_angle_deg / 90.0) * self.max_yaw_pwm * self.turn_yaw_boost
        error_yaw = self._pi_yaw(r.error) * 0.3
        yaw = self._clamp_yaw(angle_yaw + error_yaw)
        abs_err = abs(r.error)
        fwd = self.turn_forward_pwm * max(0.2, 1.0 - abs_err * 1.0)
        self._last_yaw = yaw
        self._last_fwd = int(fwd)
        return self._make_cmd(yaw, int(fwd))

    def _coast(self, heading_deg=0.0):
        self._state = self.STATE_COAST
        self._integral *= 0.8
        self._continues_history.append(False)
        if len(self._continues_history) > 10:
            self._continues_history = self._continues_history[-10:]
        elapsed = time.monotonic() - self._last_valid_t if self._last_valid_t > 0 else 999
        if elapsed > self.coast_timeout:
            self._prev_err = None
            self._integral = 0.0
            self._last_yaw = 0.0

            recent = self._continues_history[-5:] if self._continues_history else [True]
            false_count = sum(1 for v in recent if not v)
            pipe_ended = false_count >= 3

            if pipe_ended:
                # Her bitiste 180 donus + reacquire dongusu. Pass count yok,
                # COMPLETE'e gecilmiyor; manuel DURDUR ile sonlandirilir.
                self._reverse_target_hdg = (heading_deg + 180.0) % 360.0
                self._state = self.STATE_REVERSE
                self._continues_history = []
                return self._make_cmd(0, 0)
            else:
                self._state = self.STATE_FOLLOW
                self._last_valid_t = time.monotonic()
                return self._make_cmd(0, 0)

        self._last_yaw *= self.coast_yaw_decay
        fwd = max(self.turn_forward_pwm, self._last_fwd // 2)
        return self._make_cmd(self._clamp_yaw(self._last_yaw), fwd)

    def _reverse_cmd(self, heading_deg):
        """180 derece donus: hedefe yaklastikca yavasla."""
        diff = self._reverse_target_hdg - heading_deg
        diff = (diff + 180) % 360 - 180

        if abs(diff) < 10.0:
            self._state = self.STATE_REACQUIRE
            self._reacquire_start = time.monotonic()
            self._prev_err = None
            self._integral = 0.0
            self._continues_history = []
            return self._make_cmd(0, self.reacquire_forward_pwm)

        ratio = min(1.0, abs(diff) / 90.0)
        yaw_speed = max(40, int(self.reverse_yaw_pwm * ratio))
        yaw_dir = yaw_speed if diff > 0 else -yaw_speed
        return self._make_cmd(yaw_dir, 0)

    def _reacquire_cmd(self, result):
        """180 donus sonrasi: reacquire_duration_s boyunca sabit ileri @
        reacquire_forward_pwm. Sure dolunca veya mask gelirse FOLLOW'a doner.
        Sure doldugu halde mask gelmediyse coast yeniden tetiklenip dongu
        bastan basliyor olur."""
        elapsed = time.monotonic() - self._reacquire_start

        if result is not None and result.error is not None:
            self._state = self.STATE_FOLLOW
            self._last_valid_t = time.monotonic()
            self._continues_history = []
            return self._follow_cmd(result)

        if elapsed > self.reacquire_duration_s:
            self._state = self.STATE_FOLLOW
            self._last_valid_t = time.monotonic()
            return self._make_cmd(0, 0)

        return self._make_cmd(0, self.reacquire_forward_pwm)
