#!/usr/bin/env python3
"""
Pipe Tracking Algorithm - tauv-pipe based
Sadece algoritma: MaskProcessor + PipeController (FOLLOW / NO_PIPE tek davranis).
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

    def __init__(
        self,
        num_slices=8,
        slice_weights=None,
        min_mask_area=500,
        scan_for_pipe_end=True,
    ):
        self.num_slices = num_slices
        self.min_mask_area = min_mask_area
        self.scan_for_pipe_end = scan_for_pipe_end
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
        if self.scan_for_pipe_end:
            self._check_continuation(result)
        else:
            result.pipe_continues = True
            result.scan_hit_count = 0
            result.scan_rays = []
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
    """Alt kamera mask -> yaw + forward. FOLLOW / NO_PIPE.
    Yaw: maske hatasi EMA ile yumusatilir, PID + cikis slew (osilasyonu keser).
    Ileri: |pipe_angle| > esik ise viraj PWM, degilse duz PWM.
    """

    _STAB_EMA_ALPHA = 0.38
    _STAB_YAW_SLEW_PWM_PER_S = 1300.0

    STATE_FOLLOW = "FOLLOW"
    STATE_NO_PIPE = "NO_PIPE"
    # Eski GUI/import uyumlulugu (compute artik bu durumlara gecmez):
    STATE_TURNING = "TURNING"
    STATE_COAST = "COAST"
    STATE_REVERSE = "REVERSE"
    STATE_REACQUIRE = "REACQUIRE"
    STATE_COMPLETE = "COMPLETE"

    def __init__(
        self,
        neutral_pwm=1500,
        kp_yaw=40.0,
        ki_yaw=30.0,
        kd_yaw=10.0,
        max_yaw_pwm=100,
        forward_pwm=130,
        forward_pwm_curve=70,
        curve_angle_thresh_deg=20.0,
    ):
        self.neutral = neutral_pwm
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw
        self.max_yaw_pwm = max_yaw_pwm
        self.forward_pwm = forward_pwm
        self.forward_pwm_curve = forward_pwm_curve
        self.curve_angle_thresh_deg = curve_angle_thresh_deg

        self._mask_ok = False
        self._integral = 0.0
        self._ema_err = None
        self._last_ema_err = None
        self._last_control_t = 0.0
        self._prev_cmd_yaw = 0.0
        self._last_yaw = 0.0
        self._last_fwd = 0

    @property
    def state(self):
        return self.STATE_FOLLOW if self._mask_ok else self.STATE_NO_PIPE

    @property
    def pass_count(self):
        return 0

    def reset(self):
        self._mask_ok = False
        self._integral = 0.0
        self._ema_err = None
        self._last_ema_err = None
        self._last_control_t = 0.0
        self._prev_cmd_yaw = 0.0
        self._last_yaw = 0.0
        self._last_fwd = 0

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _clamp_yaw(self, v):
        return max(-self.max_yaw_pwm, min(self.max_yaw_pwm, v))

    def _measure_dt(self) -> float:
        now = time.monotonic()
        if self._last_control_t <= 0:
            dt = 0.1
        else:
            dt = max(0.02, min(0.5, now - self._last_control_t))
        self._last_control_t = now
        return dt

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
        del heading_deg
        if result is None or result.error is None:
            self._mask_ok = False
            self._integral *= 0.85
            self._ema_err = None
            self._last_ema_err = None
            self._prev_cmd_yaw = 0.0
            self._last_control_t = 0.0
            return self.stop_cmd()

        self._mask_ok = True
        return self._follow_cmd(result)

    def _pid_yaw(self, error_raw: float, dt: float) -> float:
        a = self._STAB_EMA_ALPHA
        if self._ema_err is None:
            self._ema_err = error_raw
        else:
            self._ema_err = a * error_raw + (1.0 - a) * self._ema_err
        es = self._ema_err

        self._integral += es
        max_integral = self.max_yaw_pwm / max(0.1, self.ki_yaw)
        self._integral = max(-max_integral, min(max_integral, self._integral))
        if abs(es) < 0.04:
            self._integral *= 0.88
        if es * self._integral < 0 and abs(es) > 0.02:
            self._integral *= 0.65

        dedt = 0.0
        if self._last_ema_err is not None and dt > 1e-6:
            dedt = (es - self._last_ema_err) / dt
        self._last_ema_err = es

        return self.kp_yaw * es + self.ki_yaw * self._integral + self.kd_yaw * dedt

    def _forward_offset(self, r: ProcessResult) -> int:
        if abs(r.pipe_angle_deg) > self.curve_angle_thresh_deg:
            return int(self.forward_pwm_curve)
        return int(self.forward_pwm)

    def _slew_yaw_cmd(self, desired: float, dt: float) -> float:
        lim = self._STAB_YAW_SLEW_PWM_PER_S * dt
        p = self._prev_cmd_yaw
        d = desired - p
        if d > lim:
            desired = p + lim
        elif d < -lim:
            desired = p - lim
        self._prev_cmd_yaw = desired
        return desired

    def _follow_cmd(self, r):
        dt = self._measure_dt()
        yaw_pid = self._clamp_yaw(self._pid_yaw(r.error, dt))
        yaw = self._slew_yaw_cmd(yaw_pid, dt)
        fwd = self._forward_offset(r)
        self._last_yaw = yaw
        self._last_fwd = fwd
        return self._make_cmd(yaw, fwd)



class PipeControllerFollowPWM:
    """Sadece FOLLOW: maske hatasina PID yaw (Kp+Ki+Kd) + ileri PWM. Diger state'ler yok."""

    STATE_FOLLOW = "FOLLOW"

    def __init__(
        self,
        neutral_pwm=1500,
        forward_pwm=200,
        max_yaw_pwm=200,
        kp_yaw=150.0,
        ki_yaw=200.0,
        kd_yaw=0.0,
        ema_alpha=0.6,
    ):
        self.neutral = neutral_pwm
        self.forward_pwm = forward_pwm
        self.max_yaw_pwm = max_yaw_pwm
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw
        self.ema_alpha = ema_alpha

        self._prev_err = None
        self._integral = 0.0
        self._last_err_raw: float | None = None

    @property
    def state(self):
        return self.STATE_FOLLOW

    @property
    def pass_count(self):
        return 0

    def reset(self):
        self._prev_err = None
        self._integral = 0.0
        self._last_err_raw = None

    def update_params(self, **kwargs):
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

    def _pid_yaw(self, error: float) -> float:
        smoothed = self._ema(error)
        self._integral += smoothed
        max_integral = self.max_yaw_pwm / max(0.1, self.ki_yaw)
        self._integral = max(-max_integral, min(max_integral, self._integral))
        if abs(smoothed) < 0.05:
            self._integral *= 0.9

        deriv = 0.0
        if self._last_err_raw is not None:
            deriv = error - self._last_err_raw
        self._last_err_raw = error

        return self.kp_yaw * smoothed + self.ki_yaw * self._integral + self.kd_yaw * deriv

    def compute(self, result: ProcessResult, heading_deg: float = 0.0) -> dict:
        del heading_deg
        if result is None or result.error is None:
            self._integral *= 0.5
            self._last_err_raw = None
            return self.stop_cmd()

        yaw = self._clamp_yaw(self._pid_yaw(result.error))
        abs_err = abs(result.error)
        fwd = self.forward_pwm * max(0.2, 1.0 - abs_err * 1.0)
        return self._make_cmd(yaw, int(fwd))


def _wrap_deg(d: float) -> float:
    return d % 360.0


class PipeControllerAttitude:
    """Sadece boru takibi (tek mod): maske hatasi -> hedef yaw (NED derece) + ileri PWM.

    RC yaw PID yok; yaw `set_target_attitude` ile verilir. Boru bitimi / coast / viraj /
    180 donus / yeniden yakalama state machine'i yok — boru yokken yaw mevcut heading'de
    kalir, ileri `lost_forward_pwm` ile ayarlanir (varsayilan 0).

    Vehicle.attitude.yaw GUI'de dereceye cevrilir (`heading_deg`).
    """

    STATE_FOLLOW = "FOLLOW"

    def __init__(
        self,
        neutral_pwm=1500,
        forward_pwm=200,
        heading_gain_deg=35.0,
        max_heading_step_deg=10.0,
        ema_alpha=0.6,
        lost_forward_pwm=0,
    ):
        self.neutral = neutral_pwm
        self.forward_pwm = forward_pwm
        self.heading_gain_deg = heading_gain_deg
        self.max_heading_step_deg = max_heading_step_deg
        self.ema_alpha = ema_alpha
        self.lost_forward_pwm = int(lost_forward_pwm)

        self._prev_err = None
        self._last_target_yaw_deg = 0.0

    @property
    def state(self):
        return self.STATE_FOLLOW

    @property
    def pass_count(self):
        return 0

    def reset(self):
        self._prev_err = None
        self._last_target_yaw_deg = 0.0

    def update_params(self, **kwargs):
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

    def _clamp_step(self, delta_deg: float) -> float:
        m = self.max_heading_step_deg
        return max(-m, min(m, delta_deg))

    def _motor_rc(self, fwd_offset: int) -> dict:
        return {
            "pitch": self.neutral,
            "roll": self.neutral,
            "throttle": self.neutral,
            "yaw": self.neutral,
            "forward": self.neutral + int(fwd_offset),
            "lateral": self.neutral,
        }

    def stop_cmd(self) -> dict:
        out = self._motor_rc(0)
        out["target_yaw_deg"] = None
        out["send_attitude"] = False
        return out

    def compute(self, result: ProcessResult, heading_deg: float = 0.0) -> dict:
        if result is None or result.error is None:
            target = _wrap_deg(heading_deg)
            self._last_target_yaw_deg = target
            out = self._motor_rc(self.lost_forward_pwm)
            out["target_yaw_deg"] = float(target)
            out["send_attitude"] = True
            return out

        return self._follow_cmd(result, heading_deg)

    def _emit(self, fwd_offset: int, target_yaw_deg: float) -> dict:
        self._last_target_yaw_deg = target_yaw_deg
        out = self._motor_rc(fwd_offset)
        out["target_yaw_deg"] = float(target_yaw_deg)
        out["send_attitude"] = True
        return out

    def _follow_cmd(self, r: ProcessResult, heading_deg: float) -> dict:
        smoothed = self._ema(r.error)
        delta = self.heading_gain_deg * smoothed
        delta = self._clamp_step(delta)
        target = _wrap_deg(heading_deg + delta)
        abs_err = abs(r.error)
        fwd = self.forward_pwm * max(0.2, 1.0 - abs_err * 1.0)
        return self._emit(int(fwd), target)


@dataclass
class PipeGeometryResult:
    """Maskeden ayrik boru takip gozlemi.

    Sign convention:
    - lateral_error > 0: boru merkezi goruntude sagda.
    - heading_error_deg > 0: borunun ileri ucu goruntude saga dogru.
    """

    found: bool = False
    reason: str = ""
    binary_mask: np.ndarray | None = None
    component_mask: np.ndarray | None = None
    width: int = 0
    height: int = 0
    area: int = 0
    coverage_ratio: float = 0.0
    confidence: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    lateral_error: float = 0.0
    lookahead_error: float = 0.0
    heading_error_deg: float = 0.0
    curvature: float = 0.0
    elongation: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    slice_centroids: list = field(default_factory=list)
    centerline: tuple[tuple[int, int], tuple[int, int]] | None = None
    lookahead_point: tuple[int, int] | None = None
    center_point: tuple[int, int] | None = None


class PipeGeometryProcessor:
    """SAM maskesi -> centerline, cross-track ve heading gozlemi.

    Bu sinif motor komutu uretmez; sadece goruntu uzayinda test edilebilir
    boru geometrisi cikarir. PID yerine ayrik lateral/yaw kontrolu kurmak icin
    `PipeVisualServoController` ile kullanilir.
    """

    def __init__(
        self,
        num_slices: int = 10,
        min_area_ratio: float = 0.002,
        min_slice_area_ratio: float = 0.00015,
        morph_kernel_ratio: float = 0.006,
        lookahead_y_ratio: float = 0.25,
        center_y_ratio: float = 0.55,
    ):
        self.num_slices = int(num_slices)
        self.min_area_ratio = float(min_area_ratio)
        self.min_slice_area_ratio = float(min_slice_area_ratio)
        self.morph_kernel_ratio = float(morph_kernel_ratio)
        self.lookahead_y_ratio = float(lookahead_y_ratio)
        self.center_y_ratio = float(center_y_ratio)

    def process(self, mask: np.ndarray | None) -> PipeGeometryResult:
        if mask is None:
            return PipeGeometryResult(found=False, reason="no_mask")

        h, w = mask.shape[:2]
        out = PipeGeometryResult(width=w, height=h)
        if h <= 0 or w <= 0:
            out.reason = "empty_shape"
            return out

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.size and int(mask_u8.max()) <= 1:
            binary = (mask_u8 > 0).astype(np.uint8) * 255
        else:
            _, binary = cv2.threshold(mask_u8, 128, 255, cv2.THRESH_BINARY)
        binary = self._clean_mask(binary)
        out.binary_mask = binary

        component = self._largest_component(binary)
        if component is None:
            out.reason = "no_component"
            return out

        comp_mask, area, bbox = component
        out.component_mask = comp_mask
        out.area = int(area)
        out.bbox = bbox
        out.coverage_ratio = area / float(max(1, w * h))
        min_area = max(20, int(self.min_area_ratio * w * h))
        if area < min_area:
            out.reason = f"small_area:{area}<{min_area}"
            return out

        moments = cv2.moments(comp_mask)
        if moments["m00"] <= 0:
            out.reason = "zero_moment"
            return out
        out.centroid_x = float(moments["m10"] / moments["m00"])
        out.centroid_y = float(moments["m01"] / moments["m00"])

        slices = self._slice_centroids(comp_mask)
        out.slice_centroids = slices
        if len(slices) < 2:
            out.reason = "too_few_slices"
            return out

        ys = np.array([p[2] for p in slices], dtype=np.float64)
        xs = np.array([p[1] for p in slices], dtype=np.float64)
        weights = np.array([max(1, p[3]) for p in slices], dtype=np.float64)
        try:
            slope, intercept = np.polyfit(ys, xs, deg=1, w=np.sqrt(weights))
        except Exception:
            out.reason = "line_fit_failed"
            return out

        y_top = 0.0
        y_bottom = float(h - 1)
        y_center = float(np.clip(self.center_y_ratio, 0.0, 1.0) * (h - 1))
        y_look = float(np.clip(self.lookahead_y_ratio, 0.0, 1.0) * (h - 1))

        x_top = float(slope * y_top + intercept)
        x_bottom = float(slope * y_bottom + intercept)
        x_center = float(slope * y_center + intercept)
        x_look = float(slope * y_look + intercept)
        cx = (w - 1) / 2.0
        half_w = max(1.0, w / 2.0)

        out.lateral_error = self._clip_unit((x_center - cx) / half_w)
        out.lookahead_error = self._clip_unit((x_look - cx) / half_w)
        out.heading_error_deg = float(math.degrees(math.atan2(x_top - x_bottom, y_bottom - y_top)))
        out.centerline = (
            (int(round(np.clip(x_bottom, 0, w - 1))), int(round(y_bottom))),
            (int(round(np.clip(x_top, 0, w - 1))), int(round(y_top))),
        )
        out.center_point = (
            int(round(np.clip(x_center, 0, w - 1))),
            int(round(y_center)),
        )
        out.lookahead_point = (
            int(round(np.clip(x_look, 0, w - 1))),
            int(round(y_look)),
        )

        out.curvature = self._estimate_curvature(slices)
        out.elongation = self._estimate_elongation(comp_mask)
        slice_score = min(1.0, len(slices) / max(1.0, self.num_slices * 0.75))
        area_score = min(1.0, out.coverage_ratio / max(self.min_area_ratio * 4.0, 1e-6))
        elong_score = min(1.0, out.elongation / 5.0)
        out.confidence = float(0.45 * slice_score + 0.35 * area_score + 0.20 * elong_score)
        out.found = True
        out.reason = "ok"
        return out

    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        h, w = binary.shape[:2]
        k = int(round(min(h, w) * self.morph_kernel_ratio))
        k = max(3, k | 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    @staticmethod
    def _largest_component(binary: np.ndarray):
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas) + 1)
        area = int(stats[best_label, cv2.CC_STAT_AREA])
        x = int(stats[best_label, cv2.CC_STAT_LEFT])
        y = int(stats[best_label, cv2.CC_STAT_TOP])
        ww = int(stats[best_label, cv2.CC_STAT_WIDTH])
        hh = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        comp_mask = np.zeros_like(binary)
        comp_mask[labels == best_label] = 255
        return comp_mask, area, (x, y, ww, hh)

    def _slice_centroids(self, mask: np.ndarray) -> list:
        h, w = mask.shape[:2]
        slice_h = max(1, h // max(1, self.num_slices))
        min_area = max(8, int(self.min_slice_area_ratio * w * h))
        centroids = []
        for i in range(self.num_slices):
            y0 = i * slice_h
            y1 = (i + 1) * slice_h if i < self.num_slices - 1 else h
            roi = mask[y0:y1, :]
            area = int(cv2.countNonZero(roi))
            if area < min_area:
                continue
            m = cv2.moments(roi)
            if m["m00"] <= 0:
                continue
            x = float(m["m10"] / m["m00"])
            y = int(round(y0 + m["m01"] / m["m00"]))
            centroids.append((i, x, y, area))
        return centroids

    @staticmethod
    def _estimate_curvature(slices: list) -> float:
        if len(slices) < 3:
            return 0.0
        pts = [(float(c[1]), float(c[2])) for c in sorted(slices, key=lambda item: item[2])]
        angles = []
        for a, b in zip(pts[:-1], pts[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if abs(dx) + abs(dy) > 1e-6:
                angles.append(math.atan2(dx, dy))
        peak = 0.0
        for a, b in zip(angles[:-1], angles[1:]):
            d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
            if abs(d) > abs(peak):
                peak = d
        return float(peak)

    @staticmethod
    def _estimate_elongation(mask: np.ndarray) -> float:
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            return 0.0
        pts = np.column_stack([xs, ys]).astype(np.float64)
        cov = np.cov(pts, rowvar=False)
        vals = np.linalg.eigvalsh(cov)
        vals = np.maximum(vals, 1e-6)
        return float(math.sqrt(vals[-1] / vals[0]))

    @staticmethod
    def _clip_unit(v: float) -> float:
        return float(max(-1.0, min(1.0, v)))


class PipeVisualServoController:
    """Ayrik visual-servo kontrol: lateral hata sway'e, aci hata yaw'a gider."""

    STATE_SEARCH = "SEARCH"
    STATE_ACQUIRE = "ACQUIRE"
    STATE_TRACK = "TRACK"
    STATE_LOST = "LOST"

    def __init__(
        self,
        neutral_pwm: int = 1500,
        base_forward_pwm: int = 105,
        acquire_forward_pwm: int = 35,
        min_forward_scale: float = 0.25,
        kp_lateral: float = 95.0,
        kp_yaw_angle: float = 2.0,
        kp_yaw_lookahead: float = 55.0,
        max_lateral_pwm: int = 120,
        max_yaw_pwm: int = 120,
        search_yaw_pwm: int = 45,
        lost_forward_pwm: int = 0,
        min_confidence: float = 0.25,
        acquire_frames: int = 3,
        lost_hold_s: float = 0.45,
        search_switch_s: float = 3.0,
        ema_alpha: float = 0.45,
        slew_pwm_per_s: float = 500.0,
        yaw_sign: float = 1.0,
        lateral_sign: float = 1.0,
    ):
        self.neutral = int(neutral_pwm)
        self.base_forward_pwm = int(base_forward_pwm)
        self.acquire_forward_pwm = int(acquire_forward_pwm)
        self.min_forward_scale = float(min_forward_scale)
        self.kp_lateral = float(kp_lateral)
        self.kp_yaw_angle = float(kp_yaw_angle)
        self.kp_yaw_lookahead = float(kp_yaw_lookahead)
        self.max_lateral_pwm = int(max_lateral_pwm)
        self.max_yaw_pwm = int(max_yaw_pwm)
        self.search_yaw_pwm = int(search_yaw_pwm)
        self.lost_forward_pwm = int(lost_forward_pwm)
        self.min_confidence = float(min_confidence)
        self.acquire_frames = int(acquire_frames)
        self.lost_hold_s = float(lost_hold_s)
        self.search_switch_s = float(search_switch_s)
        self.ema_alpha = float(ema_alpha)
        self.slew_pwm_per_s = float(slew_pwm_per_s)
        self.yaw_sign = float(yaw_sign)
        self.lateral_sign = float(lateral_sign)

        self._state = self.STATE_SEARCH
        self._good_frames = 0
        self._last_good_t = 0.0
        self._start_t = time.monotonic()
        self._last_t = 0.0
        self._filtered_lat = None
        self._filtered_look = None
        self._filtered_angle = None
        self._prev_yaw = 0.0
        self._prev_lat = 0.0
        self._last_debug = {}

    @property
    def state(self) -> str:
        return self._state

    @property
    def debug(self) -> dict:
        return dict(self._last_debug)

    def reset(self):
        self._state = self.STATE_SEARCH
        self._good_frames = 0
        self._last_good_t = 0.0
        self._start_t = time.monotonic()
        self._last_t = 0.0
        self._filtered_lat = None
        self._filtered_look = None
        self._filtered_angle = None
        self._prev_yaw = 0.0
        self._prev_lat = 0.0
        self._last_debug = {}

    def update_params(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def stop_cmd(self) -> dict:
        return self._make_cmd(0, 0, 0)

    def compute(self, geom: PipeGeometryResult | None, now: float | None = None) -> dict:
        t = time.monotonic() if now is None else float(now)
        dt = 0.1 if self._last_t <= 0 else max(0.02, min(0.5, t - self._last_t))
        self._last_t = t

        valid = geom is not None and geom.found and geom.confidence >= self.min_confidence
        if not valid:
            return self._lost_or_search_cmd(t, dt)

        self._last_good_t = t
        self._good_frames += 1
        self._state = self.STATE_TRACK if self._good_frames >= self.acquire_frames else self.STATE_ACQUIRE

        lat = self._ema("lat", geom.lateral_error)
        look = self._ema("look", geom.lookahead_error)
        angle = self._ema("angle", geom.heading_error_deg)

        lat_offset = self.lateral_sign * self.kp_lateral * lat
        look_delta = look - lat
        yaw_offset = self.yaw_sign * (self.kp_yaw_angle * angle + self.kp_yaw_lookahead * look_delta)
        lat_offset = self._clamp(lat_offset, self.max_lateral_pwm)
        yaw_offset = self._clamp(yaw_offset, self.max_yaw_pwm)
        lat_offset = self._slew(lat_offset, self._prev_lat, self.max_lateral_pwm, dt)
        yaw_offset = self._slew(yaw_offset, self._prev_yaw, self.max_yaw_pwm, dt)
        self._prev_lat = lat_offset
        self._prev_yaw = yaw_offset

        angle_penalty = min(1.0, abs(angle) / 45.0)
        error_penalty = min(1.0, abs(lat))
        speed_scale = 1.0 - 0.55 * error_penalty - 0.35 * angle_penalty
        speed_scale = max(self.min_forward_scale, min(1.0, speed_scale))
        if self._state == self.STATE_ACQUIRE:
            fwd = min(self.acquire_forward_pwm, int(self.base_forward_pwm * speed_scale))
        else:
            fwd = int(self.base_forward_pwm * speed_scale * max(0.35, geom.confidence))

        self._last_debug = {
            "lat": lat,
            "look": look,
            "look_delta": look_delta,
            "angle": angle,
            "confidence": geom.confidence,
            "speed_scale": speed_scale,
            "valid": True,
        }
        return self._make_cmd(yaw_offset, fwd, lat_offset)

    def _lost_or_search_cmd(self, t: float, dt: float) -> dict:
        self._good_frames = 0
        if self._last_good_t > 0 and (t - self._last_good_t) <= self.lost_hold_s:
            self._state = self.STATE_LOST
            yaw = self._slew(self._prev_yaw * 0.35, self._prev_yaw, self.max_yaw_pwm, dt)
            lat = self._slew(self._prev_lat * 0.35, self._prev_lat, self.max_lateral_pwm, dt)
            self._prev_yaw = yaw
            self._prev_lat = lat
            self._last_debug = {"valid": False, "mode": "lost_hold"}
            return self._make_cmd(yaw, self.lost_forward_pwm, lat)

        self._state = self.STATE_SEARCH
        phase = int((t - self._start_t) // max(0.5, self.search_switch_s))
        desired_yaw = self.search_yaw_pwm if phase % 2 == 0 else -self.search_yaw_pwm
        yaw = self._slew(desired_yaw, self._prev_yaw, self.max_yaw_pwm, dt)
        lat = self._slew(0.0, self._prev_lat, self.max_lateral_pwm, dt)
        self._prev_yaw = yaw
        self._prev_lat = lat
        self._last_debug = {"valid": False, "mode": "search"}
        return self._make_cmd(yaw, 0, lat)

    def _ema(self, name: str, value: float) -> float:
        attr = f"_filtered_{name}"
        prev = getattr(self, attr)
        if prev is None:
            out = float(value)
        else:
            a = max(0.0, min(1.0, self.ema_alpha))
            out = a * float(value) + (1.0 - a) * float(prev)
        setattr(self, attr, out)
        return out

    def _make_cmd(self, yaw_offset: float, fwd_offset: float, lateral_offset: float) -> dict:
        return {
            "pitch": self.neutral,
            "roll": self.neutral,
            "throttle": self.neutral,
            "yaw": self.neutral + int(round(yaw_offset)),
            "forward": self.neutral + int(round(fwd_offset)),
            "lateral": self.neutral + int(round(lateral_offset)),
        }

    def _slew(self, desired: float, previous: float, limit: int, dt: float) -> float:
        step = max(1.0, self.slew_pwm_per_s * dt)
        desired = self._clamp(desired, limit)
        delta = desired - previous
        if delta > step:
            return previous + step
        if delta < -step:
            return previous - step
        return desired

    @staticmethod
    def _clamp(v: float, limit: int) -> float:
        m = abs(float(limit))
        return float(max(-m, min(m, v)))
