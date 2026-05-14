"""EKF last-seen reacquire helpers for pipe tracking.

The last-seen target in v1 is the vehicle pose at the last reliable pipe
observation, not a projected 3D pipe point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from threading import Lock


PWM_NEUTRAL = 1500


def wrap_pi(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PoseSample:
    x: float
    y: float
    depth: float
    yaw: float
    timestamp_ms: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    initialized: bool = False
    attitude_stale: bool = True
    depth_stale: bool = True
    dvl_stale: bool = True
    heading_stale: bool = True

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "PoseSample":
        return cls(
            x=float(getattr(snapshot, "x")),
            y=float(getattr(snapshot, "y")),
            depth=float(getattr(snapshot, "depth")),
            yaw=float(getattr(snapshot, "yaw")),
            timestamp_ms=float(getattr(snapshot, "timestamp_ms")),
            vx=float(getattr(snapshot, "vx", 0.0) or 0.0),
            vy=float(getattr(snapshot, "vy", 0.0) or 0.0),
            vz=float(getattr(snapshot, "vz", 0.0) or 0.0),
            initialized=bool(getattr(snapshot, "initialized", False)),
            attitude_stale=bool(getattr(snapshot, "attitude_stale", True)),
            depth_stale=bool(getattr(snapshot, "depth_stale", True)),
            dvl_stale=bool(getattr(snapshot, "dvl_stale", True)),
            heading_stale=bool(getattr(snapshot, "heading_stale", True)),
        )

    @property
    def usable(self) -> bool:
        return (
            self.initialized
            and not self.attitude_stale
            and not self.depth_stale
            and not self.dvl_stale
            and not self.heading_stale
        )


@dataclass(frozen=True)
class LastPipeObservation:
    pose: PoseSample
    mask_ts_ms: float
    wall_time_s: float
    confidence: float
    reason: str = ""

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - float(self.wall_time_s))


@dataclass(frozen=True)
class ReturnCommand:
    state: str
    rc: dict
    target_yaw_odom_rad: float | None
    distance_m: float
    last_seen_age_s: float
    reason: str
    arrived: bool = False
    timed_out: bool = False
    heading_error_deg: float = 0.0
    target_x: float | None = None
    target_y: float | None = None


class PoseHistory:
    def __init__(self, maxlen: int = 120):
        self._samples: deque[PoseSample] = deque(maxlen=max(1, int(maxlen)))
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def append(self, sample: PoseSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def push_snapshot(self, snapshot: object) -> PoseSample:
        sample = PoseSample.from_snapshot(snapshot)
        self.append(sample)
        return sample

    def latest(self, require_usable: bool = False) -> PoseSample | None:
        with self._lock:
            for sample in reversed(self._samples):
                if not require_usable or sample.usable:
                    return sample
        return None

    def nearest(
        self,
        timestamp_ms: float,
        max_delta_ms: float,
        require_usable: bool = False,
    ) -> PoseSample | None:
        t = float(timestamp_ms)
        best: PoseSample | None = None
        best_delta = float("inf")
        with self._lock:
            for sample in self._samples:
                if require_usable and not sample.usable:
                    continue
                delta = abs(float(sample.timestamp_ms) - t)
                if delta < best_delta:
                    best_delta = delta
                    best = sample
        if best is None or best_delta > float(max_delta_ms):
            return None
        return best


def observation_from_pose_history(
    history: PoseHistory,
    mask_ts_ms: float,
    wall_time_s: float,
    confidence: float,
    reason: str = "",
    pose_match_max_delta_ms: float = 600.0,
) -> LastPipeObservation | None:
    pose = history.nearest(mask_ts_ms, pose_match_max_delta_ms, require_usable=True)
    if pose is None:
        return None
    return LastPipeObservation(
        pose=pose,
        mask_ts_ms=float(mask_ts_ms),
        wall_time_s=float(wall_time_s),
        confidence=float(confidence),
        reason=str(reason or ""),
    )


def select_return_observation(
    observations,
    now_s: float,
    current_pose: PoseSample | None = None,
    max_age_s: float = 8.0,
    target_lookback_s: float = 1.2,
    min_target_distance_m: float = 0.45,
) -> LastPipeObservation | None:
    """Pick a useful older observation instead of the final edge-of-frame hit.

    The newest valid mask is often captured at the exact place where the pipe is
    lost, which makes the return target equal to the current pose. Prefer a
    recent-but-not-last observation near ``target_lookback_s`` seconds old, and
    when current pose is available prefer samples that require actual motion.
    """

    now = float(now_s)
    max_age = max(0.0, float(max_age_s))
    lookback = max(0.0, float(target_lookback_s))
    min_distance = max(0.0, float(min_target_distance_m))

    candidates = [
        obs for obs in observations
        if obs is not None and obs.age_s(now) <= max_age
    ]
    if not candidates:
        return None

    def distance(obs: LastPipeObservation) -> float:
        if current_pose is None or not current_pose.usable:
            return 0.0
        return math.hypot(obs.pose.x - current_pose.x, obs.pose.y - current_pose.y)

    motion_candidates = candidates
    if current_pose is not None and current_pose.usable and min_distance > 0.0:
        far_enough = [obs for obs in candidates if distance(obs) >= min_distance]
        if far_enough:
            motion_candidates = far_enough

    aged = [obs for obs in motion_candidates if obs.age_s(now) >= lookback]
    pool = aged or motion_candidates

    return min(
        pool,
        key=lambda obs: (
            abs(obs.age_s(now) - lookback),
            -float(obs.confidence),
            -distance(obs),
        ),
    )


class _Pid:
    def __init__(self, kp: float, ki: float, kd: float, output_min: float = -1.0, output_max: float = 1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error: float | None = None
        self._last_t: float | None = None

    def update(self, setpoint: float, measurement: float, now_s: float) -> float:
        error = float(setpoint) - float(measurement)
        t = float(now_s)
        dt = 0.1 if self._last_t is None else max(1e-3, min(0.5, t - self._last_t))
        self._last_t = t

        self._integral += error * dt
        self._integral = max(-2.0, min(2.0, self._integral))

        derivative = 0.0
        if self._last_error is not None:
            derivative = (error - self._last_error) / dt
        self._last_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, out))


class ReturnToLastSeenController:
    STATE_IDLE = "IDLE"
    STATE_RETURN = "RETURN_LAST_SEEN"
    STATE_LOCAL_SEARCH = "LOCAL_SEARCH"

    def __init__(
        self,
        neutral_pwm: int = PWM_NEUTRAL,
        return_delay_s: float = 0.7,
        return_accept_radius_m: float = 0.35,
        return_timeout_s: float = 10.0,
        max_last_seen_age_s: float = 8.0,
        return_speed_m_s: float = 0.15,
        return_fwd_kp: float = 1.0,
        return_fwd_ki: float = 0.0,
        return_fwd_kd: float = 0.0,
        return_forward_pwm_min_offset: int = 35,
        return_forward_heading_gate_deg: float = 70.0,
        return_heading_soft_gate_deg: float = 25.0,
        return_forward_min_scale: float = 0.35,
        return_lateral_kp_pwm_per_m: float = 80.0,
        return_lateral_kd_pwm_s_per_m: float = 4.0,
        return_pwm_min: int = 1450,
        return_pwm_max: int = 1550,
        max_fwd_pwm_step: int = 20,
        align_heading_tol_deg: float = 15.0,
        return_yaw_kp_pwm_per_rad: float = 90.0,
        return_yaw_pwm_max: int = 80,
        return_yaw_sign: float = 1.0,
        local_search_yaw_pwm: int = 35,
        local_search_switch_s: float = 2.0,
    ):
        self.neutral = int(neutral_pwm)
        self.return_delay_s = float(return_delay_s)
        self.return_accept_radius_m = float(return_accept_radius_m)
        self.return_timeout_s = float(return_timeout_s)
        self.max_last_seen_age_s = float(max_last_seen_age_s)
        self.return_speed_m_s = float(return_speed_m_s)
        self.return_fwd_kp = float(return_fwd_kp)
        self.return_fwd_ki = float(return_fwd_ki)
        self.return_fwd_kd = float(return_fwd_kd)
        self.return_forward_pwm_min_offset = int(return_forward_pwm_min_offset)
        self.return_forward_heading_gate_deg = float(return_forward_heading_gate_deg)
        self.return_heading_soft_gate_deg = float(return_heading_soft_gate_deg)
        self.return_forward_min_scale = float(return_forward_min_scale)
        self.return_lateral_kp_pwm_per_m = float(return_lateral_kp_pwm_per_m)
        self.return_lateral_kd_pwm_s_per_m = float(return_lateral_kd_pwm_s_per_m)
        self.return_pwm_min = int(return_pwm_min)
        self.return_pwm_max = int(return_pwm_max)
        self.max_fwd_pwm_step = int(max_fwd_pwm_step)
        self.align_heading_tol_deg = float(align_heading_tol_deg)
        self.return_yaw_kp_pwm_per_rad = float(return_yaw_kp_pwm_per_rad)
        self.return_yaw_pwm_max = int(return_yaw_pwm_max)
        self.return_yaw_sign = float(return_yaw_sign)
        self.local_search_yaw_pwm = int(local_search_yaw_pwm)
        self.local_search_switch_s = float(local_search_switch_s)

        self._pid = _Pid(self.return_fwd_kp, self.return_fwd_ki, self.return_fwd_kd)
        self._state = self.STATE_IDLE
        self._observation: LastPipeObservation | None = None
        self._start_s = 0.0
        self._local_search_start_s = 0.0
        self._prev_fwd_pwm = self.neutral
        self._prev_lat_pwm = self.neutral
        self._prev_yaw_pwm = self.neutral
        self._prev_lateral_err: float | None = None
        self._prev_lateral_t: float | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def active(self) -> bool:
        return self._state in (self.STATE_RETURN, self.STATE_LOCAL_SEARCH)

    @property
    def observation(self) -> LastPipeObservation | None:
        return self._observation

    def update_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._pid = _Pid(self.return_fwd_kp, self.return_fwd_ki, self.return_fwd_kd)

    def reset(self) -> None:
        self._state = self.STATE_IDLE
        self._observation = None
        self._start_s = 0.0
        self._local_search_start_s = 0.0
        self._prev_fwd_pwm = self.neutral
        self._prev_lat_pwm = self.neutral
        self._prev_yaw_pwm = self.neutral
        self._prev_lateral_err = None
        self._prev_lateral_t = None
        self._pid.reset()

    def begin(
        self,
        observation: LastPipeObservation,
        now_s: float,
    ) -> None:
        self.reset()
        self._observation = observation
        self._start_s = float(now_s)
        self._state = self.STATE_RETURN

    def maybe_start(
        self,
        observation: LastPipeObservation | None,
        visual_lost_s: float,
        now_s: float,
    ) -> bool:
        if self.active:
            return True
        if observation is None:
            return False
        if float(visual_lost_s) < self.return_delay_s:
            return False
        if observation.age_s(now_s) > self.max_last_seen_age_s:
            return False
        self.begin(observation, now_s)
        return True

    def update(self, current_pose: PoseSample | None, now_s: float) -> ReturnCommand:
        obs = self._observation
        if obs is None or not self.active:
            return self._command(self.STATE_IDLE, self._neutral_cmd(), None, 0.0, 0.0, "idle")

        age = obs.age_s(now_s)

        if current_pose is None or not current_pose.usable:
            return self._command(self._state, self._neutral_cmd(), None, 0.0, age, "pose_unusable")

        dx = obs.pose.x - current_pose.x
        dy = obs.pose.y - current_pose.y
        distance = math.hypot(dx, dy)

        if self._state == self.STATE_LOCAL_SEARCH:
            rc = self._local_search_cmd(now_s)
            return self._command(self.STATE_LOCAL_SEARCH, rc, None, distance, age, "local_search")

        if now_s - self._start_s > self.return_timeout_s:
            self._enter_local_search(now_s)
            rc = self._local_search_cmd(now_s)
            return self._command(
                self.STATE_LOCAL_SEARCH,
                rc,
                None,
                distance,
                age,
                "return_timeout_search",
                timed_out=True,
            )

        if distance <= self.return_accept_radius_m:
            self._enter_local_search(now_s)
            rc = self._local_search_cmd(now_s)
            return self._command(
                self.STATE_LOCAL_SEARCH,
                rc,
                None,
                distance,
                age,
                "arrived_search",
                arrived=True,
            )

        return self._drive_to_xy(
            obs.pose.x,
            obs.pose.y,
            current_pose,
            now_s,
            age,
            self.STATE_RETURN,
            moving_reason="returning",
            aligning_reason="aligning",
        )

    def _enter_local_search(self, now_s: float) -> None:
        self._state = self.STATE_LOCAL_SEARCH
        self._local_search_start_s = float(now_s)
        self._pid.reset()
        self._prev_lateral_err = None
        self._prev_lateral_t = None

    def _drive_to_xy(
        self,
        target_x: float,
        target_y: float,
        current_pose: PoseSample,
        now_s: float,
        age: float,
        state: str,
        moving_reason: str,
        aligning_reason: str,
        arrived: bool = False,
        timed_out: bool = False,
    ) -> ReturnCommand:
        dx = float(target_x) - current_pose.x
        dy = float(target_y) - current_pose.y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        heading_err = wrap_pi(target_yaw - current_pose.yaw)
        soft_gate = math.radians(max(self.align_heading_tol_deg, self.return_heading_soft_gate_deg))
        hard_gate = math.radians(max(self.return_forward_heading_gate_deg, self.return_heading_soft_gate_deg))

        if abs(heading_err) > soft_gate:
            self._pid.reset()
            self._prev_lateral_err = None
            self._prev_lateral_t = None
            self._prev_lat_pwm = self.neutral
            rc = self._neutral_cmd()
            rc["yaw"] = self._yaw_pwm(heading_err)
            rc["forward"] = self._gated_forward_pwm(heading_err, hard_gate)
            return self._command(
                state,
                rc,
                target_yaw,
                distance,
                age,
                aligning_reason,
                arrived=arrived,
                timed_out=timed_out,
                heading_error_rad=heading_err,
                target_x=target_x,
                target_y=target_y,
            )

        ux = dx / max(distance, 1e-6)
        uy = dy / max(distance, 1e-6)
        v_along = current_pose.vx * ux + current_pose.vy * uy
        fwd_u = self._pid.update(self.return_speed_m_s, v_along, now_s)
        fwd_pwm = self._forward_pwm(fwd_u)
        fwd_pwm = self._apply_heading_forward_gate(fwd_pwm, heading_err, hard_gate)
        fwd_pwm = self._slew_pwm(fwd_pwm, self._prev_fwd_pwm)
        self._prev_fwd_pwm = fwd_pwm

        c = math.cos(current_pose.yaw)
        s = math.sin(current_pose.yaw)
        lateral_err = -dx * s + dy * c
        if self._prev_lateral_err is None or self._prev_lateral_t is None:
            dlat = 0.0
        else:
            dt = max(1e-3, min(0.5, float(now_s) - self._prev_lateral_t))
            dlat = (lateral_err - self._prev_lateral_err) / dt
        self._prev_lateral_err = lateral_err
        self._prev_lateral_t = float(now_s)

        lat_offset = (
            self.return_lateral_kp_pwm_per_m * lateral_err
            + self.return_lateral_kd_pwm_s_per_m * dlat
        )
        lat_pwm = self._offset_to_pwm(lat_offset)
        lat_pwm = self._slew_pwm(lat_pwm, self._prev_lat_pwm)
        self._prev_lat_pwm = lat_pwm

        rc = self._neutral_cmd()
        rc["yaw"] = self._yaw_pwm(heading_err)
        rc["forward"] = fwd_pwm
        rc["lateral"] = lat_pwm
        return self._command(
            state,
            rc,
            target_yaw,
            distance,
            age,
            moving_reason,
            arrived=arrived,
            timed_out=timed_out,
            heading_error_rad=heading_err,
            target_x=target_x,
            target_y=target_y,
        )

    def _local_search_cmd(self, now_s: float) -> dict:
        phase = int((float(now_s) - self._local_search_start_s) // max(0.5, self.local_search_switch_s))
        desired = self.neutral + (self.local_search_yaw_pwm if phase % 2 == 0 else -self.local_search_yaw_pwm)
        yaw_pwm = self._slew_pwm(desired, self._prev_yaw_pwm)
        self._prev_yaw_pwm = yaw_pwm
        rc = self._neutral_cmd()
        rc["yaw"] = yaw_pwm
        return rc

    def _yaw_pwm(self, heading_err_rad: float) -> int:
        max_offset = max(0, abs(int(self.return_yaw_pwm_max)))
        offset = self.return_yaw_sign * self.return_yaw_kp_pwm_per_rad * float(heading_err_rad)
        offset = max(-max_offset, min(max_offset, offset))
        desired = self.neutral + int(round(offset))
        yaw_pwm = self._slew_pwm(desired, self._prev_yaw_pwm)
        self._prev_yaw_pwm = yaw_pwm
        return yaw_pwm

    def _forward_pwm(self, fwd_u: float) -> int:
        offset = float(fwd_u) * self._half_pwm_range()
        if offset > 0.0:
            offset = max(offset, float(max(0, int(self.return_forward_pwm_min_offset))))
        return self._offset_to_pwm(offset)

    def _gated_forward_pwm(self, heading_err_rad: float, gate: float | None = None) -> int:
        gate = math.radians(max(0.0, float(self.return_forward_heading_gate_deg))) if gate is None else float(gate)
        if gate <= 0.0 or abs(heading_err_rad) > gate:
            fwd_pwm = self._slew_pwm(self.neutral, self._prev_fwd_pwm)
            self._prev_fwd_pwm = fwd_pwm
            return fwd_pwm

        min_offset = max(0, int(self.return_forward_pwm_min_offset))
        scale = self._heading_forward_scale(heading_err_rad, gate)
        desired = self._offset_to_pwm(min_offset * scale)
        fwd_pwm = self._slew_pwm(desired, self._prev_fwd_pwm)
        self._prev_fwd_pwm = fwd_pwm
        return fwd_pwm

    def _apply_heading_forward_gate(self, fwd_pwm: int, heading_err_rad: float, gate: float) -> int:
        if gate <= 0.0 or abs(heading_err_rad) > gate:
            return self.neutral
        offset = max(0.0, float(fwd_pwm) - self.neutral)
        min_offset = max(0, int(self.return_forward_pwm_min_offset))
        scale = self._heading_forward_scale(heading_err_rad, gate)
        offset = max(offset * scale, min_offset * scale)
        return self._offset_to_pwm(offset)

    def _heading_forward_scale(self, heading_err_rad: float, gate: float) -> float:
        if gate <= 1e-6:
            return 1.0
        frac = min(1.0, abs(float(heading_err_rad)) / gate)
        min_scale = max(0.0, min(1.0, float(self.return_forward_min_scale)))
        return max(min_scale, 1.0 - 0.65 * frac)

    def _neutral_cmd(self) -> dict:
        return {
            "pitch": self.neutral,
            "roll": self.neutral,
            "throttle": self.neutral,
            "yaw": self.neutral,
            "forward": self.neutral,
            "lateral": self.neutral,
        }

    def _command(
        self,
        state: str,
        rc: dict,
        target_yaw_odom_rad: float | None,
        distance_m: float,
        last_seen_age_s: float,
        reason: str,
        arrived: bool = False,
        timed_out: bool = False,
        heading_error_rad: float = 0.0,
        target_x: float | None = None,
        target_y: float | None = None,
    ) -> ReturnCommand:
        return ReturnCommand(
            state=state,
            rc=rc,
            target_yaw_odom_rad=target_yaw_odom_rad,
            distance_m=float(distance_m),
            last_seen_age_s=float(last_seen_age_s),
            reason=str(reason),
            arrived=bool(arrived),
            timed_out=bool(timed_out),
            heading_error_deg=math.degrees(float(heading_error_rad)),
            target_x=None if target_x is None else float(target_x),
            target_y=None if target_y is None else float(target_y),
        )

    def _half_pwm_range(self) -> float:
        return float(min(abs(self.return_pwm_max - self.neutral), abs(self.neutral - self.return_pwm_min)))

    def _offset_to_pwm(self, offset: float) -> int:
        lo = min(self.return_pwm_min, self.return_pwm_max)
        hi = max(self.return_pwm_min, self.return_pwm_max)
        return int(round(max(lo, min(hi, self.neutral + float(offset)))))

    def _slew_pwm(self, desired_pwm: int, previous_pwm: int) -> int:
        step = max(1, abs(int(self.max_fwd_pwm_step)))
        lo = int(previous_pwm) - step
        hi = int(previous_pwm) + step
        return int(max(lo, min(hi, int(desired_pwm))))
