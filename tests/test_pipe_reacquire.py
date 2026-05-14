import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipe_reacquire import (
    LastPipeObservation,
    PoseHistory,
    PoseSample,
    ReturnToLastSeenController,
    observation_from_pose_history,
    select_return_observation,
)


def pose(
    x=0.0,
    y=0.0,
    yaw=0.0,
    ts=0.0,
    vx=0.0,
    vy=0.0,
    initialized=True,
    stale=False,
):
    return PoseSample(
        x=x,
        y=y,
        depth=1.0,
        yaw=yaw,
        vx=vx,
        vy=vy,
        vz=0.0,
        timestamp_ms=ts,
        initialized=initialized,
        attitude_stale=stale,
        depth_stale=stale,
        dvl_stale=stale,
        heading_stale=stale,
    )


def test_pose_history_selects_nearest_timestamp():
    hist = PoseHistory()
    hist.append(pose(x=1.0, ts=1000.0))
    hist.append(pose(x=2.0, ts=1110.0))
    hist.append(pose(x=3.0, ts=1500.0))

    selected = hist.nearest(1080.0, max_delta_ms=80.0, require_usable=True)

    assert selected is not None
    assert selected.x == 2.0


def test_pose_history_rejects_stale_or_uninitialized_pose():
    hist = PoseHistory()
    hist.append(pose(x=1.0, ts=1000.0, stale=True))
    hist.append(pose(x=2.0, ts=1020.0, initialized=False))

    selected = hist.nearest(1010.0, max_delta_ms=100.0, require_usable=True)

    assert selected is None


def test_observation_from_pose_history_uses_valid_matched_pose():
    hist = PoseHistory()
    hist.append(pose(x=4.0, y=5.0, ts=2000.0))

    obs = observation_from_pose_history(
        hist,
        mask_ts_ms=2010.0,
        wall_time_s=7.0,
        confidence=0.8,
        reason="ok",
        pose_match_max_delta_ms=50.0,
    )

    assert obs is not None
    assert obs.pose.x == 4.0
    assert obs.pose.y == 5.0
    assert obs.confidence == 0.8


def test_select_return_observation_prefers_lookback_target():
    observations = [
        LastPipeObservation(pose=pose(x=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.7),
        LastPipeObservation(pose=pose(x=0.4, ts=1800.0), mask_ts_ms=1800.0, wall_time_s=1.8, confidence=0.9),
        LastPipeObservation(pose=pose(x=0.9, ts=3000.0), mask_ts_ms=3000.0, wall_time_s=3.0, confidence=0.9),
    ]

    selected = select_return_observation(
        observations,
        now_s=3.1,
        current_pose=pose(x=1.0, ts=3100.0),
        target_lookback_s=1.2,
        min_target_distance_m=0.35,
    )

    assert selected is observations[1]


def test_select_return_observation_avoids_final_near_loss_pose():
    observations = [
        LastPipeObservation(pose=pose(x=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.7),
        LastPipeObservation(pose=pose(x=0.95, ts=3000.0), mask_ts_ms=3000.0, wall_time_s=3.0, confidence=0.9),
    ]

    selected = select_return_observation(
        observations,
        now_s=3.1,
        current_pose=pose(x=1.0, ts=3100.0),
        target_lookback_s=0.1,
        min_target_distance_m=0.35,
    )

    assert selected is observations[0]


def test_return_controller_does_not_start_before_delay():
    ctrl = ReturnToLastSeenController(return_delay_s=0.7)
    obs = LastPipeObservation(pose=pose(ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)

    started = ctrl.maybe_start(obs, visual_lost_s=0.4, now_s=1.4)

    assert not started
    assert ctrl.state == ReturnToLastSeenController.STATE_IDLE


def test_return_controller_starts_after_delay_and_aligns_first():
    ctrl = ReturnToLastSeenController(return_delay_s=0.7)
    obs = LastPipeObservation(pose=pose(x=1.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)

    started = ctrl.maybe_start(obs, visual_lost_s=0.8, now_s=1.8)
    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=1.0, ts=1800.0), now_s=1.8)

    assert started
    assert cmd.state == ReturnToLastSeenController.STATE_RETURN
    assert cmd.reason == "aligning"
    assert cmd.rc["forward"] > 1500
    assert cmd.rc["lateral"] == 1500
    assert cmd.rc["yaw"] != 1500
    assert cmd.target_yaw_odom_rad is not None


def test_return_controller_pwm_clamp_and_step_limit():
    ctrl = ReturnToLastSeenController(
        return_delay_s=0.0,
        return_pwm_min=1450,
        return_pwm_max=1550,
        max_fwd_pwm_step=20,
        return_speed_m_s=1.0,
        return_fwd_kp=10.0,
        return_lateral_kp_pwm_per_m=500.0,
        align_heading_tol_deg=180.0,
    )
    obs = LastPipeObservation(pose=pose(x=5.0, y=5.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    cmd1 = ctrl.update(pose(x=0.0, y=0.0, yaw=0.0, ts=2000.0), now_s=2.0)
    cmd2 = ctrl.update(pose(x=0.0, y=0.0, yaw=0.0, ts=2100.0), now_s=2.1)

    assert cmd1.rc["forward"] == 1520
    assert cmd1.rc["lateral"] == 1520
    assert 1520 < cmd2.rc["forward"] <= 1540
    assert cmd2.rc["lateral"] == 1540


def test_return_controller_applies_min_forward_offset():
    ctrl = ReturnToLastSeenController(
        return_delay_s=0.0,
        return_speed_m_s=0.15,
        return_fwd_kp=1.0,
        return_forward_pwm_min_offset=35,
        max_fwd_pwm_step=100,
        align_heading_tol_deg=180.0,
    )
    obs = LastPipeObservation(pose=pose(x=2.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=0.0, ts=2000.0), now_s=2.0)

    assert cmd.reason == "returning"
    assert cmd.rc["forward"] == 1535


def test_return_controller_creeps_forward_while_nearly_aligned():
    ctrl = ReturnToLastSeenController(
        return_delay_s=0.0,
        return_forward_pwm_min_offset=80,
        return_forward_heading_gate_deg=70.0,
        return_heading_soft_gate_deg=25.0,
        max_fwd_pwm_step=100,
        align_heading_tol_deg=15.0,
    )
    obs = LastPipeObservation(pose=pose(x=2.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=0.5, ts=2000.0), now_s=2.0)

    assert cmd.reason == "aligning"
    assert 1500 < cmd.rc["forward"] < 1581
    assert cmd.rc["yaw"] != 1500


def test_return_controller_returns_forward_inside_direction_cone():
    ctrl = ReturnToLastSeenController(
        return_delay_s=0.0,
        return_forward_pwm_min_offset=80,
        return_forward_heading_gate_deg=70.0,
        return_heading_soft_gate_deg=25.0,
        max_fwd_pwm_step=100,
        align_heading_tol_deg=15.0,
    )
    obs = LastPipeObservation(pose=pose(x=2.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=0.4, ts=2000.0), now_s=2.0)

    assert cmd.reason == "returning"
    assert cmd.rc["forward"] > 1500
    assert cmd.rc["yaw"] != 1500


def test_active_return_does_not_abort_when_last_seen_age_exceeds_start_limit():
    ctrl = ReturnToLastSeenController(
        return_delay_s=0.0,
        max_last_seen_age_s=1.0,
        return_timeout_s=10.0,
        align_heading_tol_deg=180.0,
        max_fwd_pwm_step=100,
    )
    obs = LastPipeObservation(pose=pose(x=2.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=1.5)

    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=0.0, ts=3500.0), now_s=3.5)

    assert cmd.state == ReturnToLastSeenController.STATE_RETURN
    assert cmd.reason == "returning"


def test_return_timeout_switches_to_local_search_not_idle_search():
    ctrl = ReturnToLastSeenController(
        return_delay_s=0.0,
        return_timeout_s=1.0,
        align_heading_tol_deg=180.0,
    )
    obs = LastPipeObservation(pose=pose(x=2.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=0.0, ts=3200.0), now_s=3.2)

    assert cmd.state == ReturnToLastSeenController.STATE_LOCAL_SEARCH
    assert cmd.reason == "return_timeout_search"
    assert cmd.timed_out


def test_return_controller_enters_local_search_on_arrival():
    ctrl = ReturnToLastSeenController(return_delay_s=0.0, return_accept_radius_m=0.35)
    obs = LastPipeObservation(pose=pose(x=0.1, y=0.1, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    cmd = ctrl.update(pose(x=0.0, y=0.0, yaw=0.0, ts=2000.0), now_s=2.0)

    assert cmd.state == ReturnToLastSeenController.STATE_LOCAL_SEARCH
    assert cmd.reason == "arrived_search"
    assert cmd.arrived


def test_return_controller_can_be_cancelled_when_pipe_reappears():
    ctrl = ReturnToLastSeenController(return_delay_s=0.0)
    obs = LastPipeObservation(pose=pose(x=1.0, y=0.0, ts=1000.0), mask_ts_ms=1000.0, wall_time_s=1.0, confidence=0.9)
    assert ctrl.maybe_start(obs, visual_lost_s=1.0, now_s=2.0)

    ctrl.reset()

    assert not ctrl.active
    assert ctrl.state == ReturnToLastSeenController.STATE_IDLE


if __name__ == "__main__":
    test_pose_history_selects_nearest_timestamp()
    test_pose_history_rejects_stale_or_uninitialized_pose()
    test_observation_from_pose_history_uses_valid_matched_pose()
    test_select_return_observation_prefers_lookback_target()
    test_select_return_observation_avoids_final_near_loss_pose()
    test_return_controller_does_not_start_before_delay()
    test_return_controller_starts_after_delay_and_aligns_first()
    test_return_controller_pwm_clamp_and_step_limit()
    test_return_controller_applies_min_forward_offset()
    test_return_controller_creeps_forward_while_nearly_aligned()
    test_return_controller_returns_forward_inside_direction_cone()
    test_active_return_does_not_abort_when_last_seen_age_exceeds_start_limit()
    test_return_timeout_switches_to_local_search_not_idle_search()
    test_return_controller_enters_local_search_on_arrival()
    test_return_controller_can_be_cancelled_when_pipe_reappears()
    print("pipe reacquire tests passed")
