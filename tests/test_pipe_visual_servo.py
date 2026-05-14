import cv2
import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipe_algorithm import PipeGeometryProcessor, PipeVisualServoController


def make_pipe_mask(width=160, height=120, x_bottom=80, x_top=80, thickness=16):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(mask, (int(x_bottom), height - 1), (int(x_top), 0), 255, thickness)
    return mask


def test_centered_vertical_pipe_commands_forward_without_correction():
    proc = PipeGeometryProcessor()
    ctrl = PipeVisualServoController()

    geom = proc.process(make_pipe_mask())
    assert geom.found
    assert abs(geom.lateral_error) < 0.08
    assert abs(geom.heading_error_deg) < 3.0

    cmd = ctrl.compute(geom, now=1.0)
    assert cmd["forward"] > 1500
    assert abs(cmd["lateral"] - 1500) <= 8
    assert abs(cmd["yaw"] - 1500) <= 8


def test_right_shifted_straight_pipe_uses_lateral_more_than_yaw():
    proc = PipeGeometryProcessor()
    ctrl = PipeVisualServoController()

    geom = proc.process(make_pipe_mask(x_bottom=112, x_top=112))
    assert geom.found
    assert geom.lateral_error > 0.25
    assert abs(geom.heading_error_deg) < 3.0

    cmd = ctrl.compute(geom, now=1.0)
    assert cmd["lateral"] > 1500
    assert abs(cmd["yaw"] - 1500) < abs(cmd["lateral"] - 1500)


def test_diagonal_pipe_produces_yaw_correction():
    proc = PipeGeometryProcessor()
    ctrl = PipeVisualServoController()

    geom = proc.process(make_pipe_mask(x_bottom=70, x_top=112))
    assert geom.found
    assert geom.heading_error_deg > 10.0

    cmd = ctrl.compute(geom, now=1.0)
    assert cmd["yaw"] > 1500


def test_missing_pipe_enters_search_without_forward_motion():
    ctrl = PipeVisualServoController(search_yaw_pwm=40)
    cmd = ctrl.compute(None, now=1.0)

    assert ctrl.state == PipeVisualServoController.STATE_SEARCH
    assert cmd["forward"] == 1500
    assert cmd["yaw"] != 1500


def test_boolean_style_mask_is_accepted():
    proc = PipeGeometryProcessor()
    mask = make_pipe_mask()
    geom = proc.process((mask > 0).astype(np.uint8))

    assert geom.found
    assert geom.area > 0


if __name__ == "__main__":
    test_centered_vertical_pipe_commands_forward_without_correction()
    test_right_shifted_straight_pipe_uses_lateral_more_than_yaw()
    test_diagonal_pipe_produces_yaw_correction()
    test_missing_pipe_enters_search_without_forward_motion()
    test_boolean_style_mask_is_accepted()
    print("pipe visual servo tests passed")
