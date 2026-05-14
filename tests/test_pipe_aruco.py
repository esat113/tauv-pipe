import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipe_tracker_aruco import ArucoMarkerDetector


def aruco_available() -> bool:
    aruco = getattr(cv2, "aruco", None)
    return (
        aruco is not None
        and hasattr(aruco, "DICT_ARUCO_ORIGINAL")
        and hasattr(aruco, "generateImageMarker")
    )


def make_marker(marker_id: int, size: int = 110) -> np.ndarray:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    marker = aruco.generateImageMarker(dictionary, marker_id, size)
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)


def make_scene(ids: list[int], marker_size: int = 110, margin: int = 28) -> np.ndarray:
    width = margin + len(ids) * (marker_size + margin)
    height = marker_size + 2 * margin
    scene = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, marker_id in enumerate(ids):
        x0 = margin + i * (marker_size + margin)
        y0 = margin
        scene[y0 : y0 + marker_size, x0 : x0 + marker_size] = make_marker(marker_id, marker_size)
    return scene


def test_original_dictionary_marker_is_detected():
    if not aruco_available():
        print("cv2.aruco unavailable, skipping ArUco detection test")
        return

    detector = ArucoMarkerDetector()
    result = detector.detect(make_scene([23]), now=1.0)

    assert result.available
    assert result.detected
    assert result.ids == [23]
    assert result.read_order == [23]
    assert result.markers[0].area_frac > 0

    annotated = detector.annotate(make_scene([23]), result)
    assert annotated.shape == make_scene([23]).shape


def test_read_order_appends_only_newly_visible_ids():
    if not aruco_available():
        print("cv2.aruco unavailable, skipping ArUco order test")
        return

    detector = ArucoMarkerDetector(history_size=30)

    first = detector.detect(make_scene([4, 7]), now=1.0)
    first_order = list(first.read_order)
    assert set(first.ids) == {4, 7}
    assert set(first_order) == {4, 7}

    second = detector.detect(make_scene([4, 7]), now=1.1)
    assert second.read_order == first_order

    third = detector.detect(make_scene([4, 7, 12]), now=1.2)
    assert set(third.ids) == {4, 7, 12}
    assert third.read_order.count(12) == 1
    assert third.read_order[-1] == 12


if __name__ == "__main__":
    test_original_dictionary_marker_is_detected()
    test_read_order_appends_only_newly_visible_ids()
    print("pipe aruco tests passed")
