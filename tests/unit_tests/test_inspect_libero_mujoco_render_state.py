import numpy as np
import pytest

from toolkits.inspect_libero_mujoco_render_state import (
    array_info,
    compare_images,
    image_key_for_camera,
    parse_dummy_action,
)


def test_array_info_reports_dtype_shape_and_nbytes():
    array = np.zeros((2, 3, 4), dtype=np.uint8)

    info = array_info(array)

    assert info.dtype == "uint8"
    assert info.shape == [2, 3, 4]
    assert info.nbytes == 24


def test_compare_images_reports_pixel_differences():
    source = np.array([[[0, 10, 20], [30, 40, 50]]], dtype=np.uint8)
    target = np.array([[[0, 8, 25], [35, 40, 45]]], dtype=np.uint8)

    comparison = compare_images(source, target, image_key="agentview_image")

    assert comparison.image_key == "agentview_image"
    assert comparison.equal is False
    assert comparison.max_abs_diff == 5
    assert comparison.mean_abs_diff == pytest.approx(17 / 6)
    assert comparison.source.dtype == "uint8"
    assert comparison.target.shape == [1, 2, 3]


def test_compare_images_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_images(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((3, 2, 3), dtype=np.uint8),
            image_key="agentview_image",
        )


def test_image_key_for_camera_matches_libero_defaults():
    assert image_key_for_camera("agentview") == "agentview_image"
    assert image_key_for_camera("robot0_eye_in_hand") == "robot0_eye_in_hand_image"
    assert image_key_for_camera("frontview") == "frontview_image"


def test_parse_dummy_action_default_and_override():
    assert parse_dummy_action(None) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert parse_dummy_action("1, 2.5, -3") == [1.0, 2.5, -3.0]
    with pytest.raises(ValueError, match="invalid action vector"):
        parse_dummy_action("1,,2")
