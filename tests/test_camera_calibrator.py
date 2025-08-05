import numpy as np
import cv2
import pytest
from estv.devices.camera_calibrator import CameraCalibrator


def generate_chessboard(board_size=(6, 9), square_size=40):
    pattern_size = (board_size[0] + 1, board_size[1] + 1)
    img = np.zeros((pattern_size[1] * square_size, pattern_size[0] * square_size), dtype=np.uint8)
    for y in range(pattern_size[1]):
        for x in range(pattern_size[0]):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x * square_size, y * square_size),
                    ((x + 1) * square_size - 1, (y + 1) * square_size - 1),
                    255,
                    -1,
                )
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_calibrate_and_save_load(tmp_path):
    calib = CameraCalibrator()
    img = generate_chessboard()
    for _ in range(3):
        assert calib.add_chessboard_image(img)
    result = calib.calibrate(img.shape[:2])
    assert isinstance(result, float)
    assert calib.camera_matrix is not None
    assert calib.dist_coeffs is not None
    assert calib.reprojection_error is not None

    filename = tmp_path / "calib.npz"
    calib.save(filename)
    assert filename.exists()

    loaded = CameraCalibrator()
    loaded.load(filename)
    assert np.allclose(calib.camera_matrix, loaded.camera_matrix)
    assert np.allclose(calib.dist_coeffs, loaded.dist_coeffs)


def test_calibrate_requires_three_images():
    calib = CameraCalibrator()
    img = generate_chessboard()
    for _ in range(2):
        calib.add_chessboard_image(img)
    with pytest.raises(ValueError):
        calib.calibrate(img.shape[:2])
