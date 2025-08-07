# estv/devices/stereo_calibrator.py
"""2 台カメラの外部パラメータ（R, T）を推定するユーティリティ."""

from pathlib import Path

import cv2
import numpy as np

from estv.devices.camera_calibrator import CameraCalibrator


class StereoCalibrator:
    """内部パラメータ既知の 2 カメラをステレオキャリブレーションする簡易クラス。"""

    def __init__(
        self,
        camera1_calib: Path,
        camera2_calib: Path,
        board_size: tuple[int, int] = (6, 9),
        square_size: float = 20.0,
    ) -> None:
        # --- 各カメラの内部パラメータを読み込み
        self._calib1 = CameraCalibrator()
        self._calib1.load(camera1_calib)
        self._calib2 = CameraCalibrator()
        self._calib2.load(camera2_calib)

        # --- チェスボード情報
        self._board_size = board_size
        self._square_size = square_size
        self._objp = self._create_object_points()


    def calibrate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """2 画像から (R, T) を推定し RMS 誤差を返す."""
        found1, corners1 = self._find_corners(img1)
        found2, corners2 = self._find_corners(img2)
        if not (found1 and found2):
            raise RuntimeError("チェスボードを両方のカメラで検出できませんでした。")

        flags = (
            cv2.CALIB_FIX_INTRINSIC
            | cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_RATIONAL_MODEL
        )
        retval, *_ = cv2.stereoCalibrate(
            [self._objp],
            [corners1],
            [corners2],
            self._calib1.camera_matrix,
            self._calib1.dist_coeffs,
            self._calib2.camera_matrix,
            self._calib2.dist_coeffs,
            img1.shape[:2][::-1],
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        )
        return float(retval)  # RMS


    def _create_object_points(self) -> np.ndarray:
        objp = np.zeros((self._board_size[0] * self._board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : self._board_size[0], 0 : self._board_size[1]
        ].T.reshape(-1, 2)
        objp *= self._square_size
        return objp


    def _find_corners(
        self, img: np.ndarray
    ) -> tuple[bool, np.ndarray | None]:  # (found, corners)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self._board_size)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return found, corners
