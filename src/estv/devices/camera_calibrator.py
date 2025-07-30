# estv/devices/camera_calibrator.py

from pathlib import Path

import cv2
import numpy as np


class CameraCalibrator:
    """Estimate camera intrinsics from captured chessboard images."""

    def __init__(self) -> None:
        """Initialize calibration parameters and storage."""
        self.board_size: tuple[int, int] = (6, 9)   # 内部コーナー数 (横, 縦)
        self.square_size: float = 20.0              # 1マスの一辺（mm）

        self.object_points: list[np.ndarray] = []   # ワールド座標 (N, 3)
        self.image_points: list[np.ndarray] = []    # 画像座標 (N, 1, 2)
        self._objp: np.ndarray = self._create_object_points()

        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        self._rvecs: list[np.ndarray] | None = None
        self._tvecs: list[np.ndarray] | None = None
        self._reproj_error: float | None = None


    def _create_object_points(self) -> np.ndarray:
        """Create chessboard corner coordinates in world space."""
        objp = np.zeros((self.board_size[1] * self.board_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp


    def add_chessboard_image(self, image: np.ndarray) -> bool:
        """Detect corners from an image and store them for calibration."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        if found:
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.object_points.append(self._objp.copy())
            self.image_points.append(corners2)
        return found


    def calibrate(self, image_shape: tuple[int, int]) -> float:
        """Run calibration and compute the RMS reprojection error."""
        if len(self.object_points) < 3:
            raise ValueError("最低3枚以上のチェスボード画像が必要です。")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, image_shape[::-1], None, None
        )
        self._camera_matrix = mtx
        self._dist_coeffs = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self._reproj_error = self._calc_reprojection_error()
        return ret


    def _calc_reprojection_error(self) -> float | None:
        """Calculate the overall reprojection error after calibration."""
        if (
            self._rvecs is None
            or self._tvecs is None
            or self._camera_matrix is None
            or self._dist_coeffs is None
        ):
            return None
        total_error = 0.0
        total_points = 0
        for objp, imgp, rvec, tvec in zip(self.object_points, self.image_points, self._rvecs, self._tvecs):
            proj, _ = cv2.projectPoints(objp, rvec, tvec, self._camera_matrix, self._dist_coeffs)
            error = cv2.norm(imgp, proj, cv2.NORM_L2)
            total_error += error ** 2
            total_points += len(objp)
        return float(np.sqrt(total_error / total_points)) if total_points > 0 else None


    @property
    def camera_matrix(self) -> np.ndarray | None:
        """Calibrated camera matrix if available."""
        return self._camera_matrix


    @property
    def dist_coeffs(self) -> np.ndarray | None:
        """Calibrated distortion coefficients if available."""
        return self._dist_coeffs


    @property
    def reprojection_error(self) -> float | None:
        """RMS reprojection error from the last calibration."""
        return self._reproj_error


    def save(self, filename: str | Path) -> None:
        """Save calibration parameters to a ``.npz`` file."""
        if self._camera_matrix is None:
            raise ValueError("キャリブレーション未実施です。")
        np.savez(str(filename), camera_matrix=self._camera_matrix, dist_coeffs=self._dist_coeffs)


    def load(self, filename: str | Path) -> None:
        """Load calibration parameters from a ``.npz`` file."""
        data = np.load(str(filename))
        self._camera_matrix = data["camera_matrix"]
        self._dist_coeffs = data["dist_coeffs"]
