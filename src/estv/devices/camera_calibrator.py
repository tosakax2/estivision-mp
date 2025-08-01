# estv/devices/camera_calibrator.py
"""カメラキャリブレーションに関連するクラスを提供するモジュール。"""

from filelock import FileLock, Timeout
from pathlib import Path
import time
import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class CameraCalibrator:
    """チェスボード画像からカメラ内部パラメータを推定するクラス。"""

    def __init__(self) -> None:
        """キャリブレーション用のパラメータと保存領域を初期化する。"""
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
        """チェスボードのコーナー座標をワールド座標系で生成する。"""
        objp = np.zeros((self.board_size[1] * self.board_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp


    def add_chessboard_image(self, image: np.ndarray) -> bool:
        """画像からコーナーを検出しキャリブレーション用に保存する。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        if found:
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.object_points.append(self._objp.copy())
            self.image_points.append(corners2)
        return found


    def calibrate(self, image_shape: tuple[int, int]) -> float:
        """キャリブレーションを実行し RMS 再投影誤差を計算する。"""
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
        """キャリブレーション後の総再投影誤差を計算する。"""
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
        """キャリブレーション済みのカメラ行列。"""
        return self._camera_matrix


    @property
    def dist_coeffs(self) -> np.ndarray | None:
        """キャリブレーション済みの歪み係数。"""
        return self._dist_coeffs


    @property
    def reprojection_error(self) -> float | None:
        """直近のキャリブレーションで得られた RMS 再投影誤差。"""
        return self._reproj_error


    def save(self, filename: str | Path, timeout: float = 10.0) -> None:
        """キャリブレーション結果を ``.npz`` ファイルへ保存する（ファイルロック付き）。"""
        if self._camera_matrix is None:
            raise ValueError("キャリブレーション未実施です。")
        lock_path = str(filename) + ".lock"
        lock = FileLock(lock_path, timeout=timeout)
        try:
            with lock:
                # 一時ファイル書き込み + アトミックrename
                tmp_path = str(filename) + ".tmp"
                try:
                    np.savez(
                        tmp_path,
                        camera_matrix=self._camera_matrix,
                        dist_coeffs=self._dist_coeffs,
                        reproj_error=self._reproj_error,
                    )
                    # Windowsでも上書きrename安全
                    Path(tmp_path).replace(filename)
                finally:
                    if Path(tmp_path).exists():
                        Path(tmp_path).unlink(missing_ok=True)
        except Timeout:
            raise RuntimeError(
                f"キャリブレーションパラメータ保存時にロック獲得失敗: {filename}"
            )
        except Exception as e:
            logger.exception("キャリブレーションパラメータ保存中にエラーが発生しました")
            raise RuntimeError(
                f"キャリブレーションパラメータ保存中にエラー発生: {e}"
            ) from e


    def load(self, filename: str | Path, timeout: float = 10.0) -> None:
        """``.npz`` ファイルからキャリブレーションパラメータを読み込む（ファイルロック付き）。"""
        lock_path = str(filename) + ".lock"
        lock = FileLock(lock_path, timeout=timeout)
        try:
            with lock:
                data = np.load(str(filename))
                self._camera_matrix = data["camera_matrix"]
                self._dist_coeffs = data["dist_coeffs"]
                self._reproj_error = float(data["reproj_error"]) if "reproj_error" in data else None
        except Timeout:
            raise RuntimeError(f"キャリブレーションパラメータ読込時にロック獲得失敗: {filename}")
