# estv/devices/camera_calibrator.py

import cv2
import numpy as np


class CameraCalibrator:
    """チェスボード画像からカメラ内部パラメータを推定するクラス。"""

    def __init__(self, board_size=(6, 9), square_size=1.0):
        """
        Args:
            board_size: チェスボード交点の数 (columns, rows)
            square_size: 1マスの実寸（mm, cm, 任意単位）
        """
        self.board_size = board_size
        self.square_size = square_size
        self.object_points = []  # 3D座標(ワールド)
        self.image_points = []   # 2D座標(画像)
        self._objp = self._create_object_points()

        self._camera_matrix = None
        self._dist_coeffs = None
        self._rvecs = None
        self._tvecs = None
        self._reproj_error = None


    def _create_object_points(self):
        """チェスボード上の3D座標を生成"""
        objp = np.zeros((self.board_size[1] * self.board_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp


    def add_chessboard_image(self, image):
        """
        画像からチェスボードコーナーを検出し、内部点リストへ追加。
        Args:
            image (ndarray): キャリブレーション用画像(BGR/Gray)
        Returns:
            found (bool): コーナー検出成功か
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        if found:
            # サブピクセル補正
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.object_points.append(self._objp)
            self.image_points.append(corners2)
        return found


    def calibrate(self, image_shape):
        """
        キャリブレーション実行（内部パラメータ推定）
        Args:
            image_shape (tuple): (高さ, 幅)
        Returns:
            rms (float): 平均再投影誤差
        """
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


    def _calc_reprojection_error(self):
        """全体の再投影誤差を算出"""
        total_error = 0
        total_points = 0
        for objp, imgp, rvec, tvec in zip(self.object_points, self.image_points, self._rvecs, self._tvecs):
            proj, _ = cv2.projectPoints(objp, rvec, tvec, self._camera_matrix, self._dist_coeffs)
            error = cv2.norm(imgp, proj, cv2.NORM_L2)
            total_error += error ** 2
            total_points += len(objp)
        return np.sqrt(total_error / total_points) if total_points > 0 else None


    @property
    def camera_matrix(self):
        return self._camera_matrix


    @property
    def dist_coeffs(self):
        return self._dist_coeffs


    @property
    def reprojection_error(self):
        return self._reproj_error


    def save(self, filename):
        """パラメータをファイル保存(npz)"""
        if self._camera_matrix is None:
            raise ValueError("キャリブレーション未実施です。")
        np.savez(filename, camera_matrix=self._camera_matrix, dist_coeffs=self._dist_coeffs)


    def load(self, filename):
        """ファイルからパラメータ読込(npz)"""
        data = np.load(filename)
        self._camera_matrix = data["camera_matrix"]
        self._dist_coeffs = data["dist_coeffs"]
