# src/estv/estimators/triangulation.py

import numpy as np
import cv2

from estv.devices.stereo_calibrator import StereoParams


class OpenCVTriangulator:
    """StereoParams を用いて OpenCV の三角測量を行うクラス。"""

    def __init__(self, params: StereoParams) -> None:
        self.params = params
        # --- 正規化座標前提の射影行列
        self._p1 = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
        self._p2 = np.hstack([params.r, params.t.reshape(3, 1)]).astype(np.float64)


    def triangulate(
        self,
        pts1_px: np.ndarray,
        pts2_px: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """画素座標の対応点から3D点を復元する（カメラ1座標系, 単位m）。

        Parameters
        ----------
        pts1_px, pts2_px : (N, 2) float array
            画素座標。NaNや欠損点は事前に除外して渡すこと。

        Returns
        -------
        X : (N, 3) float64
            復元3D点（cam1座標系, m）。
        mask : (N,) bool
            両カメラで z>0 を満たす有効フラグ（チェイラリティ）。
        """
        assert pts1_px.shape == pts2_px.shape
        pts1_px = np.asarray(pts1_px, dtype=np.float64)
        pts2_px = np.asarray(pts2_px, dtype=np.float64)

        # --- (u,v)->正規化(x,y)
        p = self.params
        pts1 = cv2.undistortPoints(pts1_px.reshape(-1, 1, 2), p.k1, p.d1).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pts2_px.reshape(-1, 1, 2), p.k2, p.d2).reshape(-1, 2)

        xh = cv2.triangulatePoints(self._p1, self._p2, pts1.T, pts2.T)  # 4xN
        x = (xh[:3] / xh[3]).T  # (N,3)

        # --- チェイラリティ
        z1 = x[:, 2]
        x2 = (p.r @ x.T + p.t.reshape(3, 1)).T
        z2 = x2[:, 2]
        mask = (z1 > 0.0) & (z2 > 0.0)
        return x, mask


    def reprojection_rmse(
        self, X: np.ndarray, pts1_px: np.ndarray, pts2_px: np.ndarray
    ) -> float:
        """デバッグ用：再投影RMSE（px）。"""
        p = self.params
        rvec1 = np.zeros((3, 1), dtype=np.float64)
        tvec1 = np.zeros((3, 1), dtype=np.float64)
        rvec2, _ = cv2.Rodrigues(p.r.astype(np.float64))
        tvec2 = p.t.reshape(3, 1).astype(np.float64)

        proj1, _ = cv2.projectPoints(X, rvec1, tvec1, p.k1, p.d1)
        proj2, _ = cv2.projectPoints(X, rvec2, tvec2, p.k2, p.d2)
        e = np.vstack([
            (proj1.reshape(-1, 2) - pts1_px).ravel(),
            (proj2.reshape(-1, 2) - pts2_px).ravel(),
        ])
        return float(np.sqrt(np.mean(e**2)))
