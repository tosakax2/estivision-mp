# src/estv/devices/stereo_calibrator.py

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoParams:
    """ステレオ校正結果（内部・外部パラメータ一式）。

    Notes
    -----
    - r, t は「カメラ1座標系 → カメラ2座標系」への変換。
    - p1/p2 は整流後の投影行列。三角測量では使わず保存のみ。
    - 単位は square_size_m に依存（通常はメートル）。
    """

    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray          # (3,3)
    t: np.ndarray          # (3,)
    r1: np.ndarray         # rectification for cam1
    r2: np.ndarray         # rectification for cam2
    p1: np.ndarray         # projection for cam1 (rectified)
    p2: np.ndarray         # projection for cam2 (rectified)
    q: np.ndarray          # disparity-to-depth matrix
    rms: float
    board_size: tuple[int, int]
    square_size_m: float
    cam1_id: str
    cam2_id: str
    image_size: tuple[int, int]

    def save(self, path: Path) -> None:
        """NPZ形式で保存する。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            k1=self.k1,
            d1=self.d1,
            k2=self.k2,
            d2=self.d2,
            r=self.r,
            t=self.t,
            r1=self.r1,
            r2=self.r2,
            p1=self.p1,
            p2=self.p2,
            q=self.q,
            rms=np.float64(self.rms),
            board_size=np.asarray(self.board_size, dtype=np.int32),
            square_size_m=np.float64(self.square_size_m),
            cam1_id=self.cam1_id,
            cam2_id=self.cam2_id,
            image_size=np.asarray(self.image_size, dtype=np.int32),
        )

    @staticmethod
    def load(path: Path) -> "StereoParams":
        """NPZから読込む。"""
        data = np.load(str(path), allow_pickle=True)
        return StereoParams(
            k1=np.asarray(data["k1"]),
            d1=np.asarray(data["d1"]),
            k2=np.asarray(data["k2"]),
            d2=np.asarray(data["d2"]),
            r=np.asarray(data["r"]),
            t=np.asarray(data["t"]).reshape(3),
            r1=np.asarray(data["r1"]),
            r2=np.asarray(data["r2"]),
            p1=np.asarray(data["p1"]),
            p2=np.asarray(data["p2"]),
            q=np.asarray(data["q"]),
            rms=float(data["rms"]),
            board_size=tuple(int(x) for x in data["board_size"]),
            square_size_m=float(data["square_size_m"]),
            cam1_id=str(data["cam1_id"]),
            cam2_id=str(data["cam2_id"]),
            image_size=tuple(int(x) for x in data["image_size"]),
        )


def _make_object_points(
    board_size: tuple[int, int],
    square_size_m: float,
) -> np.ndarray:
    """チェスボード平面上の3D点（Z=0）を生成する。"""
    cols, rows = board_size  # 例：(6, 9) = 横6, 縦9 の内点数
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def stereo_calibrate(
    img_points1: Sequence[np.ndarray],
    img_points2: Sequence[np.ndarray],
    board_size: tuple[int, int],
    square_size_m: float,
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
    image_size: tuple[int, int],
    cam1_id: str,
    cam2_id: str,
) -> StereoParams:
    """内部固定でステレオ校正を行い、外部パラメータ等を返す。

    Parameters
    ----------
    img_points1, img_points2
        各フレームのチェスボード検出結果。各要素は (N,1,2) float32。
        （cv2.findChessboardCorners + cornerSubPix の出力を想定）
    board_size
        チェスボードの「内点」数 (cols, rows)。
    square_size_m
        マスの一辺の長さ（メートル）。
    k1, d1, k2, d2
        各カメラの内部パラメータ・歪み係数（単眼校正済み）。
    image_size
        画像サイズ (width, height)。
    cam1_id, cam2_id
        保存や識別用の文字列ID。

    Returns
    -------
    StereoParams
        ステレオ校正の結果一式。
    """
    assert len(img_points1) == len(img_points2), "対応フレーム数が不一致です。"
    assert len(img_points1) >= 1, "十分な枚数の画像点がありません。"

    # --- 3Dチェスボード座標（Z=0）。各フレーム同一なのでコピーレフ。
    objp = _make_object_points(board_size, square_size_m)
    obj_points = [objp] * len(img_points1)

    # --- 収束条件とフラグ（内部固定）
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )
    flags = cv2.CALIB_FIX_INTRINSIC

    rms, k1o, d1o, k2o, d2o, r, t, e, f = cv2.stereoCalibrate(
        objectPoints=obj_points,
        imagePoints1=list(img_points1),
        imagePoints2=list(img_points2),
        cameraMatrix1=k1.copy(),
        distCoeffs1=d1.copy(),
        cameraMatrix2=k2.copy(),
        distCoeffs2=d2.copy(),
        imageSize=image_size,
        criteria=criteria,
        flags=flags,
    )

    # --- 整流行列など（後工程で使う可能性があるため保存しておく）
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        cameraMatrix1=k1o,
        distCoeffs1=d1o,
        cameraMatrix2=k2o,
        distCoeffs2=d2o,
        imageSize=image_size,
        R=r,
        T=t,
        alpha=0,  # 0:切り詰め最大, 1:すべて保持
    )

    return StereoParams(
        k1=k1o,
        d1=d1o,
        k2=k2o,
        d2=d2o,
        r=r,
        t=t.reshape(3),
        r1=r1,
        r2=r2,
        p1=p1,
        p2=p2,
        q=q,
        rms=float(rms),
        board_size=board_size,
        square_size_m=float(square_size_m),
        cam1_id=cam1_id,
        cam2_id=cam2_id,
        image_size=image_size,
    )


def stereo_calibrate_and_save(
    img_points1: Sequence[np.ndarray],
    img_points2: Sequence[np.ndarray],
    board_size: tuple[int, int],
    square_size_m: float,
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
    image_size: tuple[int, int],
    cam1_id: str,
    cam2_id: str,
    save_path: Path,
) -> StereoParams:
    """ステレオ校正を実行して保存まで行うユーティリティ。"""
    params = stereo_calibrate(
        img_points1=img_points1,
        img_points2=img_points2,
        board_size=board_size,
        square_size_m=square_size_m,
        k1=k1,
        d1=d1,
        k2=k2,
        d2=d2,
        image_size=image_size,
        cam1_id=cam1_id,
        cam2_id=cam2_id,
    )
    params.save(save_path)
    return params
