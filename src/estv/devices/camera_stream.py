# estv/devices/camera_stream.py

import time

import cv2
import numpy as np
from PySide6.QtCore import (
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage


# --- 定数
MAX_LONG_SIDE_LENGTH = 320  # 長辺の最大サイズ
CAPTURE_FPS = 30 # 最大フレームレート


def resize_if_needed(frame: np.ndarray, max_length: int) -> np.ndarray:
    """長辺が ``max_length`` を超える場合のみフレームをリサイズする。

    パラメータ
    ----------
    frame : numpy.ndarray
        BGR 形式の画像配列。
    max_length : int
        長辺の最大サイズ。

    戻り値
    ------
    numpy.ndarray
        リサイズ後のフレーム。
    """
    h, w = frame.shape[:2]
    if max(w, h) > max_length:
        scale = max_length / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


class CameraStream(QThread):
    """カメラデバイスから連続でフレームを取得するスレッド。"""

    # --- シグナル
    error = Signal(str) # エラーメッセージを通知
    frame_ready = Signal(object) # フレーム取得時に通知
    q_image_ready = Signal(QImage) # GUI用のQImage取得時に通知


    def __init__(self, device_id: int) -> None:
        """指定されたデバイス番号でストリームスレッドを初期化する。"""
        super().__init__()

        # --- 引数保持
        self._device_id: int = device_id
        self._cap: cv2.VideoCapture | None = None
        self._pending_exposure: float | None = None
        # --- ソフトウェア明るさ補正値 (0-100)
        self._brightness: float = 0.0


    def run(self) -> None:
        """スレッド開始時に実行されるメインのキャプチャループ。"""
        cap = cv2.VideoCapture(self._device_id, cv2.CAP_ANY)
        self._cap = cap

        try:
            if not cap.isOpened():
                self.error.emit("カメラを開けませんでした。")
                return

            # --- カメラに希望解像度をリクエスト
            origin_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            origin_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            long_side = max(origin_w, origin_h)
            scale = min(1.0, MAX_LONG_SIDE_LENGTH / long_side) if long_side > 0 else 1.0
            target_w = int(origin_w * scale)
            target_h = int(origin_h * scale)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

            if self._pending_exposure is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, self._pending_exposure)

            # --- 映像取得ループ
            interval = 1.0 / CAPTURE_FPS  # 1フレームの理想間隔（秒）
            next_frame_time = time.perf_counter()
            while not self.isInterruptionRequested():
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time += interval

                # --- 実フレーム取得
                ret, frame = cap.read()
                if not ret:
                    break

                # --- 必要ならリサイズ
                frame = resize_if_needed(frame, MAX_LONG_SIDE_LENGTH)

                # --- 明るさ補正
                if self._brightness != 0:
                    factor = 1.0 + self._brightness / 100.0
                    frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)

                # --- ルミナンスノイズ除去
                frame = cv2.GaussianBlur(frame, (3, 3), 0)

                # --- OpenCVのBGRからQtのRGBに変換
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                qimg = QImage(
                    rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
                ).copy()

                # --- シグナルを発行
                self.frame_ready.emit(frame)
                self.q_image_ready.emit(qimg)
        finally:
            cap.release()
            self._cap = None


    @Slot()
    def stop(self) -> None:
        """キャプチャループの終了をリクエストする。"""
        self.requestInterruption()


    @Slot(float)
    def set_exposure(self, value: float) -> None:
        """露出値を設定する。"""
        self._pending_exposure = value
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, value)


    @Slot(float)
    def set_brightness(self, value: float) -> None:
        """ソフトウェアで適用する明るさ補正値を設定する。"""
        self._brightness = value
