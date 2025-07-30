# estv/devices/camera_stream.py

import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QImage


# --- 定数
MAX_LONG_SIDE_LENGTH = 640  # 長辺の最大サイズ
CAPTURE_FPS = 30 # 最大フレームレート


def resize_if_needed(frame: np.ndarray, max_length: int) -> np.ndarray:
    """Resize the frame only if its longer side exceeds ``max_length``.

    Parameters
    ----------
    frame : numpy.ndarray
        Image array in BGR format.
    max_length : int
        Maximum allowed size of the longer side.

    Returns
    -------
    numpy.ndarray
        Possibly resized frame.
    """
    h, w = frame.shape[:2]
    if max(w, h) > max_length:
        scale = max_length / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


class CameraStream(QThread):
    """Thread that continually captures frames from a camera device."""

    # --- シグナル
    error = Signal(str) # エラーメッセージを通知
    frame_ready = Signal(object) # フレーム取得時に通知
    q_image_ready = Signal(QImage) # GUI用のQImage取得時に通知


    def __init__(self, device_id: int) -> None:
        """Initialize the stream thread for the given device index."""
        super().__init__()

        # --- 引数保持
        self._device_id: int = device_id


    def run(self) -> None:
        """Main capture loop executed when the thread starts."""
        cap = cv2.VideoCapture(self._device_id, cv2.CAP_DSHOW)

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

            # --- 映像取得ループ
            interval = 1.0 / CAPTURE_FPS  # 1フレームの理想間隔（秒）
            while not self.isInterruptionRequested():
                start_time = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                # --- 必要ならリサイズ
                frame = resize_if_needed(frame, MAX_LONG_SIDE_LENGTH)

                # --- OpenCVのBGRからQtのRGBに変換
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

                # --- シグナルを発行
                self.frame_ready.emit(frame)
                self.q_image_ready.emit(qimg)

                # --- FPS制御
                elapsed = time.perf_counter() - start_time
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release() # カメラを解放


    @Slot()
    def stop(self) -> None:
        """Request graceful termination of the capture loop."""
        self.requestInterruption()
