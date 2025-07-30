# estv/devices/camera_stream_manager.py

import threading
import warnings

from PySide6.QtCore import QObject, Signal, QTimer

from estv.devices.camera_stream import CameraStream


class CameraStreamManager(QObject):
    """複数のカメラストリームを管理するクラス。"""

    streams_updated = Signal()
    frame_ready = Signal(int, object)
    q_image_ready = Signal(int, object)


    def __init__(self, auto_restart: bool = False, restart_delay_ms: int = 2000) -> None:
        super().__init__()

        # --- ストリームを保持する辞書
        self._streams: dict[int, CameraStream] = {}

        # --- 自動再起動設定
        self._auto_restart: bool = auto_restart
        self._restart_delay_ms: int = restart_delay_ms

        # --- 再起動待ちデバイス
        self._pending_restart: set[int] = set()

        # --- 排他制御用ロック
        self._lock = threading.Lock()


    def start_camera(self, device_id: int) -> None:
        """指定したデバイスIDのカメラストリームを開始する。"""
        with self._lock:
            if device_id in self._streams:
                warnings.warn(f"Camera {device_id} stream already running")
                return
            stream = CameraStream(device_id)
            stream.frame_ready.connect(lambda frame, d=device_id: self.frame_ready.emit(d, frame))
            stream.q_image_ready.connect(lambda qimg, d=device_id: self.q_image_ready.emit(d, qimg))
            stream.error.connect(lambda msg, d=device_id: self.handle_error(d, msg))
            stream.finished.connect(lambda d=device_id: self.cleanup_stream(d))
            self._streams[device_id] = stream
            stream.start()
        self.streams_updated.emit()


    def stop_camera(self, device_id: int) -> None:
        """指定したデバイスIDのカメラストリームを停止しクリーンアップする。"""
        with self._lock:
            stream = self._streams.get(device_id)

        if stream is None:
            warnings.warn(f"Camera {device_id} stream not running")
            return

        stream.stop()
        stream.wait()
        self.cleanup_stream(device_id)


    def stop_all(self) -> None:
        """全てのカメラストリームを停止する。"""
        for device_id in self.running_device_ids():
            self.stop_camera(device_id)
        self.streams_updated.emit()


    def shutdown(self) -> None:
        """すべてのカメラストリームを停止し、終了を待機する。"""
        self.stop_all()


    def handle_error(self, device_id: int, msg: str) -> None:
        """カメラストリームのエラーを処理する。"""
        print(f"[Camera {device_id}] Error: {msg}")
        with self._lock:
            if device_id in self._streams:
                stream = self._streams[device_id]
                stream.stop()

        if self._auto_restart:
            self._pending_restart.add(device_id)


    def cleanup_stream(self, device_id: int) -> None:
        """指定したデバイスIDのカメラストリームをクリーンアップする。"""
        with self._lock:
            if device_id in self._streams:
                stream = self._streams[device_id]
                stream.deleteLater()
                del self._streams[device_id]

        if self._auto_restart and device_id in self._pending_restart:
            self._pending_restart.remove(device_id)
            QTimer.singleShot(
                self._restart_delay_ms,
                lambda d=device_id: self.start_camera(d),
            )
        self.streams_updated.emit()


    def running_device_ids(self) -> list[int]:
        """現在起動中のカメラの device_id リストを返す。"""
        with self._lock:
            return list(self._streams.keys())
