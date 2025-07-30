# estv/devices/camera_stream_manager.py

import threading
import warnings

from collections.abc import Callable
from PySide6.QtCore import QObject, Signal, QTimer

from estv.devices.camera_stream import CameraStream


class CameraStreamManager(QObject):
    """複数のカメラストリームを管理するクラス。"""

    streams_updated = Signal()
    frame_ready = Signal(str, object)
    q_image_ready = Signal(str, object)


    def __init__(
        self,
        device_index_lookup: Callable[[str], int | None],
        auto_restart: bool = False,
        restart_delay_ms: int = 2000,
    ) -> None:
        super().__init__()

        # --- ストリームを保持する辞書
        self._streams: dict[str, CameraStream] = {}

        # --- カメラIDからインデックスを取得する関数
        self._device_index_lookup = device_index_lookup

        # --- 自動再起動設定
        self._auto_restart: bool = auto_restart
        self._restart_delay_ms: int = restart_delay_ms

        # --- 再起動待ちデバイス
        self._pending_restart: set[str] = set()

        # --- 排他制御用ロック
        self._lock = threading.Lock()


    def start_camera(self, camera_id: str) -> None:
        """指定したカメラIDのストリームを開始する。"""
        device_index = self._device_index_lookup(camera_id)
        if device_index is None:
            warnings.warn(f"Camera {camera_id} not found")
            return
        with self._lock:
            if camera_id in self._streams:
                warnings.warn(f"Camera {camera_id} stream already running")
                return
            stream = CameraStream(device_index)
            stream.frame_ready.connect(lambda frame, c=camera_id: self.frame_ready.emit(c, frame))
            stream.q_image_ready.connect(lambda qimg, c=camera_id: self.q_image_ready.emit(c, qimg))
            stream.error.connect(lambda msg, c=camera_id: self.handle_error(c, msg))
            stream.finished.connect(lambda c=camera_id: self.cleanup_stream(c))
            self._streams[camera_id] = stream
            stream.start()
        self.streams_updated.emit()


    def stop_camera(self, camera_id: str) -> None:
        """指定したカメラIDのストリームを停止しクリーンアップする。"""
        with self._lock:
            stream = self._streams.get(camera_id)

        if stream is None:
            warnings.warn(f"Camera {camera_id} stream not running")
            return

        stream.stop()
        stream.wait()
        self.cleanup_stream(camera_id)


    def stop_all(self) -> None:
        """全てのカメラストリームを停止する。"""
        for camera_id in self.running_device_ids():
            self.stop_camera(camera_id)
        self.streams_updated.emit()


    def shutdown(self) -> None:
        """すべてのカメラストリームを停止し、終了を待機する。"""
        self.stop_all()


    def handle_error(self, camera_id: str, msg: str) -> None:
        """カメラストリームのエラーを処理する。"""
        print(f"[Camera {camera_id}] Error: {msg}")
        with self._lock:
            if camera_id in self._streams:
                stream = self._streams[camera_id]
                stream.stop()

        if self._auto_restart:
            self._pending_restart.add(camera_id)


    def cleanup_stream(self, camera_id: str) -> None:
        """指定したカメラIDのストリームをクリーンアップする。"""
        with self._lock:
            if camera_id in self._streams:
                stream = self._streams[camera_id]
                stream.deleteLater()
                del self._streams[camera_id]

        if self._auto_restart and camera_id in self._pending_restart:
            self._pending_restart.remove(camera_id)
            QTimer.singleShot(
                self._restart_delay_ms,
                lambda c=camera_id: self.start_camera(c),
            )
        self.streams_updated.emit()


    def running_device_ids(self) -> list[str]:
        """現在起動中のカメラIDリストを返す。"""
        with self._lock:
            return list(self._streams.keys())
