# estv/devices/camera_stream_manager.py

from collections.abc import Callable
import threading
import warnings

from PySide6.QtCore import QObject, Signal, QTimer

from estv.devices.camera_stream import CameraStream


class CameraStreamManager(QObject):
    """複数の :class:`CameraStream` インスタンスを管理するクラス。"""

    streams_updated = Signal()
    frame_ready = Signal(str, object)
    q_image_ready = Signal(str, object)


    def __init__(
        self,
        device_index_lookup: Callable[[str], int | None],
        auto_restart: bool = False,
        restart_delay_ms: int = 2000,
    ) -> None:
        """マネージャーを生成する。

        パラメータ
        ----------
        device_index_lookup : Callable[[str], int | None]
            デバイスIDから ``cv2.VideoCapture`` 用のインデックスを返す関数。
        auto_restart : bool, optional
            エラー発生時に自動再起動するかどうか。
        restart_delay_ms : int, optional
            再起動を試みるまでの待ち時間（ミリ秒）。
        """
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
        """指定したIDのカメラのストリームを開始する。"""
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
        """指定したカメラのストリームを停止して後始末を行う。"""
        with self._lock:
            stream = self._streams.get(camera_id)

        if stream is None:
            warnings.warn(f"Camera {camera_id} stream not running")
            return

        stream.stop()
        stream.wait()
        self.cleanup_stream(camera_id)


    def set_exposure(self, camera_id: str, value: float) -> None:
        """指定カメラの露出を設定する。"""
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream is not None:
            stream.set_exposure(value)


    def set_gain(self, camera_id: str, value: float) -> None:
        """指定カメラのゲインを設定する。"""
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream is not None:
            stream.set_gain(value)


    def stop_all(self) -> None:
        """実行中のすべてのカメラストリームを停止する。"""
        for camera_id in self.running_device_ids():
            self.stop_camera(camera_id)
        self.streams_updated.emit()


    def shutdown(self) -> None:
        """全ストリームを停止し、スレッド終了を待つ。"""
        self.stop_all()


    def handle_error(self, camera_id: str, msg: str) -> None:
        """``camera_id`` のストリームでエラーが発生した際の処理を行う。"""
        print(f"[Camera {camera_id}] Error: {msg}")
        with self._lock:
            if camera_id in self._streams:
                stream = self._streams[camera_id]
                stream.stop()

        if self._auto_restart:
            self._pending_restart.add(camera_id)


    def cleanup_stream(self, camera_id: str) -> None:
        """終了したストリームを削除し、必要なら再起動する。"""
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
        """現在実行中のカメラIDの一覧を返す。"""
        with self._lock:
            return list(self._streams.keys())
