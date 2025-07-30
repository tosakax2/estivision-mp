# estv/devices/media_device_manager.py

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QCameraDevice, QMediaDevices


class MediaDeviceManager(QObject):
    """Monitor available media devices and expose camera information."""

    # --- シグナル
    camera_devices_update_signal = Signal(list)


    def __init__(self) -> None:
        """Initialize the manager and fetch initial device list."""
        super().__init__()

        # --- メディアデバイスの情報を取得
        self._media_devices: QMediaDevices = QMediaDevices()

        # --- カメラデバイスの変更時に呼び出し
        self._media_devices.videoInputsChanged.connect(self._on_camera_devices_changed)

        # --- カメラデバイスのリストをキャッシュ
        self._camera_devices: list[QCameraDevice] = self._media_devices.videoInputs()

        # --- シグナルを初回発行
        # シグナル接続後に必ず届くよう、0ms 後に呼び出しをスケジュール
        QTimer.singleShot(0, self._notify_camera_devices_update)


    def _on_camera_devices_changed(self) -> None:
        """React to camera device updates from ``QMediaDevices``."""
        # --- 最新のカメラデバイス一覧を再取得
        self._camera_devices: list[QCameraDevice] = self._media_devices.videoInputs()

        # --- カメラデバイス名一覧を通知
        self._notify_camera_devices_update()


    def _notify_camera_devices_update(self) -> None:
        """Emit a signal with the current list of camera devices."""
        # --- カメラデバイスのIDと名前をまとめたリストを作成
        camera_device_infos: list[dict[str, str]] = [
            {
                "id": bytes(dev.id()).decode("utf-8", errors="ignore"),
                "name": dev.description(),
            }
            for dev in self._camera_devices
        ]

        # --- シグナルを発行
        self.camera_devices_update_signal.emit(camera_device_infos)


    @property
    def camera_id_name_map(self) -> dict[str, str]:
        """Mapping of camera device IDs to their human readable names."""
        return {
            bytes(dev.id()).decode("utf-8", errors="ignore"): dev.description()
            for dev in self._camera_devices
        }

    def camera_index_by_id(self, camera_id: str) -> int | None:
        """Return the current index for the camera with ``camera_id``."""
        for idx, dev in enumerate(self._camera_devices):
            if bytes(dev.id()).decode("utf-8", errors="ignore") == camera_id:
                return idx
        return None
