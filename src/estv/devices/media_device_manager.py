# estv/devices/media_device_manager.py
"""利用可能なメディアデバイスの監視を行うモジュール。"""

from PySide6.QtCore import (
    QObject,
    QTimer,
    Signal
)
from PySide6.QtMultimedia import (
    QCameraDevice,
    QMediaDevices
)


class MediaDeviceManager(QObject):
    """利用可能なメディアデバイスを監視し、カメラ情報を提供するクラス。"""

    # --- シグナル
    camera_devices_update_signal = Signal(list)


    def __init__(self) -> None:
        """マネージャーを初期化してデバイス一覧を取得する。"""
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
        """``QMediaDevices`` からのカメラデバイス更新に対応する。"""
        # --- 最新のカメラデバイス一覧を再取得
        self._camera_devices: list[QCameraDevice] = self._media_devices.videoInputs()

        # --- カメラデバイス名一覧を通知
        self._notify_camera_devices_update()


    def _notify_camera_devices_update(self) -> None:
        """現在のカメラデバイス一覧をシグナルで通知する。"""
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
        """カメラデバイスIDから名称へのマッピング。"""
        return {
            bytes(dev.id()).decode("utf-8", errors="ignore"): dev.description()
            for dev in self._camera_devices
        }

    def camera_index_by_id(self, camera_id: str) -> int | None:
        """指定IDのカメラの現在のインデックスを返す。"""
        for idx, dev in enumerate(self._camera_devices):
            if bytes(dev.id()).decode("utf-8", errors="ignore") == camera_id:
                return idx
        return None
