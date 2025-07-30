# estv/gui/camera_preview_window.py

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QCloseEvent
from collections.abc import Callable

from estv.devices.camera_stream_manager import CameraStreamManager

from estv.gui.style_constants import TEXT_COLOR, BACKGROUND_COLOR


class CameraPreviewWindow(QDialog):
    """カメラのプレビューウィンドウ。"""

    def __init__(
        self,
        camera_stream_manager: CameraStreamManager,
        device_id: str = "",
        parent: QWidget | None = None,
        on_closed: Callable[[str], None] | None = None,
    ) -> None:
        """コンストラクタ。"""
        super().__init__(parent)
        self.setWindowTitle("ESTV - カメラプレビュー")
        self.device_id = device_id
        self.camera_stream_manager = camera_stream_manager
        self._on_closed = on_closed # 閉じたときに呼び出されるコールバック

        # --- カメラ映像用ラベル
        self.image_label = QLabel("カメラ映像がここに表示されます")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 角丸の黒背景を設定
        self.image_label.setStyleSheet(f"""
            background-color: {BACKGROUND_COLOR};
            border-radius: 8px;
            color: {TEXT_COLOR};
        """)
        self.image_label.setFixedSize(480, 480)

        self.close_button = QPushButton("閉じる")
        self.close_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.close_button)
        self.setLayout(layout)

        self.adjustSize()
        self.setFixedSize(self.size())

        # --- シグナル接続
        self.camera_stream_manager.q_image_ready.connect(self._on_image_ready)


    def _on_image_ready(self, device_id: str, qimg: QImage) -> None:
        """カメラからの映像が準備できたときに呼び出される。"""
        if device_id != self.device_id:
            return
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))


    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウが閉じられたときの処理。"""
        self.camera_stream_manager.q_image_ready.disconnect(self._on_image_ready)
        self.camera_stream_manager.stop_camera(self.device_id)
        if self._on_closed:
            self._on_closed(self.device_id)
        super().closeEvent(event)
