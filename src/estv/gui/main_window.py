# estv/gui/main_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QAbstractItemView, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCloseEvent

from estv.devices.camera_stream_manager import CameraStreamManager
from estv.devices.media_device_manager import MediaDeviceManager
from estv.gui.camera_preview_window import CameraPreviewWindow
from estv.gui.style_constants import SUCCESS_COLOR, WARNING_COLOR


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ。"""

    def __init__(self) -> None:
        """コンストラクタ。"""
        super().__init__()

        self._camera_stream_manager: CameraStreamManager = CameraStreamManager()
        self._camera_stream_manager.streams_updated.connect(self._refresh_camera_table)

        self._media_device_manager: MediaDeviceManager = MediaDeviceManager()
        self._media_device_manager.camera_devices_update_signal.connect(
            self._on_camera_devices_update
        )

        self._camera_device_infos: list[dict[str, str]] = []
        self._preview_windows: dict[int, CameraPreviewWindow] = {}

        self.setWindowTitle("ESTV - ESTiVision-MP")
        self._setup_ui()


    def _setup_ui(self) -> None:
        """UIのセットアップを行う。"""
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("接続中カメラ一覧:"))

        # --- テーブルウィジェット
        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(4)
        self.camera_table.setHorizontalHeaderLabels(["ID", "名前", "状態", "操作"])
        self.camera_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.camera_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.setColumnWidth(0, 40)   # ID
        self.camera_table.setColumnWidth(1, 200)  # 名前
        self.camera_table.setColumnWidth(2, 60)   # 状態
        self.camera_table.setColumnWidth(3, 80)   # 操作（トグルボタン）
        self.camera_table.setFixedSize(480, 200)
        self.camera_table.setStyleSheet("""
            QTableWidget::item:hover {
                background: transparent;
            }
        """)

        main_layout.addWidget(self.camera_table)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def _on_camera_devices_update(self, camera_device_infos: list[dict[str, str]]) -> None:
        """カメラデバイス情報が更新されたときに呼ばれる。"""
        self._camera_device_infos = camera_device_infos
        self._refresh_camera_table()


    def _refresh_camera_table(self) -> None:
        """カメラ一覧テーブルを再描画する。"""
        running_ids = set(self._camera_stream_manager.running_device_ids())
        row_count = len(self._camera_device_infos)
        self.camera_table.setRowCount(row_count)

        for row, info in enumerate(self._camera_device_infos):
            device_id = row
            name = info.get("name", f"Camera {device_id}")
            running = device_id in running_ids

            # --- ID
            id_item = QTableWidgetItem(str(device_id))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.camera_table.setItem(row, 0, id_item)

            # --- 名前
            name_item = QTableWidgetItem(name)
            self.camera_table.setItem(row, 1, name_item)

            # --- 状態
            status_item = QTableWidgetItem("起動中" if running else "停止中")
            status_item.setTextAlignment(Qt.AlignCenter)
            color = QColor(SUCCESS_COLOR) if running else QColor(WARNING_COLOR)
            status_item.setForeground(color)
            self.camera_table.setItem(row, 2, status_item)

            # --- 操作ボタン
            btn = QPushButton("停止" if running else "起動")
            btn.setCheckable(True)
            btn.setChecked(running)
            btn.setMinimumWidth(60)
            btn.setStyleSheet("padding: 4px 12px; margin: 4px;")
            btn.clicked.connect(lambda checked, d=device_id: self._toggle_camera(d, checked))
            self.camera_table.setCellWidget(row, 3, btn)

            self.camera_table.setRowHeight(row, 32)


    def _toggle_camera(self, device_id: int, checked: bool) -> None:
        """カメラの起動・停止トグルイベント。"""
        if checked:
            self._camera_stream_manager.start_camera(device_id)
            # --- プレビューウィンドウを開く
            if device_id not in self._preview_windows:
                preview = CameraPreviewWindow(
                    self._camera_stream_manager, device_id=device_id, parent=self, on_closed=self._on_preview_closed
                )
                preview.show()
                self._preview_windows[device_id] = preview
        else:
            # --- プレビューウィンドウを閉じる
            if device_id in self._preview_windows:
                self._preview_windows[device_id].close()
            else:
                self._camera_stream_manager.stop_camera(device_id)


    def _on_preview_closed(self, device_id: int) -> None:
        """プレビューウィンドウが閉じられたとき呼ばれる。"""
        if device_id in self._preview_windows:
            del self._preview_windows[device_id]


    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウが閉じられたときの処理。"""
        for preview in list(self._preview_windows.values()):
            preview.close()
        self._preview_windows.clear()
        self._camera_stream_manager.shutdown()
        super().closeEvent(event)
