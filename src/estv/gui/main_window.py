# estv/gui/main_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QAbstractItemView, QLabel, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCloseEvent

from estv.devices.media_device_manager import MediaDeviceManager
from estv.devices.camera_stream_manager import CameraStreamManager
from estv.gui.camera_preview_window import CameraPreviewWindow
from estv.gui.style_constants import SUCCESS_COLOR, WARNING_COLOR


class MainWindow(QMainWindow):
    """接続されたカメラを一覧表示するメインウィンドウ。"""

    def __init__(self) -> None:
        """メインウィンドウを生成しデバイスマネージャーを設定する。"""
        super().__init__()

        self._media_device_manager: MediaDeviceManager = MediaDeviceManager()
        self._camera_stream_manager: CameraStreamManager = CameraStreamManager(
            self._media_device_manager.camera_index_by_id
        )
        self._camera_stream_manager.streams_updated.connect(self._refresh_camera_table)
        self._media_device_manager.camera_devices_update_signal.connect(
            self._on_camera_devices_update
        )

        self._camera_device_infos: list[dict[str, str]] = []
        self._preview_windows: dict[str, CameraPreviewWindow] = {}

        self._estimation_active: bool = False

        self.setWindowTitle("ESTV - ESTiVision-MP")
        self._setup_ui()


    def _setup_ui(self) -> None:
        """ウィンドウ内のウィジェットとレイアウトを構成する。"""
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
        self.camera_table.setColumnWidth(1, 240)  # 名前
        self.camera_table.setColumnWidth(2, 60)   # 状態
        self.camera_table.setColumnWidth(3, 120)  # 操作（トグルボタン）
        self.camera_table.setFixedSize(480, 200)
        self.camera_table.setStyleSheet("""
            QTableWidget::item {
                background: transparent;
            }
            QTableWidget::item:hover {
                background: transparent;
                color: auto;
            }
            QTableWidget::item:selected {
                background: transparent;
                color: auto;
            }
        """)

        # --- 全カメラ用 姿勢推定トグル
        self.global_est_button = QPushButton("姿勢推定開始")
        self.global_est_button.setCheckable(True)
        self.global_est_button.setEnabled(False)
        self.global_est_button.clicked.connect(self._toggle_global_estimation)
        self._launch_enabled = True

        main_layout.addWidget(self.camera_table)
        main_layout.addWidget(self.global_est_button)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def _on_camera_devices_update(self, camera_device_infos: list[dict[str, str]]) -> None:
        """利用可能なカメラデバイス一覧の更新を処理する。"""
        self._camera_device_infos = camera_device_infos
        self._refresh_camera_table()


    def _refresh_camera_table(self) -> None:
        """カメラ状態テーブルを再描画する。"""
        running_ids = set(self._camera_stream_manager.running_device_ids())
        row_count = len(self._camera_device_infos)
        self.camera_table.setRowCount(row_count)

        for row, info in enumerate(self._camera_device_infos):
            camera_id = info.get("id", str(row))
            name = info.get("name", f"Camera {row}")
            running = camera_id in running_ids

            # --- ID
            id_item = QTableWidgetItem(str(row))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.camera_table.setItem(row, 0, id_item)

            # --- 名前
            name_item = QTableWidgetItem(name)
            self.camera_table.setItem(row, 1, name_item)

            # --- 状態
            status_item = QTableWidgetItem("推定中" if self._estimation_active and running else ("起動中" if running else "停止中"))
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
            if self._estimation_active:
                btn.setEnabled(False)
            btn.clicked.connect(lambda checked, cid=camera_id: self._toggle_camera(cid, checked))
            self.camera_table.setCellWidget(row, 3, btn)
            self.camera_table.setRowHeight(row, 32)

            self._update_start_buttons_state()


    def _toggle_camera(self, device_id: str, checked: bool) -> None:
        """トグルボタン押下でカメラを開始・停止する。"""
        # --- 推定中は新しいカメラを起動させない
        if checked and not self._launch_enabled:
            return  # 起動要求を無視
        if self._estimation_active:
            QMessageBox.warning(
                self, "操作できません",
                "このカメラは姿勢推定を実行中です。\n先に姿勢推定を停止してください。"
            )
            return
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
        self._update_global_est_button_state()


    def _update_global_est_button_state(self) -> None:
        """キャリブ済みプレビューが1つでもあればボタンを有効化。"""
        has_ready = any(p.calibration_done for p in self._preview_windows.values())
        self.global_est_button.setEnabled(has_ready)


    def _toggle_global_estimation(self, checked: bool) -> None:
        """起動済み＆キャリブレーション済みカメラの推定を一括 ON/OFF"""
        self._estimation_active = checked
        for preview in self._preview_windows.values():
            if preview.calibration_done: # キャリブ済みのみ対象
                preview.set_pose_estimation_enabled(checked)

        # --- ボタン表示を更新
        self.global_est_button.setText("姿勢推定停止" if checked else "姿勢推定開始")
        self._refresh_camera_table()

        # --- 推定中は新規カメラ起動を禁止
        self._launch_enabled = not checked
        self._update_start_buttons_state()


    def _update_start_buttons_state(self) -> None:
        """行ごとの '起動／停止' ボタンの有効・無効を更新する。"""
        rows = self.camera_table.rowCount()
        for row in range(rows):
            btn = self.camera_table.cellWidget(row, 3)
            if btn is None:
                continue
            running = btn.isChecked()
            # 起動していないボタンのみ、推定中は無効化
            btn.setEnabled(self._launch_enabled or running)


    def _on_preview_closed(self, device_id: str) -> None:
        """プレビューウィンドウが閉じられた際の後処理。"""
        if device_id in self._preview_windows:
            del self._preview_windows[device_id]
        self._update_global_est_button_state()


    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウを閉じる際にすべてのストリームを停止する。"""
        for preview in list(self._preview_windows.values()):
            preview.close()
        self._preview_windows.clear()
        self._camera_stream_manager.shutdown()
        super().closeEvent(event)
