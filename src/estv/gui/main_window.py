# estv/gui/main_window.py

import re
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from estv.devices.camera_stream_manager import CameraStreamManager
from estv.devices.media_device_manager import MediaDeviceManager
from estv.devices.stereo_calibrator import StereoParams, stereo_calibrate
from estv.gui.camera_preview_window import (
    CameraPreviewWindow,
    _calib_file_path,
    _get_data_dir,
)
from estv.gui.style_constants import (
    SUCCESS_COLOR,
    WARNING_COLOR,
)


def _stereo_file_path(cam1_id: str, cam2_id: str) -> str:
    """IDペアから外部キャリブファイルパスを生成する。"""
    safe1 = re.sub(r"[^A-Za-z0-9._-]", "_", cam1_id)
    safe2 = re.sub(r"[^A-Za-z0-9._-]", "_", cam2_id)
    if safe1 > safe2:
        safe1, safe2 = safe2, safe1
    return str(_get_data_dir() / f"stereo_{safe1}_{safe2}.npz")


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
        self._stereo_params: StereoParams | None = None

        self._estimation_active: bool = False

        self.setWindowTitle("ESTV - ESTiVision-MP")
        self._setup_ui()


    def _setup_ui(self) -> None:
        """ウィンドウ内のウィジェットとレイアウトを構成する。"""
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # --- Devicesグループボックス
        devices_group = QGroupBox("Devices")
        devices_layout = QVBoxLayout()

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(4)
        self.camera_table.setHorizontalHeaderLabels(["ID", "名前", "状態", "操作"])
        self.camera_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.camera_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.setColumnWidth(0, 40)
        self.camera_table.setColumnWidth(1, 240)
        self.camera_table.setColumnWidth(2, 60)
        self.camera_table.setColumnWidth(3, 120)
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

        devices_layout.addWidget(self.camera_table)
        devices_group.setLayout(devices_layout)


        # --- Calibrationグループボックス
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout()
        self.stereo_status_label = QLabel("未キャリブレーション")
        self.stereo_calib_button = QPushButton("ステレオキャリブレーション開始")
        self.stereo_calib_button.setEnabled(False)
        calibration_layout.addWidget(self.stereo_status_label)
        calibration_layout.addWidget(self.stereo_calib_button)
        calibration_group.setLayout(calibration_layout)
        self.stereo_calib_button.clicked.connect(self._start_stereo_calibration)

        # --- Estimationグループボックス
        estimation_group = QGroupBox("Estimation")
        estimation_layout = QVBoxLayout()
        self.global_est_button = QPushButton("姿勢推定開始")
        self.global_est_button.setCheckable(True)
        self.global_est_button.setEnabled(False)
        self.global_est_button.clicked.connect(self._toggle_global_estimation)
        self._launch_enabled = True
        estimation_layout.addWidget(self.global_est_button)
        estimation_group.setLayout(estimation_layout)

        # --- 全体レイアウトに追加
        main_layout.addWidget(devices_group)
        main_layout.addWidget(calibration_group)
        main_layout.addWidget(estimation_group)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.adjustSize()
        self.setFixedSize(self.size())


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

            self._update_stereo_button_state()
            self._update_start_buttons_state()


    def _toggle_camera(self, device_id: str, checked: bool) -> None:
        """トグルボタン押下でカメラを開始・停止する。"""
        # --- 推定中は新しいカメラを起動させない
        if checked and not self._launch_enabled:
            return  # 起動要求を無視
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
        running_count = len(self._camera_stream_manager.running_device_ids())
        max_streams = self._camera_stream_manager.max_streams
        for row in range(rows):
            btn = self.camera_table.cellWidget(row, 3)
            if btn is None:
                continue
            running = btn.isChecked()
            if not self._launch_enabled:
                btn.setEnabled(False)
            elif not running and running_count >= max_streams:
                btn.setEnabled(False)
            else:
                btn.setEnabled(True)


    def _on_preview_closed(self, device_id: str) -> None:
        """プレビューウィンドウが閉じられた際の後処理。"""
        if device_id in self._preview_windows:
            del self._preview_windows[device_id]
        self._update_stereo_button_state()
        self._update_global_est_button_state()


    def _update_stereo_button_state(self) -> None:
        """内部キャリブ済み & 起動中カメラが 2 台以上ならボタンを有効化."""
        ready = [
            p
            for p in self._preview_windows.values()
            if p.calibration_done and p.latest_frame is not None
        ]
        self.stereo_calib_button.setEnabled(len(ready) >= 2)
        if len(ready) >= 2:
            cam1, cam2 = ready[:2]
            path = Path(_stereo_file_path(cam1.device_id, cam2.device_id))
            if path.exists():
                try:
                    self._stereo_params = StereoParams.load(path)
                    self.stereo_status_label.setText(
                        f"RMS: {self._stereo_params.rms:.3f}"
                    )
                except Exception:  # pylint: disable=broad-except
                    self._stereo_params = None
                    self.stereo_status_label.setText("未キャリブレーション")
            else:
                self._stereo_params = None
                self.stereo_status_label.setText("未キャリブレーション")
        else:
            self._stereo_params = None
            self.stereo_status_label.setText("未キャリブレーション")

    def _start_stereo_calibration(self) -> None:
        """5 秒カウント後に 2 台同時撮影し外部パラメータを推定."""
        # 対象カメラを固定（先頭 2 台）
        targets = [
            p for p in self._preview_windows.values() if p.calibration_done
        ][:2]
        if len(targets) < 2:
            return

        # --- カウントダウン用ポップアップ
        popup = QDialog(self)
        popup.setWindowTitle("Stereo Calibration")
        vbox = QVBoxLayout(popup)
        label = QLabel("5", alignment=Qt.AlignCenter)
        label.setStyleSheet("font-size: 48pt;")
        vbox.addWidget(label)
        popup.setFixedSize(200, 150)
        popup.show()

        counter = {"sec": 5}

        def _tick():
            counter["sec"] -= 1
            if counter["sec"] == 0:
                timer.stop()
                popup.accept()  # 閉じる
                self._run_stereo_calibration(targets)
            else:
                label.setText(str(counter["sec"]))

        timer = QTimer(popup)
        timer.setInterval(1000)
        timer.timeout.connect(_tick)
        timer.start()


    def _run_stereo_calibration(self, previews: list) -> None:
        cam1, cam2 = previews
        img1, img2 = cam1.latest_frame, cam2.latest_frame
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "撮影失敗", "画像を取得できませんでした。")
            return

        calib_path1 = Path(_calib_file_path(cam1.device_id))
        calib_path2 = Path(_calib_file_path(cam2.device_id))
        stereo_path = Path(_stereo_file_path(cam1.device_id, cam2.device_id))

        try:
            data1 = np.load(str(calib_path1))
            data2 = np.load(str(calib_path2))
            k1, d1 = data1["camera_matrix"], data1["dist_coeffs"]
            k2, d2 = data2["camera_matrix"], data2["dist_coeffs"]

            board_size = (6, 9)
            square_size_m = 0.02

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            found1, corners1 = cv2.findChessboardCorners(gray1, board_size, None)
            found2, corners2 = cv2.findChessboardCorners(gray2, board_size, None)
            if not (found1 and found2):
                raise ValueError("チェスボードが検出できませんでした。")

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            image_size = (img1.shape[1], img1.shape[0])

            params = stereo_calibrate(
                [corners1],
                [corners2],
                board_size,
                square_size_m,
                k1,
                d1,
                k2,
                d2,
                image_size,
                cam1.device_id,
                cam2.device_id,
            )
            params.save(stereo_path)
            self._stereo_params = params
            QMessageBox.information(
                self,
                "ステレオキャリブレーション完了",
                f"推定終了: RMS 誤差 = {params.rms:.3f}",
            )
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(
                self,
                "ステレオキャリブレーション失敗",
                str(exc),
            )

        # ボタンを再び有効化
        self._update_stereo_button_state()


    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウを閉じる際にすべてのストリームを停止する。"""
        for preview in list(self._preview_windows.values()):
            preview.force_close()
        self._preview_windows.clear()
        self._camera_stream_manager.shutdown()
        super().closeEvent(event)
