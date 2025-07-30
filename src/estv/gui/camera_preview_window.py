# estv/gui/camera_preview_window.py
"""カメラプレビュー表示およびキャリブレーションを行うウィンドウ。"""

from collections.abc import Callable
import os
import re

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QProgressBar,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QCloseEvent

from estv.devices.camera_stream_manager import CameraStreamManager
from estv.devices.camera_calibrator import CameraCalibrator
from estv.gui.style_constants import (
    TEXT_COLOR,
    BACKGROUND_COLOR,
    WARNING_COLOR,
    SUBTEXT_COLOR,
)

# --- 定数
CALIB_IMAGES_REQUIRED = 20        # キャリブレーションに必要な枚数
CALIB_CAPTURE_INTERVAL_MS = 500   # キャプチャ間隔 (ミリ秒)


def _calib_file_path(device_id: str) -> str:
    """デバイスIDからキャリブレーションファイルパスを返す。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", device_id)
    return os.path.join(data_dir, f"calib_{safe_id}.npz")


class CameraPreviewWindow(QDialog):
    """カメラのプレビューウィンドウ＋キャリブレーション制御。"""

    def __init__(
        self,
        camera_stream_manager: CameraStreamManager,
        device_id: str = "",
        parent: QWidget | None = None,
        on_closed: Callable[[str], None] | None = None,
    ) -> None:
        """指定したカメラのプレビュー用ダイアログを生成する。

        パラメータ
        ----------
        camera_stream_manager : CameraStreamManager
            カメラフレームを提供するマネージャー。
        device_id : str, optional
            プレビュー対象のカメラ識別子。
        parent : QWidget | None, optional
            親ウィジェット。
        on_closed : Callable[[str], None] | None, optional
            ウィンドウ閉鎖時に ``device_id`` を引数に呼び出されるコールバック。
        """
        super().__init__(parent)
        self.setWindowTitle("ESTV - カメラプレビュー")
        self.device_id = device_id
        self.camera_stream_manager = camera_stream_manager
        self._on_closed = on_closed

        # --- キャリブ関連
        self.calibrator = CameraCalibrator()
        self.calibrating = False
        self.calibration_done = False
        self._frame_count = 0
        self._last_image = None

        # --- キャリブレーションパラメータ自動ロード
        calib_path = _calib_file_path(self.device_id)
        if os.path.exists(calib_path):
            try:
                self.calibrator.load(calib_path)
                self.calibration_done = True
                self.progress_bar_value_on_load = True
            except Exception as e:
                print(f"キャリブレーションパラメータ読み込み失敗: {e}")
                self.progress_bar_value_on_load = False
        else:
            self.progress_bar_value_on_load = False

        # --- UI構成
        self.image_label = QLabel("カメラ映像がここに表示されます")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            background-color: {BACKGROUND_COLOR};
            border-radius: 8px;
            color: {TEXT_COLOR};
        """)
        self.image_label.setFixedSize(480, 480)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, CALIB_IMAGES_REQUIRED)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m 枚")
        self.progress_bar.setFixedWidth(480)
        if self.progress_bar_value_on_load:
            self.progress_bar.setValue(self.progress_bar.maximum())

        self.calib_button = QPushButton("キャリブレーション開始")
        self.calib_button.setCheckable(True)
        self.calib_button.setStyleSheet("padding: 6px 18px;")
        self.calib_button.setFixedWidth(480)
        self.calib_button.clicked.connect(self._on_calib_toggle)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.calib_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)
        self.adjustSize()
        self.setFixedSize(self.size())

        # --- シグナル接続
        self._on_image_ready_slot = lambda cid, img: self._on_image_ready(cid, img)
        self.camera_stream_manager.q_image_ready.connect(self._on_image_ready_slot)
        self.camera_stream_manager.frame_ready.connect(self._on_frame_ready)
        
        # --- キャリブ用タイマー
        self._calib_timer = QTimer(self)
        self._calib_timer.setInterval(CALIB_CAPTURE_INTERVAL_MS)  # 0.5秒ごと
        self._calib_timer.timeout.connect(self._on_calib_frame_timer)

        self._update_status_label()


    def _on_image_ready(self, device_id: str, qimg: QImage) -> None:
        """このデバイス用の新しいプレビュー画像を処理する。

        パラメータ
        ----------
        device_id : str
            ``qimg`` を生成したカメラの識別子。
        qimg : QImage
            Qt 表示用に変換されたフレーム。
        """
        if device_id != self.device_id:
            return
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


    def _on_frame_ready(self, device_id: str, frame: np.ndarray) -> None:
        """キャリブレーション用に最新フレームを保持する。"""
        if device_id == self.device_id:
            self._last_image = frame


    def _on_calib_toggle(self, checked: bool) -> None:
        """キャリブレーションモードを開始・停止する。"""
        if checked:
            self.calib_button.setText("キャリブレーション停止")
            self.status_label.setStyleSheet(f"color: {SUBTEXT_COLOR};")
            self.status_label.setText("チェスボード画像を自動取得中...")
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.calibrating = True
            self._frame_count = 0
            self.calibrator = CameraCalibrator()
            self._calib_timer.start()
        else:
            self._stop_calibration(cancel=True)


    def _on_calib_frame_timer(self) -> None:
        """定期的にキャリブレーション用フレームを取得する。"""
        if not self.calibrating or self._last_image is None:
            return
        # --- チェスボード検出
        found = self.calibrator.add_chessboard_image(self._last_image)
        self._frame_count += 1
        if found:
            self.progress_bar.setValue(len(self.calibrator.image_points))

        # --- 十分な枚数集まったらキャリブレーション実行
        if len(self.calibrator.image_points) >= CALIB_IMAGES_REQUIRED:
            try:
                self.calibrator.calibrate(self._last_image.shape[:2])
                self.calibration_done = True
                err = self.calibrator.reprojection_error
                self.status_label.setStyleSheet(f"color: {TEXT_COLOR};")
                if err is not None:
                    self.status_label.setText(f"平均再投影誤差: {err:.3f}")
                else:
                    self.status_label.setText("平均再投影誤差: 計算不可")

                # --- プロジェクトルート直下のdata/に保存
                calib_path = _calib_file_path(self.device_id)
                self.calibrator.save(calib_path)
                print(f"キャリブレーションパラメータを保存: {calib_path}")
            except Exception as e:
                self.status_label.setStyleSheet(f"color: {WARNING_COLOR};")
                self.status_label.setText(f"キャリブレーション失敗: {str(e)}")
            self._stop_calibration(cancel=False)


    def _stop_calibration(self, cancel: bool = False) -> None:
        """キャリブレーション処理を停止し UI をリセットする。"""
        self.calibrating = False
        self._calib_timer.stop()
        self.calib_button.setChecked(False)
        self.calib_button.setText("キャリブレーション開始")
        if cancel:
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(self.progress_bar.maximum())
        self._update_status_label()
        if cancel:
            self.status_label.setText("キャリブレーションが必要です")


    def _update_status_label(self) -> None:
        """キャリブレーション状態に応じてステータス表示を更新する。"""
        if self.calibration_done:
            err = self.calibrator.reprojection_error
            self.status_label.setStyleSheet(f"color: {TEXT_COLOR};")
            if err is not None:
                self.status_label.setText(f"平均再投影誤差: {err:.3f}")
            else:
                self.status_label.setText("平均再投影誤差: 計算失敗")
        else:
            self.status_label.setStyleSheet(f"color: {WARNING_COLOR};")
            self.status_label.setText("キャリブレーションが必要です")


    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウを閉じる際にストリームを停止し後片付けを行う。"""
        if self.calibrating:
            self._stop_calibration(cancel=True)
        self.camera_stream_manager.q_image_ready.disconnect(self._on_image_ready_slot)
        self.camera_stream_manager.frame_ready.disconnect(self._on_frame_ready)
        self.camera_stream_manager.stop_camera(self.device_id)
        if self._on_closed:
            self._on_closed(self.device_id)
        super().closeEvent(event)
