# estv/gui/camera_preview_window.py
"""カメラプレビュー表示およびキャリブレーションを行うウィンドウ。"""

from collections.abc import Callable
from filelock import FileLock
import json
import logging
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt,
    QTimer,
    QObject,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from estv.devices.camera_calibrator import CameraCalibrator
from estv.devices.camera_stream_manager import CameraStreamManager
from estv.estimators.pose_drawer import draw_pose_landmarks
from estv.estimators.pose_estimator import PoseEstimator
from estv.gui.style_constants import (
    BACKGROUND_COLOR,
    SUBTEXT_COLOR,
    TEXT_COLOR,
    WARNING_COLOR,
)


logger = logging.getLogger(__name__)


# --- 定数
CALIB_IMAGES_REQUIRED = 20        # キャリブレーションに必要な枚数
CALIB_CAPTURE_INTERVAL_MS = 500   # キャプチャ間隔 (ミリ秒)


def _get_data_dir() -> Path:
    """exe化にも対応した data/ ディレクトリ取得"""
    # exeの場合はexeと同じ場所/data/
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).resolve().parents[3]
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def _calib_file_path(device_id: str) -> str:
    """デバイスIDからキャリブレーションファイルパスを返す（exe対応）"""
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", device_id)
    return str(_get_data_dir() / f"calib_{safe_id}.npz")


def _settings_file_path(device_id: str) -> str:
    """デバイスIDごとのカメラ設定ファイルパスを返す。"""
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", device_id)
    return str(_get_data_dir() / f"settings_{safe_id}.json")


class PoseEstimationWorker(QObject):
    """別スレッドで姿勢推定を実行するワーカー。"""

    result_ready = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._estimator = PoseEstimator()

    @Slot(object)
    def process_frame(self, frame: np.ndarray) -> None:
        landmarks = self._estimator.estimate(frame)
        self.result_ready.emit(landmarks)

    @Slot()
    def close(self) -> None:
        self._estimator.close()


class CameraPreviewWindow(QDialog):
    """カメラのプレビューウィンドウ＋キャリブレーション制御。"""

    inference_input = Signal(object)

    def __init__(
        self,
        camera_stream_manager: CameraStreamManager,
        device_id: str = "",
        device_name: str | None = None,
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
        self.device_id = device_id
        self.device_name = device_name or device_id
        self.setWindowTitle(f"ESTV - {self.device_name}")
        self.camera_stream_manager = camera_stream_manager
        self._on_closed = on_closed
        self._pose_estimation_enabled = False
        self._pose_thread: QThread | None = None
        self._pose_worker: PoseEstimationWorker | None = None
        self._last_landmarks: list | None = None
        self._inference_busy = False
        # MainWindow からの強制閉鎖時に推定状態を無視するフラグ
        self._force_close = False

        # --- キャリブ関連
        self.calibrator = CameraCalibrator()
        self.calibrating = False
        self.calibration_done = False
        self._frame_count = 0
        self._last_image = None
        self._first_frame_received = False

        # --- カメラ設定読み込み
        settings_path = _settings_file_path(self.device_id)
        self._exposure_value = -7
        self._brightness_value = 0
        if os.path.exists(settings_path):
            lock_path = settings_path + ".lock"
            try:
                with FileLock(lock_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                self._exposure_value = int(data.get("exposure", 0))
                self._brightness_value = int(data.get("brightness", 0))
            except Exception as e:
                logger.error("カメラ設定読み込み失敗: %s", e)

        # --- UI構成
        # --- Preview グループ
        self.image_label = QLabel("カメラ映像がここに表示されます")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            background-color: {BACKGROUND_COLOR};
            border-radius: 8px;
            color: {TEXT_COLOR};
        """)
        self.image_label.setFixedSize(480, 480)
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.image_label)
        preview_group.setLayout(preview_layout)

        # --- Calibration グループ
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, CALIB_IMAGES_REQUIRED)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m 枚")
        self.progress_bar.setFixedWidth(480)

        self.calib_button = QPushButton("キャリブレーション開始")
        self.calib_button.setCheckable(True)
        self.calib_button.setStyleSheet("padding: 6px 18px;")
        self.calib_button.setFixedWidth(480)
        self.calib_button.clicked.connect(self._on_calib_toggle)

        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()
        calib_layout.addWidget(self.status_label)
        calib_layout.addWidget(self.progress_bar)
        calib_layout.addWidget(self.calib_button)
        calib_group.setLayout(calib_layout)

        # --- Config グループ
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-13, -1)
        self.exposure_slider.setValue(self._exposure_value)
        self.exposure_slider.setFixedWidth(480)
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(self._brightness_value)
        self.brightness_slider.setFixedWidth(480)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)

        config_group = QGroupBox("Config")
        config_layout = QVBoxLayout()
        config_layout.addWidget(QLabel("露出"))
        config_layout.addWidget(self.exposure_slider)
        config_layout.addWidget(QLabel("明るさ"))
        config_layout.addWidget(self.brightness_slider)
        config_group.setLayout(config_layout)

        # --- 全体レイアウト
        layout = QVBoxLayout()
        layout.addWidget(preview_group)
        layout.addWidget(calib_group)
        layout.addWidget(config_group)
        self.setLayout(layout)
        self.adjustSize()
        self.setFixedSize(self.size())

        # --- シグナル接続
        self._on_image_ready_slot = lambda cid, img: self._on_image_ready(cid, img)
        self.camera_stream_manager.q_image_ready.connect(self._on_image_ready_slot)
        self.camera_stream_manager.frame_ready.connect(self._on_frame_ready)

        self.camera_stream_manager.set_exposure(self.device_id, self._exposure_value)
        self.camera_stream_manager.set_brightness(self.device_id, self._brightness_value)

        # --- キャリブレーションパラメータ自動ロード
        calib_path = _calib_file_path(self.device_id)
        if os.path.exists(calib_path):
            try:
                self.calibrator.load(calib_path)
                self.calibration_done = True
                self.progress_bar_value_on_load = True
                self.progress_bar.setValue(self.progress_bar.maximum())
            except Exception as e:
                logger.error("キャリブレーションパラメータ読み込み失敗: %s", e)
                self.progress_bar_value_on_load = False
        else:
            self.progress_bar_value_on_load = False
        
        # --- キャリブ用タイマー
        self._calib_timer = QTimer(self)
        self._calib_timer.setInterval(CALIB_CAPTURE_INTERVAL_MS)  # 0.5秒ごと
        self._calib_timer.timeout.connect(self._on_calib_frame_timer)

        self._update_status_label()
        # 最低１つキャリブ済みプレビューができたら、MainWindow のグローバルボタンを更新
        main_win = parent
        if hasattr(main_win, "_update_global_est_button_state"):
            main_win._update_global_est_button_state()
        if hasattr(main_win, "_update_stereo_button_state"):
            main_win._update_stereo_button_state()


    def _on_image_ready(self, device_id: str, qimg: QImage) -> None:
        if device_id != self.device_id:
            return

        # もし姿勢推定オンなら最新推定結果で骨格を重畳
        if (
            self._pose_estimation_enabled
            and self._last_image is not None
            and self._last_landmarks is not None
        ):
            img_pose = draw_pose_landmarks(self._last_image, self._last_landmarks)
            rgb = cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


    def _on_frame_ready(self, device_id: str, frame: np.ndarray) -> None:
        """キャリブレーション用に最新フレームを保持し推定を要求する。"""
        if device_id != self.device_id:
            return
        self._last_image = frame
        if not self._first_frame_received:
            self._first_frame_received = True
            main_win = self.parent()
            if hasattr(main_win, "_update_stereo_button_state"):
                main_win._update_stereo_button_state()
        if (
            self._pose_estimation_enabled
            and self._pose_worker is not None
            and not self._inference_busy
        ):
            self._inference_busy = True
            self.inference_input.emit(frame)


    def _on_pose_result(self, landmarks: list) -> None:
        """姿勢推定結果を受け取り保存する。"""
        self._last_landmarks = landmarks
        self._inference_busy = False


    @property
    def latest_frame(self) -> np.ndarray | None:
        """直近フレーム（ステレオキャリブレーション用に公開）"""
        return self._last_image


    def _on_exposure_changed(self, value: int) -> None:
        """露出スライダー変更時にカメラへ反映し設定を保存する。"""
        self.camera_stream_manager.set_exposure(self.device_id, float(value))
        self._save_settings()


    def _on_brightness_changed(self, value: int) -> None:
        """明るさスライダー変更時にカメラへ反映し設定を保存する。"""
        self.camera_stream_manager.set_brightness(self.device_id, float(value))
        self._save_settings()


    def _save_settings(self) -> None:
        """現在の露出と明るさ補正をデバイスごとに保存する。"""
        settings_path = _settings_file_path(self.device_id)
        lock_path = settings_path + ".lock"
        data = {
            "exposure": int(self.exposure_slider.value()),
            "brightness": int(self.brightness_slider.value()),
        }
        try:
            with FileLock(lock_path):
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
        except Exception as e:
            logger.error("カメラ設定保存失敗: %s", e)


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
                    self.status_label.setText(f"再投影誤差: {err:.3f}")
                else:
                    self.status_label.setText("再投影誤差: 計算不可")

                # --- プロジェクトルート直下のdata/に保存
                calib_path = _calib_file_path(self.device_id)
                self.calibrator.save(calib_path)
                logger.info("キャリブレーションパラメータを保存: %s", calib_path)
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
            # --- 既存のキャリブレーション結果があれば再読み込み
            calib_path = _calib_file_path(self.device_id)
            if os.path.exists(calib_path):
                try:
                    self.calibrator.load(calib_path)
                    self.calibration_done = True
                    self.progress_bar.setValue(self.progress_bar.maximum())
                except Exception as e:
                    logger.error("キャリブレーションパラメータ読み込み失敗: %s", e)
                    self.calibration_done = False
                    self.progress_bar.setValue(0)
            else:
                self.calibration_done = False
                self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(self.progress_bar.maximum())
        self._update_status_label()
        # キャリブ状態が変わったので、MainWindow のグローバル推定ボタンを更新
        main_win = self.parent()
        if hasattr(main_win, "_update_global_est_button_state"):
            main_win._update_global_est_button_state()
        if hasattr(main_win, "_update_stereo_button_state"):
            main_win._update_stereo_button_state()


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


    def set_pose_estimation_enabled(self, enabled: bool) -> None:
        """プレビュー単位の推定 ON/OFF（外部制御専用）"""
        if enabled and not self._pose_estimation_enabled:
            if self._pose_worker is None:
                self._pose_worker = PoseEstimationWorker()
                self._pose_thread = QThread(self)
                self._pose_worker.moveToThread(self._pose_thread)
                self._pose_worker.result_ready.connect(self._on_pose_result)
                self.inference_input.connect(self._pose_worker.process_frame)
                self._pose_thread.start()
        elif not enabled and self._pose_estimation_enabled:
            if self._pose_worker is not None and self._pose_thread is not None:
                self.inference_input.disconnect(self._pose_worker.process_frame)
                self._pose_worker.result_ready.disconnect(self._on_pose_result)
                self._pose_thread.quit()
                self._pose_thread.wait()
                self._pose_worker.close()
                self._pose_worker.deleteLater()
                self._pose_thread.deleteLater()
                self._pose_worker = None
                self._pose_thread = None
                self._last_landmarks = None
                self._inference_busy = False
        self._pose_estimation_enabled = enabled
        # 姿勢推定中は内部キャリブボタンを無効化
        self.calib_button.setEnabled(not enabled)


    def force_close(self) -> None:
        """MainWindow からの終了時に強制的に閉じるためのヘルパー"""
        self._force_close = True
        self.close()


    def closeEvent(self, event: QCloseEvent) -> None:
        """ウィンドウを閉じる際にストリームを停止し後片付けを行う。"""
        if self.calibrating:
            self._stop_calibration(cancel=True)
        # 推定中は閉じられない（ただし MainWindow からの強制閉鎖は除く）
        if self._pose_estimation_enabled and not self._force_close:
            QMessageBox.warning(
                self, "カメラを停止できません",
                "このカメラは姿勢推定を実行中です。\n"
                "先に姿勢推定を停止してください。"
            )
            event.ignore()
            return
        self.set_pose_estimation_enabled(False)
        self._force_close = False
        self.camera_stream_manager.q_image_ready.disconnect(self._on_image_ready_slot)
        self.camera_stream_manager.frame_ready.disconnect(self._on_frame_ready)
        self.camera_stream_manager.stop_camera(self.device_id)
        self._save_settings()
        if self._on_closed:
            self._on_closed(self.device_id)
        super().closeEvent(event)
