# estv/estimators/pose_estimators.py

import mediapipe as mp
import numpy as np


class PoseLandmark:
    """推論結果として返すキーポイント情報"""

    def __init__(self, x: float, y: float, z: float, visibility: float) -> None:
        self.x = x  # 正規化座標（0～1、画像幅基準）
        self.y = y  # 正規化座標（0～1、画像高さ基準）
        self.z = z  # カメラからの相対奥行き（任意スケール）
        self.visibility = visibility  # 信頼度（0～1）


    def as_tuple(self) -> tuple:
        return self.x, self.y, self.z, self.visibility


class PoseEstimator:
    """MediaPipe Poseを用いた姿勢推定モジュール（CPU動作）"""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )


    def estimate(self, frame: np.ndarray) -> list[PoseLandmark]:
        """BGR画像（np.ndarray）を受け取り、ランドマーク情報リストを返す"""
        # MediaPipeはRGB入力が必要
        image_rgb = frame[..., ::-1]

        # 推論
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return []

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(
                PoseLandmark(lm.x, lm.y, lm.z, lm.visibility)
            )
        return landmarks


    def close(self) -> None:
        """リソース解放"""
        self.pose.close()
