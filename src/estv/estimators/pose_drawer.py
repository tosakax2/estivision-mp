# estv/estimators/pose_drawer.py

import cv2
import mediapipe as mp
import numpy as np

from estv.estimators.pose_estimator import PoseLandmark


# --- 定数
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

LINE_COLOR = (255, 255, 0)      # 線の色（BGR）
LINE_THICKNESS = 2              # 線の太さ

POINT_SIZE = 4                  # ポイントの一辺ピクセル数
POINT_COLOR = (255, 255, 255)   # ポイントの色（BGR）
POINT_EDGE_COLOR = (0, 0, 0)    # ポイントの外枠の色（BGR）
POINT_EDGE_THICKNESS = 2        # ポイントの外枠の太さ


def draw_pose_landmarks(
    image: np.ndarray,
    landmarks: list[PoseLandmark | None]
) -> np.ndarray:
    """画像にMediaPipe Pose骨格を描画して返す。"""

    if not landmarks:
        return image

    img = image.copy()
    h, w = img.shape[:2]

    # キーポイントのピクセル座標化
    points = [
        (int(lm.x * w), int(lm.y * h)) if lm is not None else None
        for lm in landmarks
    ]

    # --- 骨格ライン描画
    for idx1, idx2 in POSE_CONNECTIONS:
        pt1 = points[idx1] if idx1 < len(points) else None
        pt2 = points[idx2] if idx2 < len(points) else None
        if pt1 and pt2:
            cv2.line(img, pt1, pt2, LINE_COLOR, LINE_THICKNESS, lineType=cv2.LINE_AA)

    # --- キーポイント描画
    for pt in points:
        if pt:
            x, y = pt
            # 外枠
            cv2.rectangle(
                img,
                (x - POINT_SIZE // 2 - 1, y - POINT_SIZE // 2 - 1),
                (x + POINT_SIZE // 2 + 1, y + POINT_SIZE // 2 + 1),
                POINT_EDGE_COLOR,
                thickness=POINT_EDGE_THICKNESS,
                lineType=cv2.LINE_AA
            )
            # 本体
            cv2.rectangle(
                img,
                (x - POINT_SIZE // 2, y - POINT_SIZE // 2),
                (x + POINT_SIZE // 2, y + POINT_SIZE // 2),
                POINT_COLOR,
                thickness=-1,
                lineType=cv2.LINE_AA
            )

    return img
