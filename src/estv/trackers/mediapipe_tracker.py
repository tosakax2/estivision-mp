from typing import Optional

import numpy as np

from estv.estimators.pose_estimator import PoseLandmark
from estv.trackers.tracker_base import BaseTracker, VirtualTrackerResult


class MediaPipeVirtualTracker(BaseTracker):
    """MediaPipeのランドマークから仮想トラッカー(腰・胸・両足・両肘)を生成"""

    def __init__(self):
        pass


    def update(self, landmarks: list[Optional[PoseLandmark]]) -> list[VirtualTrackerResult]:
        results = []

        # --- 1. 腰（左右hipの中点）
        hips = [23, 24]
        hip_pts = [landmarks[i] for i in hips if i < len(landmarks) and landmarks[i] is not None]
        if len(hip_pts) == 2:
            hip_pos = np.mean([[lm.x, lm.y, lm.z] for lm in hip_pts], axis=0)
            results.append(VirtualTrackerResult("Hips", hip_pos, np.array([0, 0, 0, 1])))

        # --- 2. 胸（左右shoulderの中点）
        shoulders = [11, 12]
        shoulder_pts = [landmarks[i] for i in shoulders if i < len(landmarks) and landmarks[i] is not None]
        if len(shoulder_pts) == 2:
            chest_pos = np.mean([[lm.x, lm.y, lm.z] for lm in shoulder_pts], axis=0)
            results.append(VirtualTrackerResult("Chest", chest_pos, np.array([0, 0, 0, 1])))

        # --- 3. 両足首
        for side, idx in [("LeftFoot", 27), ("RightFoot", 28)]:
            if idx < len(landmarks) and landmarks[idx] is not None:
                lm = landmarks[idx]
                results.append(VirtualTrackerResult(side, np.array([lm.x, lm.y, lm.z]), np.array([0, 0, 0, 1])))

        # --- 4. 両肘
        for side, idx in [("LeftElbow", 13), ("RightElbow", 14)]:
            if idx < len(landmarks) and landmarks[idx] is not None:
                lm = landmarks[idx]
                results.append(VirtualTrackerResult(side, np.array([lm.x, lm.y, lm.z]), np.array([0, 0, 0, 1])))
        return results
