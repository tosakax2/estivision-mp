from abc import ABC, abstractmethod

import numpy as np

from estv.estimators.pose_estimator import PoseLandmark


class VirtualTrackerResult:

    def __init__(self, name: str, position: np.ndarray, rotation: np.ndarray):
        self.name = name  # e.g. "Hips", "Head"
        self.position = position  # shape: (3,)
        self.rotation = rotation  # shape: (4,) quaternion


class BaseTracker(ABC):

    @abstractmethod
    def update(self, landmarks: list[PoseLandmark | None]) -> list[VirtualTrackerResult]:
        """与えられた姿勢推定ランドマークから仮想トラッカー情報を計算して返す。"""
        raise NotImplementedError
