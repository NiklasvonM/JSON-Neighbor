from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import StandardScaler


class ScalingStrategy(ABC):
    @abstractmethod
    def scale(self, data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def from_string(cls, scaling_strategy: str) -> ScalingStrategy:
        if scaling_strategy == "standard":
            return StandardScaling()
        raise ValueError(f"Unknown scaling strategy: {scaling_strategy}")


class StandardScaling(ScalingStrategy):
    scaler: StandardScaler

    def __init__(self):
        self.scaler = StandardScaler()

    def scale(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)
