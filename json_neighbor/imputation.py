from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def from_string(cls, imputation_strategy: str) -> ImputationStrategy:
        if imputation_strategy == "median":
            return MedianImputation()
        raise ValueError(f"Unknown imputation strategy: {imputation_strategy}")


class MedianImputation(ImputationStrategy):
    def impute(self, data: np.ndarray) -> np.ndarray:
        return np.nan_to_num(data, nan=np.nanmedian(data))
