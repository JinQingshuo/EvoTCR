"""Training API for new factor plugins; labels are scoped before fitting."""

from dataclasses import dataclass
from typing import Protocol, Any
import numpy as np


@dataclass(frozen=True)
class FitView:
    features: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    training_fingerprint: str

    @classmethod
    def subset(cls, features, labels, row_ids, validation_ids, fingerprint):
        row_ids = np.asarray(row_ids, dtype=int)
        if len(set(row_ids)) != len(row_ids) or set(row_ids) & set(validation_ids):
            raise ValueError("Training IDs must be unique and disjoint from validation")
        if np.any(row_ids < 0) or np.any(row_ids >= len(features)):
            raise ValueError("Training index out of bounds")
        x = np.array(features[row_ids], copy=True)
        y = np.array(labels[row_ids], copy=True)
        x.flags.writeable = False
        y.flags.writeable = False
        return cls(x, y, row_ids.copy(), fingerprint)


class TrainableFactor(Protocol):
    """Serialize state without pickle; include config and fit-view fingerprint."""

    def fit(self, training: FitView, config: dict) -> dict[str, Any]: ...
    def transform(self, features: np.ndarray, state: dict) -> np.ndarray: ...
