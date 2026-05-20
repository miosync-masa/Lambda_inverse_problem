"""
Anomaly scorer abstract base class.

Each concrete scorer takes the raw events and a populated
:class:`Lambda3Result` and returns a 1-D ndarray of per-event anomaly
scores. Scorers are independent — they may be combined or evaluated in
isolation by :mod:`lambda3_detector.scorers.score_integrator` or by
ablation studies.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..config import Lambda3Result


class AnomalyScorer(ABC):
    """Strategy-pattern base class for per-event anomaly scoring."""

    @abstractmethod
    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        """Return anomaly scores of shape (n_events,)."""
        raise NotImplementedError

    def __call__(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        return self.score(events, lambda3_result)
