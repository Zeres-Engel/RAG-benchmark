from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        queries: List[str],
        ground_truths_list: List[List[str]] | List[str],
        predictions: List[str],
    ) -> Dict[str, float | int]:
        """
        Compute metrics over predictions vs ground truths and return an aggregate dict.
        Required keys: score, accuracy, hallucination, missing, n_miss, n_correct,
        n_hallucination, total.
        """
        raise NotImplementedError


