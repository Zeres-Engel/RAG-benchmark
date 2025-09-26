from __future__ import annotations

import math
import re
from typing import Dict, List

from loguru import logger
from tqdm.auto import tqdm

from evaluators.base import Evaluator


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def _em(pred: str, refs: List[str]) -> int:
    p = _normalize(pred)
    return 1 if any(p == _normalize(r) for r in refs) else 0


def _anls(pred: str, refs: List[str]) -> float:
    # Average Normalized Levenshtein Similarity over refs, max per instance
    def lev(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                cur = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        return dp[n]

    p = _normalize(pred)
    if not refs:
        return 0.0
    scores = []
    for r in refs:
        r_ = _normalize(r)
        if not p and not r_:
            scores.append(1.0)
        else:
            d = lev(p, r_)
            m = max(len(p), len(r_))
            s = 1 - d / m if m > 0 else 0.0
            scores.append(s)
    return max(scores) if scores else 0.0


class OfflineEvaluator(Evaluator):
    def __init__(self, use_anls: bool = True):
        self.use_anls = use_anls

    def evaluate(
        self,
        queries: List[str],
        ground_truths_list: List[List[str]] | List[str],
        predictions: List[str],
    ) -> Dict[str, float | int]:
        # normalize ground truths to list[list[str]]
        if ground_truths_list and isinstance(ground_truths_list[0], str):
            ground_truths_list = [[gt] for gt in ground_truths_list]  # type: ignore[assignment]

        n_miss, n_correct = 0, 0
        n = len(predictions)

        for i in tqdm(range(n), desc="Evaluating Predictions (offline)"):
            pred = (predictions[i] or "").strip()
            refs = ground_truths_list[i]
            low = pred.lower()
            if "i don't know" in low:
                n_miss += 1
                continue

            # EM first; if not exact, use ANLS threshold 0.5 as correct (configurable later)
            if _em(pred, refs) == 1:
                n_correct += 1
            else:
                if self.use_anls and _anls(pred, refs) >= 0.5:
                    n_correct += 1

        results = {
            "score": (2 * n_correct + n_miss) / n - 1 if n else 0.0,
            "accuracy": n_correct / n if n else 0.0,
            "hallucination": (n - n_correct - n_miss) / n if n else 0.0,
            "missing": n_miss / n if n else 0.0,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
        }
        logger.info(results)
        return results


