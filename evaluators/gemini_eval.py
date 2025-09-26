from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from loguru import logger
from tqdm.auto import tqdm

from evaluators.base import Evaluator

# Prefer new google-genai SDK: from google import genai
_genai_client = None
try:  # new SDK
    from google import genai as google_genai  # type: ignore
    _sdk = "google-genai"
except Exception:
    google_genai = None  # type: ignore
    _sdk = None

if _sdk is None:
    try:  # fallback: old google.generativeai
        import google.generativeai as generativeai  # type: ignore
        _sdk = "google-generativeai"
    except Exception:
        generativeai = None  # type: ignore
        _sdk = None


SYSTEM_PROMPT = (
    "You are an accuracy judge. Given a Question, Ground truth, and Prediction, "
    "return a JSON object with fields: score (0 or 1) and explanation (string). "
    "Score 1 if the prediction is correct or semantically equivalent; otherwise 0."
)


def _load_gemini_api_key() -> Optional[str]:
    # Priority: env var → keys file
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    # try load from data/keys/gemini_keys.json
    keys_path = os.path.join("data", "keys", "gemini_keys.json")
    try:
        with open(keys_path, "r") as f:
            data = json.load(f)
        for entry in data.get("api_keys", []):
            if entry.get("active") and entry.get("key"):
                return entry["key"]
    except Exception:
        pass
    return None


class GeminiEvaluator(Evaluator):
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        if _sdk is None:
            raise ImportError(
                "Gemini SDK not installed. Install 'google-genai' (preferred) or 'google-generativeai'."
            )

        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        api_key = api_key or _load_gemini_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set and no active key found in data/keys/gemini_keys.json")

        if _sdk == "google-genai":
            self.client = google_genai.Client(api_key=api_key)  # type: ignore
            self.mode = "new"
        else:
            generativeai.configure(api_key=api_key)  # type: ignore
            self.client = generativeai.GenerativeModel(self.model_name)  # type: ignore
            self.mode = "old"

    def _judge_once(self, question: str, ground_truth: str, prediction: str) -> int:
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Prediction: {prediction}\n"
            f"Return JSON only."
        )
        if self.mode == "new":
            resp = self.client.models.generate_content(model=self.model_name, contents=prompt)  # type: ignore
            text = getattr(resp, "text", None) or "{}"
        else:
            resp = self.client.generate_content(prompt)  # type: ignore
            text = getattr(resp, "text", None) or "{}"
        try:
            obj = json.loads(text)
            return int(obj.get("score", 0))
        except Exception:
            return 0

    def evaluate(
        self,
        queries: List[str],
        ground_truths_list: List[List[str]] | List[str],
        predictions: List[str],
    ) -> Dict[str, float | int]:
        if ground_truths_list and isinstance(ground_truths_list[0], str):
            ground_truths_list = [[gt] for gt in ground_truths_list]  # type: ignore[assignment]

        n_miss, n_correct = 0, 0
        for i, pred in enumerate(tqdm(predictions, total=len(predictions), desc="Evaluating (Gemini)")):
            pl = (pred or "").lower()
            if "i don't know" in pl:
                n_miss += 1
                continue
            q = queries[i]
            gts = ground_truths_list[i]
            acc = 0
            for gt in gts:
                if (pred or "").strip().lower() == gt.strip().lower():
                    acc = 1
                    break
                acc = self._judge_once(q, gt, pred or "")
                if acc == 1:
                    break
            if acc == 1:
                n_correct += 1

        n = len(predictions)
        return {
            "score": (2 * n_correct + n_miss) / n - 1 if n else 0.0,
            "accuracy": n_correct / n if n else 0.0,
            "hallucination": (n - n_correct - n_miss) / n if n else 0.0,
            "missing": n_miss / n if n else 0.0,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
        }


