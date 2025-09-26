from __future__ import annotations

import os
from typing import Dict, List

from loguru import logger
from openai import OpenAI
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm

from evaluators.base import Evaluator
from local_evaluation import (
    attempt_api_call,
    log_response,
    parse_response,
    trim_predictions_to_max_token_length,
)


def _system_message() -> str:
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


class OpenAIEvaluator(Evaluator):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EVALUATION_MODEL_NAME", "gpt-4-0125-preview")
        self.client = OpenAI()

    def evaluate(
        self,
        queries: List[str],
        ground_truths_list: List[List[str]] | List[str],
        predictions: List[str],
    ) -> Dict[str, float | int]:
        n_miss, n_correct = 0, 0
        system_message = _system_message()

        # normalize ground_truths_list to list[list[str]]
        if ground_truths_list and isinstance(ground_truths_list[0], str):
            ground_truths_list = [[gt] for gt in ground_truths_list]  # type: ignore[assignment]

        for _idx, prediction in enumerate(tqdm(predictions, total=len(predictions), desc="Evaluating Predictions")):
            query = queries[_idx]
            ground_truths = ground_truths_list[_idx]

            prediction = trim_predictions_to_max_token_length(prediction).strip()
            prediction_lowercase = prediction.lower()

            if "i don't know" in prediction_lowercase:
                n_miss += 1
                continue

            accuracy = -1
            for ground_truth in ground_truths:
                ground_truth_lowercase = ground_truth.lower()
                messages = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                    },
                ]

                if prediction_lowercase == ground_truth_lowercase:
                    accuracy = 1
                    break
                elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
                    accuracy = 1
                    break
                elif ("invalid" in prediction_lowercase) ^ ("invalid" in ground_truth_lowercase):
                    accuracy = 0
                    continue
                else:
                    response = attempt_api_call(self.client, self.model_name, messages)
                    if response:
                        log_response(messages, response)
                        _, accuracy = parse_response(response)
                        if accuracy == 1:
                            break

            if accuracy == 1:
                n_correct += 1

        n = len(predictions)
        results = {
            "score": (2 * n_correct + n_miss) / n - 1,
            "accuracy": n_correct / n,
            "hallucination": (n - n_correct - n_miss) / n,
            "missing": n_miss / n,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
        }
        logger.info(results)
        return results


