import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

import sys
if 'pipeline' not in sys.path:
    sys.path.insert(0, 'pipeline')

from retrieval_augmented_generation.documents_manager import DocumentsManager
from retrieval_augmented_generation.document_loaders import InputDocument, InputSource, DocumentType
import retrieval_augmented_generation.vectorstores.qdrant_vectorstore  # noqa: F401
import retrieval_augmented_generation.embeddings.sentence_transformer_embeddings  # noqa: F401
import retrieval_augmented_generation.document_loaders.text_processor  # noqa: F401
import retrieval_augmented_generation.document_loaders.pdf_processor  # noqa: F401
import retrieval_augmented_generation.document_loaders.url_processor  # noqa: F401


class Evaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.doc_manager = DocumentsManager(self.config, self.logger)

    def _load_config(self, config_path: str) -> dict:
        candidates = [Path(config_path)]
        # project root locations
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / 'config.yaml')
        candidates.append(project_root / 'config' / 'config.yaml')
        # cwd fallbacks
        candidates.append(Path('config.yaml').resolve())
        candidates.append((Path.cwd() / 'config' / 'config.yaml').resolve())
        for p in candidates:
            if p.is_file():
                with p.open('r') as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError("Config not found. Tried: " + ", ".join(str(p) for p in candidates))

    def _setup_logger(self):
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    async def build_collection(self, df: pd.DataFrame, collection: str) -> None:
        await self.doc_manager.create_collection(
            collection_name=collection,
            model_name="Qwen/Qwen3-Embedding-0.6B",
            recreate=True
        )

        for i, row in df.iterrows():
            text = str(row.get('Gold_REF') or '').strip()
            if not text or text.lower() == 'nan':
                continue
            input_doc = InputDocument(
                content=text,
                type=DocumentType.TEXT,
                source=InputSource.RAW_TEXT,
                metadata={"row_index": int(i)}
            )
            await self.doc_manager.add_document(collection_name=collection, document=input_doc)

    async def evaluate(self, dataset_path: str, out_dir: str = "results", top_k: int = 3) -> Dict:
        df = pd.read_csv(dataset_path)
        collection = "eval_collection"
        await self.build_collection(df, collection)

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Inference JSON (per question with top1..topk)
        inference_items: List[Dict] = []

        total = 0
        hits_at_1 = 0
        hits_at_3_total = 0
        hallucination_scores: List[float] = []
        all_scores: List[float] = []

        for qi, row in df.iterrows():
            question = str(row.get('Question') or '').strip()
            if not question:
                continue
            total += 1

            results = await self.doc_manager.search_documents(
                collection_name=collection,
                query=question,
                limit=top_k,
                use_rerank=False
            )

            # Build per-rank entries
            per_ranks: List[Dict] = []
            local_hits = 0
            local_answer_included_hits = 0
            for rk, r in enumerate(results, start=1):
                doc = r.get('document') or {}
                inner = doc.get('document') if isinstance(doc, dict) else None
                content = ''
                metadata = {}
                if isinstance(inner, dict):
                    content = inner.get('content', '')
                    metadata = inner.get('metadata', {}) or {}
                elif isinstance(doc, dict):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {}) or {}
                score = r.get('score') if 'score' in r else (doc.get('score') if isinstance(doc, dict) else 0.0)
                try:
                    all_scores.append(float(score or 0.0))
                except Exception:
                    pass

                is_hit = int(metadata.get('row_index', -1)) == int(qi)
                if is_hit:
                    local_hits += 1

                # Answer-included detection (for non-binary with non-empty answer)
                answer_included = False
                q_type = str(row.get('Type') or '').strip()
                ans_cell = row.get('Answer')
                has_answer = (ans_cell is not None) and (not pd.isna(ans_cell)) and (str(ans_cell).strip() != '')
                if q_type.lower() != 'binary' and has_answer:
                    try:
                        answer_included = str(ans_cell).strip().lower() in (content or '').lower()
                    except Exception:
                        answer_included = False
                if answer_included:
                    local_answer_included_hits += 1

                per_ranks.append({
                    "rank": rk,
                    "score": float(score or 0.0),
                    "row_index": int(metadata.get('row_index', -1)),
                    "content": content,
                    "is_hit": bool(is_hit),
                    "answer_included": bool(answer_included)
                })

            top1_hit = False
            if per_ranks:
                top1_hit = bool(per_ranks[0].get("row_index", -999) == int(qi))
                if top1_hit:
                    hits_at_1 += 1
            hits_at_3_total += local_hits

            # Keep Gold_REF and Answer as string (including 'nan') for inspection
            gold_ref_str = str(row.get('Gold_REF'))
            answer_str = str(row.get('Answer'))

            # Additional per-question flags
            top1_answer_included = any(x.get('answer_included') for x in per_ranks[:1]) if per_ranks else False
            answer_is_nan = pd.isna(row.get('Answer'))
            retrieved_non_empty = any((x.get('content') or '').strip() != '' for x in per_ranks)
            is_hallucination = bool(answer_is_nan and retrieved_non_empty)
            if is_hallucination and per_ranks:
                try:
                    hallucination_scores.extend(float(x.get('score') or 0.0) for x in per_ranks)
                except Exception:
                    pass

            inference_items.append({
                "q_index": int(qi),
                "question": question,
                "top_k": int(top_k),
                "gold_ref": gold_ref_str,
                "answer": answer_str,
                "results": per_ranks,
                "top1_hit": bool(top1_hit),
                "top3_hit": bool(local_hits > 0),
                "top3_hits_count": int(local_hits),
                "top1_answer_included": bool(top1_answer_included),
                "top3_answer_included_count": int(local_answer_included_hits),
                "hallucination": bool(is_hallucination),
                "answer_is_nan_detection": "pandas.isna on original cell"
            })

        recall_at_1 = hits_at_1 / total if total else 0.0
        # Recall@3 should be the percentage of questions that have at least 1 hit in top-3
        hits_at_3_questions = sum(1 for x in inference_items if x.get('top3_hit'))
        recall_at_3 = hits_at_3_questions / total if total else 0.0

        # Derived metrics for new signals
        answer_included_top1 = sum(1 for x in inference_items if x.get('top1_answer_included'))
        answer_included_top3_total = sum(int(x.get('top3_answer_included_count') or 0) for x in inference_items)
        hallucination_questions = sum(1 for x in inference_items if x.get('hallucination'))
        if hallucination_scores:
            h_min = float(min(hallucination_scores))
            h_max = float(max(hallucination_scores))
            h_avg = float(sum(hallucination_scores) / len(hallucination_scores))
        else:
            h_min = 0.0
            h_max = 0.0
            h_avg = 0.0
        if all_scores:
            s_min = float(min(all_scores))
            s_max = float(max(all_scores))
            s_avg = float(sum(all_scores) / len(all_scores))
        else:
            s_min = 0.0
            s_max = 0.0
            s_avg = 0.0

        # Save JSONs
        inference_json = Path(out_dir) / "inference_results.json"
        metrics_json = Path(out_dir) / "eval_metrics.json"
        with inference_json.open('w', encoding='utf-8') as f:
            json.dump(inference_items, f, indent=2, ensure_ascii=False)
        with metrics_json.open('w', encoding='utf-8') as f:
            json.dump({
                "total_questions": total,
                "top_k": int(top_k),
                "recall_at_1": recall_at_1,
                "recall_at_3": recall_at_3,
                "answer_included_top1_rate": (answer_included_top1 / total) if total else 0.0,
                "answer_included_top3_avg": (answer_included_top3_total / (total * max(1, top_k))) if total else 0.0,
                "hallucination_rate": (hallucination_questions / total) if total else 0.0,
                "hallucination_count": int(hallucination_questions),
                "hallucination_min_score": h_min,
                "hallucination_max_score": h_max,
                "hallucination_avg_score": h_avg,
                "score_min": s_min,
                "score_max": s_max,
                "score_avg": s_avg
            }, f, indent=2, ensure_ascii=False)

        return {
            "inference_json": str(inference_json),
            "metrics_json": str(metrics_json),
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "hallucination_min_score": h_min,
            "hallucination_max_score": h_max,
            "hallucination_avg_score": h_avg,
            "score_min": s_min,
            "score_max": s_max,
            "score_avg": s_avg,
            "total_questions": total
        }


