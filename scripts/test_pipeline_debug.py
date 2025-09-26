import argparse
import json
import os
from typing import Any, List, Dict
import asyncio
import yaml

from tqdm.auto import tqdm

# Local imports
from models.user_config import UserModel
from local_evaluation import load_data_in_batches


def generate_predictions_min(dataset_path: str, model: Any, limit: int | None) -> tuple[List[str], List[str], List[str], List[List[Dict[str, Any]]]]:
    queries: List[str] = []
    gts: List[str] = []
    preds: List[str] = []
    search_results_all: List[List[Dict[str, Any]]] = []

    batch_size = model.get_batch_size()
    written = 0

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"):
        # Ground truths in example_data are strings
        batch_ground_truths = batch["answer"]
        batch_search_results = batch["search_results"]
        # Build a shallow copy without answers for model
        batch_for_model = {
            "interaction_id": batch["interaction_id"],
            "query": batch["query"],
            "search_results": batch_search_results,
            "query_time": batch["query_time"],
        }
        batch_predictions = model.batch_generate_answer(batch_for_model)

        queries.extend(batch["query"])
        gts.extend(batch_ground_truths)
        preds.extend(batch_predictions)
        search_results_all.extend(batch_search_results)

        if limit is not None:
            written += len(batch_predictions)
            if written >= limit:
                # Trim to exact limit if we exceeded
                queries = queries[:limit]
                gts = gts[:limit]
                preds = preds[:limit]
                break

    return queries, gts, preds, search_results_all


def build_context_from_search_results(sr_list: List[Dict[str, Any]], top_k: int = 2, max_chars: int = 1200) -> str:
    chunks: List[str] = []
    for sr in (sr_list or [])[:top_k]:
        name = sr.get("page_name") or sr.get("title") or sr.get("name") or ""
        snippet = sr.get("page_snippet") or sr.get("snippet") or sr.get("content") or ""
        url = sr.get("page_url") or sr.get("url") or sr.get("link") or ""
        part = "\n".join(filter(None, [str(name), str(snippet), str(url)]))
        if part:
            chunks.append(part)
    ctx = "\n\n".join(chunks)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars]
    return ctx


async def _async_rerank(query: str, docs: List[Dict[str, Any]], rtype: str, rname: str, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
    if rtype in ["cross_encoder", "qwen_cross_encoder"]:
        from retrieval_augmented_generation.rerankings.cross_encoder_reranking import (
            CrossEncoderRerankingModel,
            QwenCrossEncoderRerankingModel,
        )
        cls = QwenCrossEncoderRerankingModel if rtype == "qwen_cross_encoder" else CrossEncoderRerankingModel
        cfg = {"type": rtype, "name": rname, "top_k": top_k}
        model = cls(cfg, logger=None)
    elif rtype == "flag_reranker":
        from retrieval_augmented_generation.rerankings.flag_reranking import FlagRerankingModel
        cfg = {"type": rtype, "name": rname, "top_k": top_k}
        model = FlagRerankingModel(cfg, logger=None)
    else:
        return docs
    return await model.compress_documents(query, docs, top_k=top_k, score_threshold=score_threshold)


def rerank_and_build_context(query: str, sr_list: List[Dict[str, Any]], top_k: int = 4, score_threshold: float = 0.5, max_chars: int = 1600) -> tuple[str, List[Dict[str, Any]]]:
    # Load rerank config
    try:
        with open("config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
        rcfg = cfg.get("reranking", {})
        rtype = rcfg.get("type", "qwen_cross_encoder")
        rname = rcfg.get("name", "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
    except Exception:
        rtype = "qwen_cross_encoder"
        rname = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"

    # Build docs for reranker
    docs = []
    for sr in (sr_list or []):
        name = sr.get("page_name") or sr.get("title") or sr.get("name") or ""
        snippet = sr.get("page_snippet") or sr.get("snippet") or sr.get("content") or ""
        url = sr.get("page_url") or sr.get("url") or sr.get("link") or ""
        content = "\n".join(filter(None, [str(name), str(snippet), str(url)]))
        if content:
            docs.append({"document": {"content": content}, "_meta": {"page_name": name, "page_url": url}})

    reranked: List[Dict[str, Any]] = docs
    try:
        reranked = asyncio.run(_async_rerank(query, docs, rtype, rname, top_k=top_k, score_threshold=score_threshold))
    except RuntimeError:
        # In case of existing loop, skip rerank
        pass

    # Build context from reranked
    chunks: List[str] = []
    for item in reranked[:top_k]:
        content = item.get("document", {}).get("content", "")
        chunks.append(content)
    ctx = "\n\n".join(chunks)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars]
    return ctx, reranked[:top_k]


def gemini_generate_answer(query: str, context: str, model_name: str) -> str:
    # Prefer new google-genai SDK
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # try load from data/keys/gemini_keys.json
        try:
            with open("data/keys/gemini_keys.json", "r") as f:
                data = json.load(f)
            for entry in data.get("api_keys", []):
                if entry.get("active") and entry.get("key"):
                    api_key = entry["key"]
                    break
        except Exception:
            pass
    try:
        from google import genai as google_genai  # type: ignore
        client = google_genai.Client(api_key=api_key)  # type: ignore
        prompt = (
            "You are a precise QA assistant. Answer STRICTLY using the provided context.\n"
            "If context is insufficient, reply exactly: i don't know.\n"
            "Return a short answer matching the question type (yes/no, entity, number, etc.).\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        resp = client.models.generate_content(model=model_name, contents=prompt)  # type: ignore
        ans = getattr(resp, "text", None) or ""
        return ans.strip()
    except Exception:
        # fallback to old SDK
        import google.generativeai as generativeai  # type: ignore
        if api_key:
            generativeai.configure(api_key=api_key)  # type: ignore
        model = generativeai.GenerativeModel(model_name)  # type: ignore
        prompt = (
            "You are a precise QA assistant. Answer STRICTLY using the provided context.\n"
            "If context is insufficient, reply exactly: i don't know.\n"
            "Return a short answer matching the question type (yes/no, entity, number, etc.).\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        resp = model.generate_content(prompt)  # type: ignore
        ans = getattr(resp, "text", None) or ""
        return ans.strip()


def compute_offline_metrics(queries: List[str], gts: List[str], preds: List[str]) -> dict:
    # Normalize ground truths to list[list[str]] for the offline evaluator
    from evaluators.offline_eval import OfflineEvaluator

    evaluator = OfflineEvaluator(use_anls=True)
    # Wrap ground truths as list of list
    gts_wrapped = [[gt] for gt in gts]
    return evaluator.evaluate(queries, gts_wrapped, preds)


def compute_gemini_metrics(queries: List[str], gts: List[str], preds: List[str], model_name: str) -> dict:
    # Optional Gemini evaluation for debugging purposes only (not used in core pipeline)
    from evaluators.gemini_eval import GeminiEvaluator

    evaluator = GeminiEvaluator(model_name=model_name)
    gts_wrapped = [[gt] for gt in gts]
    return evaluator.evaluate(queries, gts_wrapped, preds)


def main():
    parser = argparse.ArgumentParser(description="End-to-end debug runner for inference + metrics")
    parser.add_argument("--dataset", default="example_data/dev_data.jsonl.bz2")
    parser.add_argument("--limit", type=int, default=20, help="Max samples to process (None for all)")
    parser.add_argument("--print_samples", type=int, default=5, help="How many samples to print for inspection")
    parser.add_argument("--use_gemini", action="store_true", help="Also compute Gemini metrics (debug only)")
    parser.add_argument("--gemini_model", default=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    parser.add_argument("--out", default="data/results/gemini_eval_results.jsonl", help="Where to write per-sample JSONL with Gemini judgement")
    parser.add_argument("--step", action="store_true", help="Print prompt to terminal and get Gemini answer one-by-one")
    parser.add_argument("--sleep_s", type=float, default=1.0, help="Sleep seconds between Gemini calls (rate limit friendly)")
    args = parser.parse_args()

    # Instantiate current UserModel (RizlumRAGV1Model recommended)
    model = UserModel()

    # Generate predictions
    queries, ground_truths, predictions, search_results_all = generate_predictions_min(args.dataset, model, args.limit)

    # Print a few samples
    to_print = min(args.print_samples, len(predictions))
    for i in range(to_print):
        print("=" * 120)
        print(f"IDX: {i}")
        print(f"QUERY: {queries[i]}")
        print(f"PREDICTION: {predictions[i]}")
        print(f"GROUND_TRUTH: {ground_truths[i]}")

    # Defer metrics until after we collect Gemini answers (to match your flow)

    # Optional Gemini metrics (debug only)
    if args.use_gemini:
    # 1) Rerank to build RAG context (top-2 chunks with content), then Gemini answer from that context
        gemini_answers: List[str] = []
        rag_debug_blocks: List[str] = []
        reranked_lists: List[List[Dict[str, Any]]] = []
        gemini_prompts: List[str] = []
        for idx, (q, srs) in enumerate(zip(queries, search_results_all)):
            ctx, reranked = rerank_and_build_context(q, srs, top_k=2, score_threshold=0.5, max_chars=2000)
            # Build a QA prompt USING the content context (top-2 chunks)
            qa_prompt = (
                "Analyze the following two document chunks and use them to answer the question.\n"
                "Provide a concise answer and a brief explanation grounded in the chunks.\n\n"
                f"Documents (2 chunks):\n{ctx}\n\n"
                f"User question: {q}\n"
                "Answer (with brief explanation):"
            )
            # Step-by-step: print prompt, then answer, then continue
            if args.step:
                print("\n=== RAG CHUNKS (sample {}/{}) ===".format(idx + 1, len(queries)))
                for c_idx, it in enumerate(reranked[:2]):
                    content = (it.get("document", {}) or {}).get("content", "")
                    print(f"Chunk {c_idx+1} score={it.get('rerank_score')}:\n{content}\n")
                print("=== GEMINI PROMPT ===")
                print(qa_prompt)
            ga = gemini_generate_answer(q, qa_prompt, args.gemini_model)
            if args.step:
                print("--- GEMINI ANSWER ---")
                print(ga)
                import time
                time.sleep(max(0.0, args.sleep_s))
            gemini_answers.append(ga)
            reranked_lists.append(reranked)
            gemini_prompts.append(qa_prompt)
            # prepare RAG system print block
            block_lines = ["--- RAG System Output (top-4) ---"]
            for idx, it in enumerate(reranked):
                block_lines.append(f"[{idx}] score={it.get('rerank_score')} name={it.get('_meta',{}).get('page_name')}")
            block_lines.append("--- Context (truncated) ---")
            block_lines.append(ctx[:600])
            rag_debug_blocks.append("\n".join(block_lines))

        print("\nRAG outputs (first few):")
        for i in range(min(args.print_samples, len(reranked_lists))):
            lst = reranked_lists[i]
            print("--- RAG Chunks (top-2) ---")
            for idx, it in enumerate(lst[:2]):
                content = (it.get("document", {}) or {}).get("content", "")
                print(f"Chunk {idx+1} score={it.get('rerank_score')}:\n{content[:600]}")
            print("--- Prompt (truncated) ---")
            print(gemini_prompts[i][:600])

        print("\nGemini-generated answers (first few):")
        for i in range(min(args.print_samples, len(gemini_answers))):
            print("-" * 30)
            print(f"Q: {queries[i]}")
            print(f"A_gemini: {gemini_answers[i]}")
            print(f"GT: {ground_truths[i]}")

        # 2) After answering all questions, compute metrics
        offline_results = compute_offline_metrics(queries, ground_truths, gemini_answers)
        print("\nOffline metrics (EM/ANLS-based) over Gemini answers:")
        print(json.dumps(offline_results, ensure_ascii=False, indent=2))

        gemini_results = compute_gemini_metrics(queries, ground_truths, gemini_answers, args.gemini_model)
        print("\nGemini metrics (judge Gemini-generated answers):")
        print(json.dumps(gemini_results, ensure_ascii=False, indent=2))

        # Per-sample JSONL with correctness
        from evaluators.gemini_eval import GeminiEvaluator
        evaluator = GeminiEvaluator(model_name=args.gemini_model)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        written = 0
        with open(args.out, "w") as f:
            for q, gt, pred, reranked, qa_prompt in zip(queries, ground_truths, gemini_answers, reranked_lists, gemini_prompts):
                # Note: we also want to include the exact prompt given to Gemini
                try:
                    score = evaluator._judge_once(q, gt, pred or "")  # 0/1
                except Exception:
                    score = 0
                # Extract exactly top-2 chunks and scores
                c1 = (reranked[0].get("document", {}) or {}).get("content", "") if len(reranked) > 0 else ""
                s1 = reranked[0].get("rerank_score") if len(reranked) > 0 else None
                c2 = (reranked[1].get("document", {}) or {}).get("content", "") if len(reranked) > 1 else ""
                s2 = reranked[1].get("rerank_score") if len(reranked) > 1 else None

                line = {
                    "query": q,
                    "ground_truth": gt,
                    "prediction": pred,
                    "correct": bool(score),
                    "score": int(score),
                    "chunk_1_text": c1,
                    "chunk_1_score": s1,
                    "chunk_2_text": c2,
                    "chunk_2_score": s2,
                    # include the prompt shown to Gemini for reproducibility
                    "gemini_prompt": qa_prompt,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
        print(f"\nWrote per-sample Gemini judgements: {args.out} ({written} lines)")


if __name__ == "__main__":
    main()


