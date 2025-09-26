import re
import asyncio
from typing import Any, Dict, List, Optional

import yaml
import os

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

try:
    from models.rizlum_RAG_v1.embeddings.sentence_transformer_embeddings import (
        SentenceTransformerEmbeddingModel,
    )
except Exception:
    SentenceTransformerEmbeddingModel = None  # type: ignore
    try:
        # Fallback to sentence-transformers directly
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        SentenceTransformer = None  # type: ignore

from models.utils import trim_predictions_to_max_token_length


def _normalize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def _score(query: str, candidate: str) -> int:
    q_terms = set(_normalize(query))
    c_terms = set(_normalize(candidate))
    if not q_terms or not c_terms:
        return 0
    return len(q_terms & c_terms)


class RizlumRAGV1Model:
    def __init__(self):
        self.batch_size = 4
        self._config = self._load_config()
        self._enable_rerank = bool(self._config.get("enable_reranking", False))
        self._rerank_top_k: int = int(self._config.get("reranking", {}).get("top_k", 4))
        self._reranker = self._maybe_init_reranker(self._config) if self._enable_rerank else None
        # Qdrant + embeddings (optional)
        self._qdrant = self._maybe_init_qdrant(self._config)
        self._embedder = self._maybe_init_embedder(self._config)
        self._collection = os.getenv("RAG_COLLECTION", "voice-rules")
        # Prompt chunk char limit from config (fallback 500)
        try:
            vs = self._config.get("vector_store", {})
            self._prompt_chunk_char_limit = int(vs.get("chunk_size", 500))
        except Exception:
            self._prompt_chunk_char_limit = 500
        self._qdrant_ok = False
        if self._qdrant is not None:
            try:
                cols = self._qdrant.get_collections()
                names = [c.name for c in getattr(cols, "collections", [])]
                # If collection not found, mark as not ok
                if self._collection in names:
                    self._qdrant_ok = True
                else:
                    print(f"[RizlumRAG] Qdrant collection '{self._collection}' not found; available: {names}")
            except Exception as _e:
                print(f"[RizlumRAG] Qdrant health check failed: {_e}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open("config/config.yaml", "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _maybe_init_reranker(self, config: Dict[str, Any]):
        try:
            rerank_cfg = config.get("reranking", {})
            rtype = rerank_cfg.get("type", "qwen_cross_encoder")
            rname = rerank_cfg.get("name")
            # Lazy import to avoid heavy deps if not enabled
            if rtype in ["cross_encoder", "qwen_cross_encoder"]:
                from retrieval_augmented_generation.rerankings.cross_encoder_reranking import (
                    CrossEncoderRerankingModel,
                    QwenCrossEncoderRerankingModel,
                )
                cls = QwenCrossEncoderRerankingModel if rtype == "qwen_cross_encoder" else CrossEncoderRerankingModel
                # Minimal config shape
                _cfg = {"type": rtype, "name": rname, "top_k": self._rerank_top_k}
                return cls(_cfg, logger=None)  # logger optional here
            elif rtype == "flag_reranker":
                from retrieval_augmented_generation.rerankings.flag_reranking import FlagRerankingModel
                _cfg = {"type": rtype, "name": rerank_cfg.get("name"), "top_k": self._rerank_top_k}
                return FlagRerankingModel(_cfg, logger=None)
        except Exception:
            return None
        return None

    def _maybe_init_qdrant(self, config: Dict[str, Any]):
        try:
            if QdrantClient is None:
                return None
            vs = config.get("vector_store", {})
            if (vs.get("type") or "").lower() != "qdrant":
                return None
            # Allow QDRANT_URL override (e.g., http://localhost:6333)
            url = os.getenv("QDRANT_URL")
            if url:
                client = QdrantClient(url=url, api_key=vs.get("api_key") or os.getenv("QDRANT_API_KEY"))
            else:
                host = vs.get("host", "localhost")
                port = int(vs.get("port", 6333))
                api_key = vs.get("api_key") or os.getenv("QDRANT_API_KEY")
                use_https = bool(vs.get("use_https", False))
                client = QdrantClient(host=host, port=port, api_key=api_key, https=use_https)
            return client
        except Exception:
            return None

    def _maybe_init_embedder(self, config: Dict[str, Any]):
        try:
            # Prefer project wrapper; fallback to sentence-transformers
            embs = config.get("embedding_models", [])
            model_name = None
            device = "cpu"
            if embs:
                model_name = embs[0].get("name")
                device = embs[0].get("device", "cpu")
            if not model_name:
                return None
            if SentenceTransformerEmbeddingModel is not None:
                return SentenceTransformerEmbeddingModel(model_name=model_name, device=device)
            if 'SentenceTransformer' in globals() and SentenceTransformer is not None:
                st = SentenceTransformer(model_name)
                class _STWrapper:
                    def __init__(self, _m): self.m = _m
                    def embed_query(self, text: str):
                        vec = self.m.encode([text], normalize_embeddings=True)
                        return vec[0].tolist()
                return _STWrapper(st)
            return None
        except Exception:
            return None

    def _qdrant_search(self, query: str, limit: int, score_threshold: float) -> List[Dict[str, Any]]:
        """Search Qdrant and return normalized results with page_name/page_snippet/page_url."""
        out: List[Dict[str, Any]] = []
        if not (self._qdrant and self._embedder and self._qdrant_ok):
            return out
        try:
            vec = self._embedder.embed_query(query)  # type: ignore[attr-defined]
            hits = self._qdrant.search(
                collection_name=self._collection,
                query_vector=vec,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )
            for h in hits:
                payload = getattr(h, "payload", {}) or {}
                content = payload.get("content") or payload.get("text") or payload.get("chunk") or ""
                title = payload.get("title") or payload.get("page_name") or ""
                url = payload.get("url") or payload.get("page_url") or ""
                out.append({"page_name": title, "page_snippet": content, "page_url": url})
        except Exception as _e:
            print(f"[RizlumRAG] Qdrant search error: {_e}")
        return out

    async def _async_rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        try:
            return await self._reranker.compress_documents(query, docs, top_k=top_k, score_threshold=0.0)  # type: ignore[union-attr]
        except Exception:
            return docs

    def get_batch_size(self) -> int:
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Build one strong QA prompt per query from the top-2 content chunks (after optional rerank).
        The prompt asks the model to analyze the documents first, then answer the user question.
        """
        queries: List[str] = batch["query"]
        search_results: List[List[Dict[str, Any]]] = batch["search_results"]

        outputs: List[str] = []
        for i, query in enumerate(queries):
            # Prefer Qdrant retrieval if configured and available
            score_th = float(self._config.get("default_score_threshold", 0.0))
            if self._qdrant_ok and self._embedder is not None:
                qdrant_results = self._qdrant_search(query, limit=max(4, self._rerank_top_k or 4), score_threshold=score_th)
                results = qdrant_results if qdrant_results else (search_results[i] if i < len(search_results) else [])
            else:
                results = search_results[i] if i < len(search_results) else []

            # Collect candidate CONTENT chunks (prefer snippet/content/abstract; fallback to page_result slice)
            candidates: List[str] = []
            seen_contents = set()
            for r in results[:20]:
                title = r.get("page_name") or r.get("title") or r.get("name") or ""
                body = (
                    r.get("page_snippet")
                    or r.get("snippet")
                    or r.get("content")
                    or r.get("abstract")
                    or ""
                )
                if not body and isinstance(r.get("page_result"), str):
                    body = r.get("page_result")[:800]
                # Limit each chunk length from config (vector_store.chunk_size)
                chunk_text = "\n".join(filter(None, [title, body])).strip()
                if len(chunk_text) > self._prompt_chunk_char_limit:
                    chunk_text = chunk_text[: self._prompt_chunk_char_limit]
                if chunk_text and chunk_text not in seen_contents:
                    seen_contents.add(chunk_text)
                    candidates.append(chunk_text)

            if not candidates:
                outputs.append(
                    (
                        "Analyze the following documents. First, extract and organize the key facts that are relevant to the user question.\n"
                        "Then, based only on these documents, provide a concise final answer.\n\n"
                        "Documents (2 chunks):\n\n"
                        f"User question: {query}\n"
                        "Final answer:"
                    ).strip()
                )
                continue

            # Rerank (if configured) and take top-2 content chunks
            top_k = 2
            ranked_docs: List[Dict[str, Any]]
            if self._reranker is not None:
                docs = [{"document": {"content": c}} for c in candidates]
                try:
                    ranked_docs = asyncio.run(self._async_rerank(query, docs, top_k))
                except RuntimeError:
                    ranked_docs = docs[:top_k]
            else:
                scored = sorted(candidates, key=lambda c: _score(query, c), reverse=True)
                ranked_docs = [{"document": {"content": c}} for c in scored[:top_k]]

            # Compose the two chunks into a context
            chunks: List[str] = []
            for d in ranked_docs[:top_k]:
                c = (d.get("document", {}) or {}).get("content", "")
                if c:
                    ct = c.strip()
                    if len(ct) > self._prompt_chunk_char_limit:
                        ct = ct[: self._prompt_chunk_char_limit]
                    if ct not in chunks:
                        chunks.append(ct)
            context = "\n\n".join(chunks)
            if len(context) > 4000:
                context = context[:4000]

            prompt = (
                "Analyze the following documents. First, extract and organize the key facts that are relevant to the user question.\n"
                "Then, based only on these documents, provide a concise final answer.\n\n"
                f"Documents (2 chunks):\n{context}\n\n"
                f"User question: {query}\n"
                "Final answer:"
            )
            outputs.append(prompt)

        return outputs


