from FlagEmbedding import FlagReranker

from retrieval_augmented_generation_v1.base_factory import BaseRerankingModel, RerankingModelFactory


@RerankingModelFactory.register("flag_reranker")
class FlagRerankingModel(BaseRerankingModel):
    """FlagRerankingModel is a reranking model that uses the FlagReranker from FlagEmbedding library."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.model = FlagReranker(model_name_or_path=config.get("name", ""), use_fp16=True)

    async def compress_documents(
        self, query: str, documents: list[dict], top_k: int = None, score_threshold: float = 0.0
    ) -> list[dict]:
        """
        Rerank documents based on the query using the local rerank model.

        Args:
            query (str): The query to use for reranking.
            documents (list[Document]): The list of documents to rerank.

        Returns:
            list[Document]: The reranked list of documents.
        """
        if not top_k:
            top_k = self.top_k
        pairs = [(query, doc.get("document", {}).get("content", "")) for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            if float(score) >= score_threshold:
                results.append(
                    {
                        **doc,
                        "rerank_score": float(score),
                    }
                )
        sorted_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        sorted_results = sorted_results[:top_k]
        return sorted_results
