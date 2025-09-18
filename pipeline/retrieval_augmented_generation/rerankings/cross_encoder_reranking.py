from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from retrieval_augmented_generation.base_factory import BaseRerankingModel, RerankingModelFactory


@RerankingModelFactory.register("cross_encoder")
class CrossEncoderRerankingModel(BaseRerankingModel):
    """A reranking model that uses a cross-encoder to rerank documents based on a query."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.model = HuggingFaceCrossEncoder(
            model_name=config.get("name", ""),
            model_kwargs={"trust_remote_code": True},
        )

    def get_pairs(self, query: str, documents: list[dict]) -> list[tuple[str, str]]:
        return [(query, doc.get("document", {}).get("content", "")) for doc in documents]

    async def compress_documents(
        self, query: str, documents: list[dict], top_k: int = None, score_threshold: float = 0.0
    ) -> list[dict]:
        """
        Rerank documents based on the query using the local rerank model.

        Args:
            query (str): The query to use for reranking.
            documents (list[Document]): The list of documents to rerank.
            top_k (int, optional): The maximum number of documents to return.
            score_threshold (float, optional): Minimum score threshold for returned documents.

        Returns:
            list[Document]: The reranked list of documents.
        """
        if not top_k:
            top_k = self.top_k
        pairs = self.get_pairs(query, documents)
        scores = self.model.score(pairs)
        scores = scores.tolist()
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            if score >= score_threshold:
                results.append(
                    {
                        **doc,
                        "rerank_score": float(score),
                    }
                )
        sorted_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        sorted_results = sorted_results[:top_k]
        return sorted_results


@RerankingModelFactory.register("qwen_cross_encoder")
class QwenCrossEncoderRerankingModel(CrossEncoderRerankingModel):
    """
    A reranking model that uses a cross-encoder to rerank documents based on a query,
    specifically for Qwen models.
    """

    @staticmethod
    def format_queries(query, instruction=None):
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"

    @staticmethod
    def format_document(document):
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"<Document>: {document}{suffix}"

    def get_pairs(self, query: str, documents: list[dict]) -> list[tuple[str, str]]:
        return [
            (self.format_queries(query), self.format_document(doc.get("document", {}).get("content", "")))
            for doc in documents
        ]
