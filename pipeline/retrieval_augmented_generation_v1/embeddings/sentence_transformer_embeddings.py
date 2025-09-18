from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from retrieval_augmented_generation.base_factory import BaseEmbeddingModel, EmbeddingModelFactory
from retrieval_augmented_generation.document_loaders import Document


@EmbeddingModelFactory.register("huggingface")
class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, config, logger):
        self.device = config.device if config.device else "cpu"
        super().__init__(config, logger)

    def _initialize_model(self):
        model_kwargs = {"device": self.device}
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.model_name, model_kwargs=model_kwargs)
        self.embeddings.embed_query("This is a warm up for the model")
        self.logger.info(f"Initialized SentenceTransformer model: {self.config.model_name}")

    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for multiple documents using SentenceTransformer."""
        texts = [doc.content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embeddings.embed_query(text)
