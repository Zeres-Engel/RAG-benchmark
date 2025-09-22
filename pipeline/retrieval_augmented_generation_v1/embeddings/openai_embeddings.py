from typing import List

from langchain_openai import OpenAIEmbeddings

from retrieval_augmented_generation_v1.base_factory import BaseEmbeddingModel, EmbeddingModelFactory
from retrieval_augmented_generation_v1.document_loaders import Document


@EmbeddingModelFactory.register("openai")
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def _initialize_model(self):
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.config.api_key, openai_api_base=self.config.api_base)
        self.logger.info(f"Initialized OpenAI model: {self.config.model_name}")

    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for multiple documents using OpenAI API."""
        texts = [doc.content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embeddings.embed_query(text)
