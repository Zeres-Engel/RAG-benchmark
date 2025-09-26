import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from retrieval_augmented_generation.document_loaders import Document


class EmbeddingModelType(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""

    model_type: EmbeddingModelType
    model_name: str
    dimension: int
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, config: EmbeddingConfig, logger: logging.Logger):
        self.config = config
        self.model = None
        self.embeddings = None
        self.logger = logger
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        """Initialize the embedding model"""
        pass

    @abstractmethod
    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
