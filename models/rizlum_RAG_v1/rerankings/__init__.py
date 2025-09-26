import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankingConfig:
    """Configuration for reranking models"""

    type: str
    name: str
    top_k: int = 4


class BaseRerankingModel(ABC):
    """Abstract base class for reranking models"""

    def __init__(self, config: RerankingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.top_k = config.get("top_k", 10)  # Default to 10 if not specified

    @abstractmethod
    async def compress_documents(self, query: str, documents: list[dict], top_k: int, score_threshold: float):
        """Rerank documents based on the query"""
        pass
