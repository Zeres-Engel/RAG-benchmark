import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from qdrant_client.http.models import VectorParams

from retrieval_augmented_generation_v1.document_loaders import Document


@dataclass
class VectorStoreConfig:
    type: str
    host: str
    port: int
    api_key: Optional[str] = None
    use_async: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 50


class Distance_Metric(Enum):
    """
    Enum for distance metrics used in vector stores.
    """

    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"
    MANHATTAN = "Manhattan"


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    """

    def __init__(self, config: VectorStoreConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        model_name: str,
        vector_config: VectorParams,
    ):
        """
        Create a new collection in the vector store.
        """
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata about a collection.
        Returns None if the collection does not exist.
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str):
        """
        Delete a collection from the vector store.
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: List[List[float]] = None,
    ) -> List[str]:
        """Add multiple documents to a collection"""
        pass

    @abstractmethod
    async def add_bulk_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: List[List[float]] = None,
    ) -> List[str]:
        """
        Add multiple documents to a collection in bulk.
        This method is optimized for adding large numbers of documents efficiently.
        """
        pass

    @abstractmethod
    async def delete_documents_by_sources(self, collection_name: str, sources: List[Dict[str, str]]):
        """Delete multiple documents from a collection by their sources."""
        pass

    @abstractmethod
    async def search_documents(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents in a collection using a query embedding.
        """
        pass
