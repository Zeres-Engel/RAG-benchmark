from abc import ABC
from typing import Dict, Generic, Type, TypeVar

from retrieval_augmented_generation_v1.document_loaders import BaseDocumentLoader
from retrieval_augmented_generation_v1.embeddings import BaseEmbeddingModel
from retrieval_augmented_generation_v1.rerankings import BaseRerankingModel
from retrieval_augmented_generation_v1.vectorstores import BaseVectorStore

# Define a type variable for the base class that factories will create
T = TypeVar("T")


class BaseFactory(ABC, Generic[T]):
    """
    Generic abstract factory class for creating instances of registered classes.

    Type parameter T should be the base class/interface that all registered
    classes must implement.
    """

    @classmethod
    def register(cls, name: str):
        """
        Register a new implementation.

        Args:
            name: The name of the implementation to register.
        """

        def decorator(subclass: Type[T]) -> Type[T]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        """
        Create a new instance of the registered implementation.

        Args:
            name: The name of the registered implementation.
            *args: Positional arguments to pass to the constructor.
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            An instance of the registered implementation.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown implementation '{name}'. Available: {available}")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        """
        Get a list of all registered implementation names.

        Returns:
            List of registered implementation names.
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an implementation is registered.

        Args:
            name: The name to check.

        Returns:
            True if the implementation is registered, False otherwise.
        """
        return name in cls._registry


# Updated factory classes using the BaseFactory template
class EmbeddingModelFactory(BaseFactory[BaseEmbeddingModel]):
    """
    Factory class for creating embedding model instances.
    """

    _registry: Dict[str, Type[BaseEmbeddingModel]] = {}


class RerankingModelFactory(BaseFactory[BaseRerankingModel]):
    """
    Factory class for creating reranking model instances.
    """

    _registry: Dict[str, Type[BaseRerankingModel]] = {}


class VectorStoreFactory(BaseFactory[BaseVectorStore]):
    """
    Factory class for creating vector store instances.
    """

    _registry: Dict[str, Type[BaseVectorStore]] = {}


class DocumentLoaderFactory(BaseFactory[BaseDocumentLoader]):
    """
    Factory class for creating document loader instances.
    """

    _registry: Dict[str, Type[BaseDocumentLoader]] = {}
