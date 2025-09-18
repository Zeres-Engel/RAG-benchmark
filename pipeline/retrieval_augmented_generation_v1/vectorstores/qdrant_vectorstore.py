import os
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams

from retrieval_augmented_generation.base_factory import VectorStoreFactory
from retrieval_augmented_generation.document_loaders import Document
from retrieval_augmented_generation.vectorstores import BaseVectorStore


@VectorStoreFactory.register("qdrant")
class QdrantVectorStore(BaseVectorStore):
    """Main class for managing documents in Qdrant with multiple embedding models"""

    def __init__(
        self,
        logger,
        config,
    ):
        self.logger = logger
        self.qdrant_host = config.get("host", "localhost")
        self.qdrant_port = config.get("port", "6333")
        self.qdrant_api_key = config.get("api_key", None)
        self.use_async = config.get("use_async", True)
        self.on_disk = config.get("on_disk", True)

        # Initialize clients
        if self.use_async:
            self.client = AsyncQdrantClient(host=self.qdrant_host, port=self.qdrant_port, api_key=self.qdrant_api_key)
        else:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, api_key=self.qdrant_api_key)

        self.logger.info("Initialized QdrantDocumentManager")

    async def save_collection_metadata(self, collection_name: str, model_name: str, vector_size: int):
        """Save the metadata for a collection"""
        # Use dummy vector with correct size (content irrelevant)
        dummy_vector = np.zeros(vector_size).tolist()

        config_point = PointStruct(
            id=0, vector=dummy_vector, payload={"type": "config", "model_name": model_name, "vector_size": vector_size}
        )

        await self.client.upsert(collection_name=collection_name, points=[config_point])

    async def get_collection_metadata(self, collection_name: str):
        """Retrieve metadata for a collection"""
        try:
            result = await self.client.retrieve(collection_name=collection_name, ids=[0])
            if result:
                return result[0].payload
        except Exception as e:
            self.logger.error(f"Failed to retrieve metadata for collection {collection_name}: {e}")
            raise ValueError(f"Collection {collection_name} not found or has no metadata")
        return {}

    async def collections_exist(self, collection_name: str) -> bool:
        existing = (await self.client.get_collections()).collections
        if any(col.name == collection_name for col in existing):
            return True
        return False

    async def create_collection(
        self,
        collection_name: str,
        model_name: str,
        vector_config: VectorParams,
    ):
        """Create a collection for storing embeddings"""
        try:
            if not await self.collections_exist(collection_name):
                vector_config.on_disk = self.on_disk
                if self.use_async:
                    await self.client.create_collection(
                        collection_name=collection_name, vectors_config=vector_config, on_disk_payload=self.on_disk
                    )
                else:
                    self.client.create_collection(
                        collection_name=collection_name, vectors_config=vector_config, on_disk_payload=self.on_disk
                    )
                await self.save_collection_metadata(
                    collection_name=collection_name, model_name=model_name, vector_size=vector_config.size
                )
            else:
                self.logger.info(f"Collection {collection_name} already exists, skipping creation")

            self.logger.info(f"Created collection: {collection_name} with model: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    async def get_embedding_model(self, collection_name: str) -> str:
        """Get the embedding model name used for a collection"""
        if not await self.collections_exist(collection_name):
            raise ValueError(f"Collection {collection_name} not found")
        metadata = await self.get_collection_metadata(collection_name)
        model_name = metadata.get("model_name", "")
        return model_name

    async def delete_collection(self, collection_name: str):
        """Delete a collection if it exists"""
        try:
            if self.use_async:
                await self.client.delete_collection(collection_name)
            else:
                self.client.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist

    async def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: List[List[float]] = None,
    ) -> List[str]:
        """Add multiple documents to a collection"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        try:
            # Create points for Qdrant
            points = []
            for doc, embedding in zip(documents, embeddings):
                point = PointStruct(
                    id=doc.id,
                    vector=embedding,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "title": doc.title,
                        "source": doc.source,
                        "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
                    },
                )
                points.append(point)

                # Upload to Qdrant
                if self.use_async:
                    await self.client.upsert(collection_name=collection_name, points=points)
                else:
                    self.client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            self.logger.error(f"Failed to add documents to {collection_name}: {e}")
            raise

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
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        try:
            payloads = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "title": doc.title,
                    "source": doc.source,
                    "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
                }
                for doc in documents
            ]
            ids = [doc.id for doc in documents]

            if self.use_async:
                self.client.upload_collection(
                    collection_name=collection_name,
                    vectors=embeddings,
                    payload=payloads,
                    ids=ids,
                )
        except Exception as e:
            self.logger.error(f"Failed to prepare documents for bulk upload: {e}")
            raise

    async def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection"""
        try:
            if not await self.collections_exist(collection_name):
                return {}

            info = await self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise

    async def delete_documents_by_sources(self, collection_name: str, sources: List[Dict[str, str]]):
        """Delete documents by filtering on the 'source' field"""
        try:
            # Build filter for the source field
            filter_conditions = {"sources": sources}
            search_filter = self._get_search_filter(filter_conditions)
            ids_to_delete = []
            next_page_offset = None
            while True:
                # Find matching document IDs
                if self.use_async:
                    results, next_page_offset = await self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=search_filter,
                        with_payload=False,
                        limit=100,
                        offset=next_page_offset,
                    )
                    ids_to_delete.extend([point.id for point in results])
                else:
                    results, next_page_offset = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=search_filter,
                        with_payload=False,
                        limit=100,
                        offset=next_page_offset,
                    )
                    ids_to_delete.extend([point.id for point in results])
                if next_page_offset is None:
                    break

            self.logger.info(f"Found {len(ids_to_delete)} documents in {collection_name} with source(s): {sources}")
            if not len(ids_to_delete):
                self.logger.info(f"No documents found in {collection_name} with source(s): {sources}")
                return

            # Delete found documents
            if self.use_async:
                await self.client.delete(
                    collection_name=collection_name, points_selector=models.PointIdsList(points=ids_to_delete)
                )
            else:
                self.client.delete(
                    collection_name=collection_name, points_selector=models.PointIdsList(points=ids_to_delete)
                )

            self.logger.info(f"Deleted {len(ids_to_delete)} documents from {collection_name} with source(s): {sources}")
        except Exception as e:
            self.logger.error(f"Failed to delete documents from {collection_name} by source: {e}")
            raise

    def _get_search_filter(self, filter_conditions: Dict[str, Dict[str, str]]) -> Optional[models.Filter]:
        # Example filter_conditions:
        # {
        #   "sources": [
        #     {"source": "document.pdf", "location": "/Rizlum/Sales"},
        #     {"source": "https://example.com", "location": "/Rizlum/Sales"},
        #   ]
        # }
        # Example converted_filter_conditions:
        # {
        #   "should": [
        #     {"key": "source", "match": {"value": "/Rizlum/Sales/document.pdf"}},
        #     {"key": "source", "match": {"value": "/Rizlum/Sales/https://example.com"}},
        #   ]
        # }

        search_filter = None
        converted_filter_conditions = {}
        if filter_conditions:
            # TODO: For first version to avoid too complex filter conditions, only support "should" conditions
            # and "sources" key
            converted_filter_conditions = {"should": []}
            for key, values in filter_conditions.items():
                if not isinstance(values, list):
                    values = [values]
                if key == "sources":
                    for value in values:
                        combined_value = (
                            os.path.join(value["location"], value["source"])
                            if value.get("location")
                            else value["source"]
                        )
                        converted_filter_conditions["should"].append(
                            {"key": "source", "match": {"value": combined_value}}
                        )
                else:
                    for value in values:
                        converted_filter_conditions["should"].append({"key": key, "match": {"value": value}})

        if converted_filter_conditions:
            filter_kwargs = {}
            if converted_filter_conditions.get("must"):
                filter_kwargs["must"] = [
                    models.FieldCondition(key=cond["key"], match=models.MatchValue(value=cond["match"]["value"]))
                    for cond in converted_filter_conditions["must"]
                ]
            if converted_filter_conditions.get("should"):
                filter_kwargs["should"] = [
                    models.FieldCondition(key=cond["key"], match=models.MatchValue(value=cond["match"]["value"]))
                    for cond in converted_filter_conditions["should"]
                ]
            if converted_filter_conditions.get("must_not"):
                filter_kwargs["must_not"] = [
                    models.FieldCondition(key=cond["key"], match=models.MatchValue(value=cond["match"]["value"]))
                    for cond in converted_filter_conditions["must_not"]
                ]
            if filter_kwargs:
                search_filter = models.Filter(**filter_kwargs)

        return search_filter

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
        search_filter = self._get_search_filter(filter_conditions)
        try:
            if self.use_async:
                results = await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=search_filter,
                    with_payload=True,
                )
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=search_filter,
                    with_payload=True,
                )

            return [
                {
                    "id": res.id,
                    "score": res.score,
                    "document": {
                        "content": res.payload.get("content", ""),
                        "metadata": res.payload.get("metadata", {}),
                        "title": res.payload.get("title", ""),
                        "source": res.payload.get("source", ""),
                        "timestamp": res.payload.get("timestamp", None),
                        "model_name": res.payload.get("model_name", ""),
                    },
                }
                for res in results
            ]
        except Exception as _:
            raise
