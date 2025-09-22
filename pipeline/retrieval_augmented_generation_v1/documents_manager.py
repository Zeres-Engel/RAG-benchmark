import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from qdrant_client.http.models import VectorParams

from retrieval_augmented_generation_v1.base_factory import (
    DocumentLoaderFactory,
    EmbeddingModelFactory,
    RerankingModelFactory,
    VectorStoreFactory,
)
from retrieval_augmented_generation_v1.document_loaders import (
    BaseDocumentLoader,
    Document,
    DocumentType,
    InputDocument,
    InputSource,
)
from retrieval_augmented_generation_v1.embeddings import (
    BaseEmbeddingModel,
    EmbeddingConfig,
)
from retrieval_augmented_generation_v1.vectorstores import Distance_Metric
from retrieval_augmented_generation_v1.document_loaders.semantic_splitter import SemanticTextSplitter


class DocumentsManager:
    """
    DocumentsManager extends QdrantVectorStore to manage documents in a Qdrant vector store.
    It provides methods to add, delete, and retrieve documents, as well as to update their metadata.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.vector_store_manager = VectorStoreFactory.create(
            name=config.get("vector_store", {}).get("type", "qdrant"),
            config=config.get("vector_store", {}),
            logger=logger,
        )
        self.processors: Dict[DocumentType, BaseDocumentLoader] = {}
        self._initialize_processors()
        self.embedding_models: dict[str, BaseEmbeddingModel] = {}
        for model_config in config.get("embedding_models", []):
            try:
                embeddings_config = EmbeddingConfig(
                    model_type=model_config.get("type"),
                    model_name=model_config.get("name"),
                    dimension=model_config.get("dimension"),
                    api_key=model_config.get("api_key"),
                    api_base=model_config.get("api_base"),
                    device=model_config.get("device", "cpu"),
                    batch_size=model_config.get("batch_size", 32),
                    max_length=model_config.get("max_length", 512),
                )
                self.add_embedding_model(embeddings_config)
            except Exception as e:
                model_name = model_config.get("name")
                self.logger.error(f"Failed to add embedding model {model_name}: {e}")

        self.rerank_available = config.get("enable_reranking", False)
        if self.rerank_available:
            rerank_config = config.get("reranking", {})
            self.rerank_model = RerankingModelFactory.create(
                rerank_config.get("type"), config=rerank_config, logger=self.logger
            )
            self.logger.info(f"Initialized reranking model: {rerank_config.get('name')}")
        self.batch_size = config.get("batch_size", 32)
        # Initialize semantic splitter for query processing
        self.semantic_splitter = SemanticTextSplitter(
            similarity_threshold=config.get("vector_store", {}).get("semantic_threshold", 0.7),
            min_chunk_size=config.get("vector_store", {}).get("min_chunk_size", 100),
            max_chunk_size=config.get("vector_store", {}).get("chunk_size", 1000)
        )

    def _initialize_processors(self):
        """
        Initialize document processors for different document types.
        This method should be overridden in subclasses to add specific processors.
        """
        processor_types = [
            DocumentType.TEXT,
            DocumentType.PDF,
            DocumentType.URL,
            # DocumentType.DOCX,
            # DocumentType.XLSX,
            # DocumentType.PPTX,
            # DocumentType.CSV,
            # DocumentType.IMAGE,
            # DocumentType.JSON,
        ]

        for doc_type in processor_types:
            try:
                processor = DocumentLoaderFactory.create(doc_type.value, config=self.config, logger=self.logger)
                self.processors[doc_type] = processor
                self.logger.info(f"Initialized processor for {doc_type.value}")
            except Exception as e:
                raise ValueError(f"Failed to initialize processor for {doc_type.value}: {e}") from e

    def detect_document_type(self, document: str) -> tuple[DocumentType, InputSource]:
        """
        Detect the type of the document based on its content or metadata.
        This method should be overridden in subclasses to implement specific detection logic.
        """
        # Default implementation, can be overridden
        try:
            if re.match(r"^\s*https?://", document):
                return DocumentType.URL, InputSource.URL
            if len(document) > 300:
                # If the document is a long string, treat it as raw text
                return DocumentType.TEXT, InputSource.RAW_TEXT
            path = Path(document)
            if path.is_file():
                document_type = DocumentType.UNKNOWN
                ext = path.suffix.lower()
                if ext == ".pdf":
                    document_type = DocumentType.PDF
                elif ext == ".docx":
                    document_type = DocumentType.DOCX
                elif ext == ".xlsx":
                    document_type = DocumentType.XLSX
                elif ext == ".pptx":
                    document_type = DocumentType.PPTX
                elif ext == ".csv":
                    document_type = DocumentType.CSV
                elif ext == ".json":
                    document_type = DocumentType.JSON
                elif ext == ".html" or ext == ".htm":
                    document_type = DocumentType.HTML
                elif ext == ".txt" or ext == "":
                    document_type = DocumentType.TEXT
                return document_type, InputSource.FILE_PATH
            return DocumentType.TEXT, InputSource.RAW_TEXT
        except Exception as e:
            self.logger.error(f"Error detecting document type: {e}")
            return DocumentType.UNKNOWN, InputSource.UNKNOWN

    async def create_collection(
        self,
        collection_name: str,
        model_name: str,
        distance_metric: Distance_Metric = Distance_Metric.COSINE,
        recreate: bool = False,
    ):
        """Create a collection for storing embeddings"""
        if model_name not in self.embedding_models:
            raise ValueError(f"Embedding model {model_name} not found")

        model = self.embedding_models[model_name]
        vector_size = model.config.dimension

        try:
            if recreate:
                await self.vector_store_manager.delete_collection(collection_name)
            vector_config = VectorParams(size=vector_size, distance=distance_metric)
            await self.vector_store_manager.create_collection(
                collection_name=collection_name,
                model_name=model_name,
                vector_config=vector_config,
            )
            self.logger.info(f"Created collection: {collection_name} with model: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    async def delete_collection(self, collection_name: str):
        """Delete a collection from the vector store"""
        try:
            await self.vector_store_manager.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise e

    async def add_document(
        self, collection_name: str, document: Union[str, InputDocument], batch_size: Optional[int] = None
    ) -> str:
        """
        Add a new document to vector store by text or file path or link.
        """
        try:
            model, _ = await self.get_embedding_model(collection_name)
            if isinstance(document, str):
                doc_type, source_type = self.detect_document_type(document)
                # If document is a string, treat it as raw text
                document = InputDocument(content=document, type=doc_type, source=source_type)
            elif isinstance(document, InputDocument):
                doc_type = document.type
                source_type = document.source
                if not doc_type or not source_type:
                    doc_type, source_type = self.detect_document_type(document.content)
                    document.type = doc_type
                    document.source = source_type

            documents_processed, raw_text = await self.processors[document.type].process(document=document)
            # Process documents in batches
            if batch_size is None:
                batch_size = self.batch_size
            for i in range(0, len(documents_processed), batch_size):
                batch_docs = documents_processed[i : i + batch_size]

                # Generate embeddings
                embeddings = await model.embed_documents(documents=batch_docs)
                await self.vector_store_manager.add_documents(
                    collection_name=collection_name, documents=batch_docs, embeddings=embeddings
                )
            self.logger.info(f"Successfully added documents to {collection_name}")
            return raw_text

        except Exception as e:
            self.logger.error(f"Failed to add document to {collection_name}: {e}")
            raise e

    async def add_documents(
        self, collection_name: str, documents: List[Union[str, InputDocument]], **kwargs
    ) -> List[str]:
        """
        Add multiple documents to the vector store.
        """
        results = []
        for document in documents:
            results.extend(await self.add_document(collection_name=collection_name, document=document, **kwargs))
        return results

    async def delete_documents_by_sources(self, collection_name: str, sources: List[Dict[str, str]]):
        """
        Delete documents from the vector store by their sources.
        Parameters:
            collection_name (str): The name of the collection to delete documents from.
            sources (List[Dict[str, str]]): A list of sources to delete documents by.
        """
        await self.vector_store_manager.delete_documents_by_sources(collection_name=collection_name, sources=sources)

    async def add_bulk_document(
        self, collection_name: str, documents: List[Document], batch_size: Optional[int] = None
    ) -> list[str]:
        """
        Add a new document to vector store by text or file path or link.
        """
        try:
            model, _ = await self.get_embedding_model(collection_name)
            # Process documents in batches
            if batch_size is None:
                batch_size = self.batch_size
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]

                # Generate embeddings
                embeddings = await model.embed_documents(documents=batch_docs)
                await self.vector_store_manager.add_bulk_documents(
                    collection_name=collection_name, documents=batch_docs, embeddings=embeddings
                )
            self.logger.info(f"Successfully added documents to {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to add document to {collection_name}: {e}")
            raise e

    async def load_documents(self, documents: List[Union[str, InputDocument]]) -> tuple[list[Document], list[str]]:
        try:
            all_documents = []
            raw_all = []
            for document in documents:
                if isinstance(document, str):
                    doc_type, source_type = self.detect_document_type(document)
                    # If document is a string, treat it as raw text
                    document = InputDocument(content=document, type=doc_type, source=source_type)
                elif isinstance(document, InputDocument):
                    doc_type = document.type
                    source_type = document.source
                    if not doc_type or not source_type:
                        doc_type, source_type = self.detect_document_type(document.content)
                        document.type = doc_type
                        document.source = source_type

                documents_processed, raw_text = await self.processors[document.type].process(document=document)
                all_documents.extend(documents_processed)
                raw_all.append(raw_text)
            return all_documents, raw_all
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            raise e

    async def get_embedding_model(self, collection_name: str) -> tuple[BaseEmbeddingModel, str]:
        """Get the embedding model for a given collection
        Parameters:
            collection_name (str): The name of the collection to get the embedding model for.
        Returns:
            model (BaseEmbeddingModel): The embedding model instance.
            model_name (str): The name of the embedding model.
        """
        model_name = await self.vector_store_manager.get_embedding_model(collection_name=collection_name)
        model = self.embedding_models.get(model_name)
        return model, model_name

    def add_embedding_model(self, config: EmbeddingConfig):
        """Add an embedding model to the manager"""
        model_type = config.model_type
        model_name = config.model_name
        try:
            self.logger.info(f"Adding embedding model: {model_name} with config: {config}")
            model = EmbeddingModelFactory.create(model_type, config=config, logger=self.logger)
            self.embedding_models[model_name] = model
            self.logger.info(f"Added embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to add embedding model {model_name}: {e}")
            raise

    def get_supported_embedding_models(self) -> List[str]:
        """Get a list of supported embedding models"""
        return list(self.embedding_models.keys())

    async def search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict] = None,
        use_rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Search for similar documents"""
        try:
            if not score_threshold:
                score_threshold = self.config.get("default_score_threshold", 0.0)
            model, _ = await self.get_embedding_model(collection_name)
            # Generate query embedding
            tic = time.time()
            query_embedding = await model.embed_text(query)
            self.logger.debug(f"Generated query embedding in {time.time() - tic:.2f} seconds")

            results = await self.vector_store_manager.search_documents(
                collection_name=collection_name,
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )
            self.logger.info(
                f"Search in '{collection_name}' took {time.time() - tic:.2f} seconds, found {len(results)} results"
            )

            if not results:
                self.logger.info(f"No results found in collection '{collection_name}'")
                return []

            if self.rerank_available:
                if use_rerank:
                    self.logger.info("Reranking results using the rerank model")
                    # Use the rerank model to compress documents
                    tic = time.time()
                    results = await self.rerank_model.compress_documents(
                        query=query, documents=results, top_k=rerank_top_k, score_threshold=score_threshold
                    )
                    self.logger.info(f"Reranked results in {time.time() - tic:.2f} seconds")
                else:
                    self.logger.info("Skipping reranking, returning raw results")
            else:
                if use_rerank:
                    self.logger.warning("Falback to raw results, reranking not available")
                results = [{"document": doc} for doc in results]

            self.logger.info(f"Found {len(results)} results for query in {collection_name}")
            return results

        except Exception as e:
            self.logger.error(f"Failed to search in {collection_name}: {e}")
            raise

    async def get_collection_info(self, collection_name: str) -> Dict[str, Union[str, int]]:
        """Get information about a collection"""
        try:
            info = await self.vector_store_manager.get_collection_info(collection_name)
            if not info:
                self.logger.warning(f"Collection '{collection_name}' not found")
                return {}
            self.logger.info(f"Retrieved info for collection '{collection_name}': {info}")
            return info
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise e

    async def semantic_search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict] = None,
        use_rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Semantic search with multi-step retrieval:
        1. Break query into semantic parts
        2. For each semantic part, find 3 similar chunks  
        3. Aggregate and deduplicate results
        """
        try:
            if not score_threshold:
                score_threshold = self.config.get("default_score_threshold", 0.0)
            model, _ = await self.get_embedding_model(collection_name)
            
            # Step 1: Break query into semantic parts - get unique semantic parts
            query_parts = self.semantic_splitter.split_text(query)
            if not query_parts:
                query_parts = [query]
            
            # Remove duplicates while preserving order
            unique_parts = []
            seen_parts = set()
            for part in query_parts:
                part_clean = part.strip().lower()
                if part_clean not in seen_parts:
                    unique_parts.append(part)
                    seen_parts.add(part_clean)
            
            # If we only have 1 unique part, use it as-is (get TOP 3 from this part)
            # If we have multiple unique parts, use up to 3 of them
            final_parts = unique_parts[:3]
            
            self.logger.info(f"Using {len(final_parts)} unique semantic parts: {[part[:30] + '...' if len(part) > 30 else part for part in final_parts]}")
            
            # Step 2: Search for each unique semantic part - collect results
            all_results = []
            
            for i, query_part in enumerate(final_parts):
                self.logger.debug(f"Searching for semantic part {i+1}: {query_part[:50]}...")
                
                # Generate query embedding for this part
                tic = time.time()
                query_embedding = await model.embed_text(query_part)
                self.logger.debug(f"Generated embedding for part {i+1} in {time.time() - tic:.2f} seconds")
                
                # Search for this part (get 3 results per semantic part)
                part_results = await self.vector_store_manager.search_documents(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    limit=3,  # 3 chunks per semantic part
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions,
                )
                
                # Add ALL results from this part (allow duplicates across parts)
                for result in part_results:
                    result['semantic_part'] = i + 1
                    result['semantic_query'] = query_part
                    all_results.append(result)
            
            self.logger.info(f"Collected {len(all_results)} total chunks from {len(final_parts)} unique semantic parts")
            
            # Step 3: From 9 total chunks, select top-k best chunks based on score
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            selected_results = all_results[:limit]
            
            self.logger.info(f"Selected top-{limit} chunks from {len(all_results)} total chunks")
            
            if not selected_results:
                self.logger.info(f"No results found in collection '{collection_name}'")
                return []

            # Step 4: Apply reranking if enabled to the selected results
            if self.rerank_available and use_rerank:
                self.logger.info("Reranking selected semantic search results")
                tic = time.time()
                final_results = await self.rerank_model.compress_documents(
                    query=query, documents=selected_results, top_k=rerank_top_k, score_threshold=score_threshold
                )
                self.logger.info(f"Reranked semantic results in {time.time() - tic:.2f} seconds")
            else:
                if use_rerank and not self.rerank_available:
                    self.logger.warning("Fallback to raw results, reranking not available")
                final_results = [{"document": doc} for doc in selected_results]

            self.logger.info(f"Returning {len(final_results)} semantic search results for {collection_name}")
            return final_results

        except Exception as e:
            self.logger.error(f"Failed to perform semantic search in {collection_name}: {e}")
            raise
