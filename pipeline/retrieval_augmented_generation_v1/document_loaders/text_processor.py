from langchain.schema import Document as LangChainDocument
from langchain_community.document_loaders import TextLoader

from retrieval_augmented_generation_v1.base_factory import BaseDocumentLoader, DocumentLoaderFactory
from retrieval_augmented_generation_v1.document_loaders import Document, DocumentType, InputDocument, InputSource
from retrieval_augmented_generation_v1.document_loaders.semantic_splitter import SemanticTextSplitter


@DocumentLoaderFactory.register("text")
class TextProcessor(BaseDocumentLoader):
    """Text processor"""

    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        # Use semantic splitter instead of recursive character splitter
        self.semantic_splitter = SemanticTextSplitter(
            similarity_threshold=config.get("vector_store", {}).get("semantic_threshold", 0.7),
            min_chunk_size=config.get("vector_store", {}).get("min_chunk_size", 100),
            max_chunk_size=config.get("vector_store", {}).get("chunk_size", 1000)
        )

    def _accepts(self, document: InputDocument) -> bool:
        """Check if the document is a text document"""
        return document.type == DocumentType.TEXT

    async def process(self, document: InputDocument) -> tuple[list[Document], str]:
        """Clean and normalize the text document"""
        if not document:
            raise ValueError("Document content cannot be empty")
        if not self._accepts(document):
            raise ValueError(f"Unsupported document type: {document.type}")
        try:
            raw_docs = []
            loaded_text = ""
            if document.source == InputSource.FILE_PATH:
                loader = TextLoader(str(document.content), encoding="utf-8")
                raw_docs.extend(await loader.aload())
                loaded_text = "\n\n".join([doc.page_content for doc in raw_docs])
            elif document.source == InputSource.RAW_TEXT:
                content = document.content
                raw_docs.append(LangChainDocument(page_content=content))
                loaded_text = content
            else:
                raise ValueError(f"Unsupported source type: '{document.source}' for text processing")
            # Use semantic splitter instead of recursive character splitter
            processed_docs = self.semantic_splitter.split_documents(raw_docs)
            docs = []
            for i, processed_doc in enumerate(processed_docs):
                doc = Document(
                    id=processed_doc.metadata.get("id", None),
                    content=processed_doc.page_content,
                    title="Text Document",
                    source=document.metadata.get("source"),
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "chunk_id": i,
                        "total_chunks": len(processed_docs),
                        "document_type": DocumentType.TEXT.value,
                        "processor": "TextProcessor",
                    },
                )
                docs.append(doc)
            return docs, loaded_text
        except Exception as e:
            raise ValueError(f"Error processing text document: {e}")
