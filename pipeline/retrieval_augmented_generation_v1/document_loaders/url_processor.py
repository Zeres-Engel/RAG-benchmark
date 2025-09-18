from langchain_community.document_loaders import WebBaseLoader

from retrieval_augmented_generation.base_factory import BaseDocumentLoader, DocumentLoaderFactory
from retrieval_augmented_generation.document_loaders import Document, DocumentType, InputDocument, InputSource


@DocumentLoaderFactory.register("url")
class URLProcessor(BaseDocumentLoader):
    """URL processor"""

    def __init__(self, config, logger=None):
        super().__init__(config, logger)

    def _accepts(self, document: InputDocument) -> bool:
        """Check if the document is a URL document"""
        return document.type == DocumentType.URL

    async def process(self, document: InputDocument) -> tuple[list[Document], str]:
        """Process a URL document"""
        if not self._accepts(document):
            raise ValueError(f"Unsupported document type: {document.type}")
        try:
            raw_docs = []
            loaded_text = ""
            if document.source == InputSource.URL:
                loader = WebBaseLoader(str(document.content))
                raw_docs.extend(loader.load())
                loaded_text = "\n\n".join([doc.page_content for doc in raw_docs])
            else:
                raise ValueError(f"Unsupported source type: '{document.source}' for URL processing")
            processed_docs = await self.split_docs(raw_docs)
            docs = []
            for i, processed_doc in enumerate(processed_docs):
                doc = Document(
                    id=processed_doc.metadata.get("id", None),
                    content=processed_doc.page_content,
                    title="Web Document",
                    source=document.metadata.get("source"),
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "chunk_id": i,
                        "total_chunks": len(processed_docs),
                        "document_type": DocumentType.URL.value,
                        "processor": "URLProcessor",
                    },
                )
                docs.append(doc)
            return docs, loaded_text
        except Exception as e:
            raise ValueError(f"Error processing URL document: {e}")
