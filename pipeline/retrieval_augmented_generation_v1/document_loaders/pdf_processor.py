from langchain_community.document_loaders import PyPDFLoader

from retrieval_augmented_generation_v1.base_factory import BaseDocumentLoader, DocumentLoaderFactory
from retrieval_augmented_generation_v1.document_loaders import Document, DocumentType, InputDocument, InputSource


@DocumentLoaderFactory.register("pdf")
class PDFProcessor(BaseDocumentLoader):
    """PDF processor"""

    def __init__(self, config, logger=None):
        super().__init__(config, logger)

    def _accepts(self, document: Document) -> bool:
        """Check if the document is a PDF document"""
        return document.type == DocumentType.PDF

    async def process(self, document: InputDocument) -> tuple[list[Document], str]:
        """Process a PDF document"""
        if not self._accepts(document):
            raise ValueError(f"Unsupported document type: {document.type}")
        try:
            raw_docs = []
            loaded_text = ""
            if document.source == InputSource.FILE_PATH:
                loader = PyPDFLoader(str(document.content))
                raw_docs.extend(await loader.aload())
                loaded_text = "\n\n".join([doc.page_content for doc in raw_docs])
            elif document.source == InputSource.BINARY_DATA:
                content = document.content
                raw_docs.append(Document(page_content=content))
                loaded_text = content
            else:
                raise ValueError(f"Unsupported source type: '{document.source}' for PDF processing")

            docs = []
            for i, raw_doc in enumerate(raw_docs):
                page = Document(
                    id=raw_doc.metadata.get("id", None),
                    content=raw_doc.page_content,
                    title=f"PDF Page {i + 1}",
                    source=document.metadata.get("source"),
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "page_index": i,
                        "total_pages": len(raw_docs),
                        "document_type": DocumentType.PDF.value,
                        "processor": "PDFProcessor",
                    },
                )
                chunks = await self.split_docs([raw_doc])
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        id=chunk.metadata.get("id", None),
                        content=chunk.page_content,
                        title=f"{page.title} - Chunk {j + 1}",
                        source=page.source,
                        metadata={
                            **page.metadata,
                            "chunk_id": j,
                            "total_chunks": len(chunks),
                        },
                    )
                    docs.append(doc)
            return docs, loaded_text
        except Exception as e:
            raise ValueError(f"Error processing PDF document: {e}")
