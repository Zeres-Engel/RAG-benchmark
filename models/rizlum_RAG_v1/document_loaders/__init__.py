import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Document:
    """Document data structure"""

    id: Optional[Union[int, uuid.UUID]] = None
    content: str = ""
    metadata: Dict[str, Any] = None
    title: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DocumentType(Enum):
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    URL = "url"
    UNKNOWN = "unknown"


class InputSource(Enum):
    FILE_PATH = "file_path"
    URL = "url"
    RAW_TEXT = "raw_text"
    BINARY_DATA = "binary_data"
    UNKNOWN = "unknown"


@dataclass
class InputDocument:
    """
    InputDocument represents a document to be processed.
    It contains the content of the document and its metadata.
    """

    id: Optional[Union[int, uuid.UUID]] = None
    content: Union[str, bytes, Path] = None
    source: InputSource = None
    type: DocumentType = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders"""

    @abstractmethod
    def __init__(self, config, logger: logging.Logger):
        """Initialize the document processor with configuration"""
        self.logger = logger
        chunk_size = config.get("vector_store", {}).get("chunk_size", 1000)
        chunk_overlap = config.get("vector_store", {}).get("chunk_overlap", 50)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @abstractmethod
    async def process(self, document: InputDocument) -> tuple[list[Document], str]:
        pass

    async def split_docs(self, raw_docs: list[LangchainDocument]) -> list[LangchainDocument]:
        return self.splitter.split_documents(raw_docs)
