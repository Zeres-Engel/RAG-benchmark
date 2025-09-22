import re
from typing import List

from langchain.schema import Document as LangchainDocument
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticTextSplitter:
    """
    Semantic text splitter that groups sentences based on semantic similarity
    instead of fixed character counts.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Basic sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text.strip())
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences."""
        embeddings = self.model.encode(sentences)
        return embeddings
    
    def _group_similar_sentences(self, sentences: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """Group sentences based on semantic similarity."""
        groups = []
        used_indices = set()
        
        for i, sentence in enumerate(sentences):
            if i in used_indices:
                continue
                
            current_group = [i]
            used_indices.add(i)
            current_text_length = len(sentence)
            
            # Find similar sentences within size limits
            for j in range(i + 1, len(sentences)):
                if j in used_indices:
                    continue
                    
                # Check if adding this sentence would exceed max size
                if current_text_length + len(sentences[j]) > self.max_chunk_size:
                    break
                
                # Calculate similarity
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                
                if similarity >= self.similarity_threshold:
                    current_group.append(j)
                    used_indices.add(j)
                    current_text_length += len(sentences[j])
            
            groups.append(current_group)
        
        return groups
    
    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Split documents using semantic chunking."""
        result_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < self.min_chunk_size:
                    continue
                    
                new_doc = LangchainDocument(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_method": "semantic",
                        "total_chunks": len(chunks)
                    }
                )
                result_docs.append(new_doc)
        
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """Split a single text into semantic chunks."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return [text] if text.strip() else []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings
        embeddings = self._get_sentence_embeddings(sentences)
        
        # Group similar sentences
        groups = self._group_similar_sentences(sentences, embeddings)
        
        # Create chunks from groups
        chunks = []
        for group in groups:
            chunk_sentences = [sentences[i] for i in group]
            chunk_text = ' '.join(chunk_sentences)
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
