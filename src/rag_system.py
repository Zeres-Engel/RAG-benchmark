"""
Simple RAG System for document retrieval and response generation
"""

import time
from typing import Dict, List, Optional
from .prompt_template import create_rag_prompt, format_chat_response


class SimpleRAGSystem:
    """
    Simple RAG system that retrieves documents and generates responses
    """
    
    def __init__(self, document_manager, logger):
        self.document_manager = document_manager
        self.logger = logger
    
    async def chat_with_rag(self, collection_name: str, query: str, limit: int = 5) -> Dict:
        """
        Chat with RAG: retrieve documents, create prompt, and generate response
        
        Args:
            collection_name: Name of the collection to search
            query: User question
            limit: Number of documents to retrieve
            
        Returns:
            Dict with query, rag_prompt, response, and metadata
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            self.logger.info(f"🔍 Retrieving documents for query: {query}")
            search_start = time.time()
            
            retrieved_docs = await self.document_manager.search_documents(
                collection_name=collection_name,
                query=query,
                limit=limit,
                use_rerank=True,
                rerank_top_k=3
            )
            
            search_time = time.time() - search_start
            
            if not retrieved_docs:
                return {
                    'query': query,
                    'rag_prompt': f"No relevant documents found for: {query}",
                    'response': "I don't have enough information in my knowledge base to answer your question.",
                    'metadata': {
                        'num_documents': 0,
                        'search_time': search_time,
                        'total_time': time.time() - start_time
                    }
                }
            
            # Step 2: Create RAG prompt with retrieved context
            self.logger.info(f"📝 Creating RAG prompt with {len(retrieved_docs)} documents")
            rag_prompt = create_rag_prompt(query, retrieved_docs)
            
            # Step 3: Generate response (simulated for now - you can add actual LLM here)
            self.logger.info("🤖 Generating response...")
            response = self._generate_response(rag_prompt, retrieved_docs)
            
            total_time = time.time() - start_time
            
            # Step 4: Return structured result
            result = {
                'query': query,
                'rag_prompt': rag_prompt,
                'response': response,
                'retrieved_documents': retrieved_docs,
                'metadata': {
                    'num_documents': len(retrieved_docs),
                    'search_time': search_time,
                    'total_time': total_time
                }
            }
            
            self.logger.info(f"✅ RAG chat completed in {total_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ RAG chat failed: {e}")
            return {
                'query': query,
                'rag_prompt': f"Error occurred while processing: {query}",
                'response': f"Sorry, I encountered an error: {str(e)}",
                'metadata': {
                    'num_documents': 0,
                    'search_time': 0,
                    'total_time': time.time() - start_time,
                    'error': str(e)
                }
            }
    
    def _generate_response(self, rag_prompt: str, retrieved_docs: List) -> str:
        """
        Generate response based on RAG prompt and retrieved documents
        
        For now this is a simple rule-based response.
        You can replace this with actual LLM API calls later.
        """
        
        # Simple response generation based on retrieved content
        if not retrieved_docs:
            return "I don't have relevant information to answer your question."
        
        # Extract key information from top documents
        top_doc = retrieved_docs[0]
        doc_content = top_doc.get('document', {}).get('content', '')
        score = top_doc.get('score', 0)
        
        # Create a basic response
        response = f"""Based on the retrieved documents (top match score: {score:.3f}), here's what I found:

{doc_content}

This information comes from my knowledge base and appears to be most relevant to your question. The retrieval system found {len(retrieved_docs)} related documents with varying relevance scores."""

        return response
    
    def format_for_display(self, result: Dict) -> str:
        """
        Format the RAG result for terminal display
        """
        return format_chat_response(
            query=result['query'],
            rag_prompt=result['rag_prompt'],
            generated_response=result['response'],
            metadata=result['metadata']
        )
