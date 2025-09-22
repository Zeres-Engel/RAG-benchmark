"""
RAG Prompt Templates for generating responses with retrieved context
"""

def create_rag_prompt(query: str, retrieved_documents: list) -> str:
    """
    Create RAG prompt with retrieved documents as context
    
    Args:
        query: User question
        retrieved_documents: List of retrieved documents with content and metadata
    
    Returns:
        Formatted prompt for RAG generation
    """
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_documents, 1):
        doc_content = doc.get('document', {}).get('content', '')
        score = doc.get('score', 0)
        rerank_score = doc.get('rerank_score')
        
        context_parts.append(f"Document {i} (Score: {score:.3f}" + 
                           (f", Rerank: {rerank_score:.3f}" if rerank_score else "") + 
                           f"):\n{doc_content}")
    
    context = "\n\n".join(context_parts)
    
    # RAG prompt template
    rag_prompt = f"""Based on the following retrieved documents, please answer the user's question. Use the information from the documents to provide an accurate and helpful response.

==== RETRIEVED CONTEXT ====
{context}

==== USER QUESTION ====
{query}

==== RESPONSE ====
Based on the retrieved documents above, here is my response:

"""
    
    return rag_prompt


def format_chat_response(query: str, rag_prompt: str, generated_response: str, metadata: dict) -> str:
    """
    Format the complete chat response for display
    
    Args:
        query: Original user question
        rag_prompt: The full RAG prompt that was used
        generated_response: AI generated response
        metadata: Additional metadata (timing, scores, etc.)
    
    Returns:
        Formatted display string
    """
    
    output = f"""
🔍 USER QUERY:
{query}

📄 RAG PROMPT GENERATED:
{'-'*80}
{rag_prompt}
{'-'*80}

🤖 AI RESPONSE:
{generated_response}

📊 METADATA:
- Documents retrieved: {metadata.get('num_documents', 0)}
- Search time: {metadata.get('search_time', 0):.3f}s
- Total time: {metadata.get('total_time', 0):.3f}s
"""
    
    return output
