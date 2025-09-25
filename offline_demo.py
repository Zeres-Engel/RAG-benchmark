#!/usr/bin/env python3
"""
LightRAG Offline Demo - fully offline with Ollama
Using lightweight models: gemma2:2b (LLM) + nomic-embed-text (embedding)
"""

import os
import sys
import asyncio

# Ensure local lightrag package is used
PROJECT_PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "pipeline")
if PROJECT_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PIPELINE_DIR)

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

# Setup logging
setup_logger("lightrag", level="INFO")

# Working directory - data will be stored here
WORKING_DIR = "./offline_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Logs directory under working dir
LOG_DIR = os.path.join(WORKING_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
# Reconfigure logger to write into our logs folder
setup_logger("lightrag", level="INFO", log_file_path=os.path.join(LOG_DIR, "lightrag.log"))

async def initialize_offline_rag():
    """Initialize LightRAG with a fully offline configuration"""
    print("🚀 Initializing LightRAG with lightweight offline models...")
    
    # Initialize shared storage for LightRAG internals (single-process)
    initialize_share_data(workers=1)
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Use gemma2:2b - lightweight LLM
        llm_model_func=ollama_model_complete,
        llm_model_name='gemma2:2b',
        # Reduce context and token sizes for small models
        llm_model_kwargs={"options": {"num_ctx": 8192}},
        chunk_token_size=800,
        chunk_overlap_token_size=50,
        summary_max_tokens=300,
        # Use nomic-embed-text for embeddings - compact and fast
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # nomic-embed-text dimension is 768
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text"
            )
        ),
        # Lower concurrency to reduce resource usage
        llm_model_max_async=2,
        embedding_func_max_async=4,
    )
    
    # IMPORTANT: Initialize storage and pipeline status
    print("📦 Initializing storage backends...")
    await rag.initialize_storages()
    
    print("⚙️ Initializing pipeline status...")
    await initialize_pipeline_status()
    
    print("✅ LightRAG initialized successfully!")
    return rag

async def demo_offline_lightrag():
    """Demo running LightRAG fully offline"""
    try:
        # Initialize RAG instance
        rag = await initialize_offline_rag()
        
        # Sample English documents - multiple and longer texts
        docs = [
            """
            LightRAG is a simple and fast Retrieval-Augmented Generation system.
            It uses a knowledge graph to efficiently store and retrieve information.
            LightRAG supports multiple backends such as OpenAI, Ollama, and Hugging Face.
            The system is particularly effective for natural language processing tasks.
            In production, LightRAG can be bound to different storage backends such as PostgreSQL, Neo4j, or FAISS.
            """,
            """
            Knowledge graphs store entities and relationships. In LightRAG, entity extraction and relation merging
            are performed asynchronously with configurable concurrency. A token budget divides the context window among
            entities, relations, and chunks to produce balanced prompts for the LLM. Reranking can further refine retrieved
            chunks, though lightweight environments may disable it for speed and simplicity.
            """,
            """
            Ollama serves local language and embedding models. For lightweight CPU or low-memory GPU environments,
            gemma2:2b is a reasonable choice for generation and nomic-embed-text for embeddings. Adjusting the context size,
            chunk size, and overlap reduces memory use while maintaining acceptable quality for small demos.
            """,
        ]
        
        print(f"\n📝 Inserting sample documents into the knowledge graph...")
        for i, d in enumerate(docs, 1):
            print(f"- Doc {i} length: {len(d.strip())} chars")
        await rag.ainsert(docs)
        print("✅ Documents inserted successfully!")
        
        # Test queries with different modes
        queries = [
            "What is LightRAG?",
            "Which backends does this system support?",
            "Why is LightRAG effective?",
            "How does LightRAG use a knowledge graph?",
            "What lightweight models are suitable for offline demos?",
        ]
        
        modes = ["naive", "local", "global", "hybrid"]
        
        # Collect results and write to JSON
        import json, time
        results = {
            "model": {
                "llm": "gemma2:2b",
                "embedding": "nomic-embed-text"
            },
            "queries": []
        }

        for query in queries:
            print(f"\n❓ Query: {query}")
            print("=" * 50)
            entry = {"query": query, "modes": {}}
            
            for mode in modes:
                print(f"\n🔍 Mode: {mode}")
                try:
                    result = await rag.aquery(
                        query,
                        param=QueryParam(mode=mode, enable_rerank=False)
                    )
                    trimmed = (result or "")[:200]
                    entry["modes"][mode] = result
                    print(f"📝 Answer: {trimmed}...")
                except Exception as e:
                    entry["modes"][mode] = {"error": str(e)}
                    print(f"❌ Error in {mode} mode: {e}")

            results["queries"].append(entry)

        out_path = os.path.join(WORKING_DIR, f"results_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📦 Results written to: {out_path}")
                
        print(f"\n🎉 Demo completed successfully!")
        print(f"💾 Data saved in: {WORKING_DIR}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'rag' in locals():
            try:
                await rag.finalize_storages()
                print("🧹 Storage finalized")
            except:
                pass

if __name__ == "__main__":
    print("🎯 LightRAG Offline Demo Starting...")
    print("📌 Using models: gemma2:2b (LLM) + nomic-embed-text (embedding)")
    print("🔒 Fully offline - no API key required!")
    print()
    
    asyncio.run(demo_offline_lightrag())
