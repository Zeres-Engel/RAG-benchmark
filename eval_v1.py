#!/usr/bin/env python3
"""
Evaluation runner for v1 pipeline with semantic chunking and search
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from evaluation.evaluator_v1 import EvaluatorV1


async def main():
    parser = argparse.ArgumentParser(description="Run evaluation with v1 pipeline (semantic chunking)")
    parser.add_argument("--dataset", default="data/dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k results to retrieve")
    parser.add_argument("--config", default="config/config_v1.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    print("🚀 RAG Evaluation Pipeline V1 (Semantic Chunking)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Top-K: {args.top_k}")
    print(f"Config: {args.config}")
    print("=" * 60)
    
    evaluator = EvaluatorV1(config_path=args.config)
    
    results = await evaluator.evaluate(
        dataset_path=args.dataset,
        top_k=args.top_k,
        out_dir="results"
    )
    
    print(f"\n✅ V1 Evaluation completed!")
    print(f"📊 Total Questions: {results['total_questions']}")
    print(f"📈 Recall@1: {results['recall_at_1']:.3f}")
    print(f"📈 Recall@3: {results['recall_at_3']:.3f}")
    print(f"📁 Results saved:")
    print(f"   - Inference: {results['inference_json']}")
    print(f"   - Metrics: {results['metrics_json']}")


if __name__ == "__main__":
    asyncio.run(main())
