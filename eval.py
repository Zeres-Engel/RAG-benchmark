#!/usr/bin/env python3
"""
Evaluation runner (single file)
- Metric: Hits@3 (count of relevant chunks in top-3 per question)
- Relevance: retrieved chunk belongs to the same source row (metadata.row_index matches question index)
"""

import asyncio
import csv
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

# Ensure project modules are importable
if 'pipeline' not in sys.path:
    sys.path.insert(0, 'pipeline')

from retrieval_augmented_generation.documents_manager import DocumentsManager
from retrieval_augmented_generation.document_loaders import InputDocument, InputSource, DocumentType

# Register factories
import retrieval_augmented_generation.vectorstores.qdrant_vectorstore  # noqa: F401
import retrieval_augmented_generation.embeddings.sentence_transformer_embeddings  # noqa: F401
import retrieval_augmented_generation.document_loaders.text_processor  # noqa: F401
import retrieval_augmented_generation.document_loaders.pdf_processor  # noqa: F401
import retrieval_augmented_generation.document_loaders.url_processor  # noqa: F401


from src.evaluation import Evaluator


async def evaluate(dataset_path: str = 'data/dataset.csv', top_k: int = 3, config_path: str | None = 'config/config.yaml') -> Dict:
    evaluator = Evaluator(config_path=config_path or 'config/config.yaml')
    return await evaluator.evaluate(dataset_path=dataset_path, out_dir='results', top_k=top_k)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/dataset.csv')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    asyncio.run(evaluate(args.dataset, args.top_k, args.config))


