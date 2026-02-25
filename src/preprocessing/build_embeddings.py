"""Rebuild all embeddings with the current embedding model."""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.rag_pipeline_knn import RAGPipelineKNN
from core.hnsw_storage import HNSWStorageManager
from core.embedding_manager import EmbeddingManager
from core.config import Config
import numpy as np


def rebuild():
    config = Config()
    print(f"Using embedding model: {config.EMBEDDING_MODEL}")

    # Load chunks
    chunks_file = "data/chunks/document_handbook_mei_2024_chunks.json"
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # Initialize embedding model
    embedding_mgr = EmbeddingManager()
    print(f"Embedding dimension: {embedding_mgr.get_embedding_dimension()}")

    # Step 1: Rebuild KNN storage (embeddings_knn.json)
    print("\n--- Building KNN storage ---")
    rag = RAGPipelineKNN(storage_file="data/embeddings/embeddings_knn.json")
    for chunk in chunks:
        embedding = embedding_mgr.embed_text(chunk["content"])
        rag.storage_manager.insert_document(
            content=chunk["content"],
            embedding=embedding,
            metadata=chunk.get("metadata", {})
        )
    print(f"KNN storage: {rag.get_stats()['total_documents']} documents")

    # Step 2: Rebuild HNSW storage (embeddings_hnsw.json)
    print("\n--- Building HNSW storage ---")
    dim = embedding_mgr.get_embedding_dimension()
    hnsw = HNSWStorageManager("data/embeddings/embeddings_hnsw.json", dim=dim)
    for chunk in chunks:
        embedding = embedding_mgr.embed_text(chunk["content"])
        hnsw.add_document(
            content=chunk["content"],
            embedding=np.array(embedding),
            metadata=chunk.get("metadata", {})
        )
    print(f"HNSW storage: {hnsw.get_stats()['total_documents']} documents")

    print("\nDone! Both storages rebuilt with multilingual embeddings.")


if __name__ == "__main__":
    rebuild()
