"""Rebuild all embeddings with the current embedding model."""
import json

import numpy as np

from core.vector_search_pipeline_knn import VectorSearchPipelineKNN
from core.hnsw_storage import HNSWStorageManager
from core.embedding_manager import EmbeddingManager
from core.config import Config


def rebuild(
    chunks_file: str,
    knn_storage_path: str,
    hnsw_storage_path: str,
):

    config = Config()
    print(f"Using embedding model: {config.EMBEDDING_MODEL}")

    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # Initialize embedding model
    embedding_mgr = EmbeddingManager()
    print(f"Embedding dimension: {embedding_mgr.get_embedding_dimension()}")

    # Step 1: Rebuild KNN storage (embeddings_knn.json)
    print("\n--- Building KNN storage ---")
    rag = VectorSearchPipelineKNN(storage_file=knn_storage_path)
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
    hnsw = HNSWStorageManager(hnsw_storage_path, dim=dim)
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
    # ===== CONFIGURATION =====
    CONFIG = {
        "chunks_file": "data/chunks/..........json",
        "knn_storage": "data/embeddings/embeddings_knn.json",
        "hnsw_storage": "data/embeddings/embeddings_hnsw.json",
    }
    # =========================

    rebuild(
        CONFIG["chunks_file"],
        CONFIG["knn_storage"],
        CONFIG["hnsw_storage"],
    )
