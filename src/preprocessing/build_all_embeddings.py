"""Build KNN and HNSW embedding stores for multiple models.

Iterates through a list of embedding models, encodes all chunks
with each model using document-mode encoding, and saves separate
KNN/HNSW storage files per model.

Usage (from project root):
    uv run python -m preprocessing.build_all_embeddings
"""
import json
import gc
from pathlib import Path

import numpy as np

from core.model_embedding_manager import ModelEmbeddingManager
from core.json_storage import JSONStorageManager
from core.hnsw_storage import HNSWStorageManager


def build_for_model(
    model_name: str,
    chunks: list,
    output_dir: str,
    short_name: str,
):
    """Build KNN + HNSW stores for a single embedding model."""
    print("\n" + "=" * 60)
    print(f"  Building embeddings: {short_name} ({model_name})")
    print("=" * 60)

    emb_mgr = ModelEmbeddingManager(model_name)
    dim = emb_mgr.get_embedding_dimension()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    knn_path = str(out / f"embeddings_{short_name}_knn.json")
    hnsw_path = str(out / f"embeddings_{short_name}_hnsw.json")

    # -- KNN storage --
    print(f"\n  KNN → {knn_path}")
    knn = JSONStorageManager(knn_path)
    for i, chunk in enumerate(chunks):
        embedding = emb_mgr.embed_document(chunk["content"])
        knn.insert_document(
            content=chunk["content"],
            embedding=embedding,
            metadata=chunk.get("metadata", {}),
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"    KNN: {i + 1}/{len(chunks)}")

    # -- HNSW storage --
    print(f"\n  HNSW → {hnsw_path}")
    hnsw = HNSWStorageManager(hnsw_path, dim=dim)
    for i, chunk in enumerate(chunks):
        embedding = emb_mgr.embed_document(chunk["content"])
        hnsw.add_document(
            content=chunk["content"],
            embedding=np.array(embedding),
            metadata=chunk.get("metadata", {}),
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"    HNSW: {i + 1}/{len(chunks)}")

    print(f"\n  Done: {short_name} — {len(chunks)} chunks, dim={dim}")

    # Free GPU memory before loading next model
    del emb_mgr
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main(config):
    """Build embedding stores for all configured models."""
    chunks_file = config["chunks_file"]
    output_dir = config["output_dir"]

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    for model_cfg in config["models"]:
        build_for_model(
            model_name=model_cfg["model_name"],
            chunks=chunks,
            output_dir=output_dir,
            short_name=model_cfg["short_name"],
        )

    print("\n" + "=" * 60)
    print("  ALL MODELS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    CONFIG = {
        "chunks_file": (
            "data/chunks/DSL_handboek_mei_2024_chunks.json"
        ),
        "output_dir": "data/index/embeddings",
        "models": [
            {
                "model_name": (
                    "sentence-transformers/"
                    "paraphrase-multilingual-MiniLM-L12-v2"
                ),
                "short_name": "minilm",
            },
            {
                "model_name": "clips/e5-large-trm-nl",
                "short_name": "e5nl",
            },
            {
                "model_name": "Octen/Octen-Embedding-8B",
                "short_name": "octen8b",
            },
            {
                "model_name": "Qwen/Qwen3-Embedding-8B",
                "short_name": "qwen3_8b",
            },
            {
                "model_name": "nvidia/llama-embed-nemotron-8b",
                "short_name": "nemotron8b",
            },
        ],
    }

    main(CONFIG)
