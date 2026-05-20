"""Build and save a SPLADE sparse index from a chunks JSON file.

Run this once per document collection.  The resulting index JSON can
then be loaded by SPLADERetriever at search time to skip re-encoding.

Usage (from project root):
    python -m preprocessing.build_splade_index
"""
from retrieval.splade import SPLADERetriever


def main(config):
    """Build and persist a SPLADE sparse index."""
    chunks_path = config["chunks_file"]
    index_path = config["index_output"]
    model_name = config.get(
        "model_name", SPLADERetriever.DEFAULT_MODEL
    )

    print("=" * 60)
    print("SPLADE Index Builder")
    print("=" * 60)
    print(f"Chunks : {chunks_path}")
    print(f"Model  : {model_name}")
    print(f"Output : {index_path}")
    print("-" * 60)

    # index_path is passed so the constructor auto-saves
    # after building.  If the file already exists it will
    # be loaded instead — delete it first to force rebuild.
    retriever = SPLADERetriever(
        chunks_path=chunks_path,
        model_name=model_name,
        device=config.get("device", None),
        max_length=config.get("max_length", 256),
        batch_size=config.get("batch_size", 32),
        index_path=index_path,
        doc_topn=config.get("doc_topn", 256),
        weight_threshold=config.get("weight_threshold", 0.01),
    )

    stats = retriever.get_stats()
    print(f"Chunks encoded  : {stats['total_chunks']}")
    print(f"Avg nonzero dims: {stats['avg_nonzero_dims']}")
    print(f"Vocab size      : {stats['vocab_size']}")
    print(f"Device          : {stats['device']}")
    print("\nDone.")


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    CONFIG = {
        # Input: chunks produced by chunker.py
        "chunks_file": "data/chunks/DSL_handboek_mei_2024_chunks.json",

        # Output: where to save the SPLADE sparse index
        "index_output": "data/index/splade_index.json",

        # Model (HuggingFace model id)
        # Options:
        #   "naver/splade-cocondenser-ensembledistil"  (recommended, ~200 MB)
        #   "naver/splade-v3"                          (newest)
        #   "naver/efficient-splade-VI-BT-large-doc"   (asymmetric doc encoder)
        "model_name": "naver/splade-cocondenser-ensembledistil",

        # Encoding settings
        "max_length": 256,   # Truncate chunks at this many tokens
        "batch_size": 32,    # Chunks per GPU/CPU batch (lower if OOM)
        "device": None,      # None = auto (cuda if available, else cpu)

        # Pruning
        "doc_topn": 256,           # Max non-zero dims per doc vector
        "weight_threshold": 0.01,  # Drop weights below this value
    }
    # ========================

    main(CONFIG)
