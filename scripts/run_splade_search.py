"""Test SPLADE sparse retrieval on chunked documents.

Two modes:
  1. Fresh build  — encode chunks from scratch (slow, saves index).
  2. Load index   — load a pre-built index (fast).

Set "splade_index_file" in CONFIG to a path to enable mode 2.
Leave it as None to always encode fresh (mode 1).
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.splade import SPLADERetriever  # noqa: E402
from core.azure_openai import AzureOpenAIClient  # noqa: E402


def main(config):
    """Run SPLADE retrieval test with optional RAG generation."""
    # Initialize reranker if enabled
    reranker = None
    if config.get("rerank", False):
        reranker_type = config.get("reranker_type", "llm").lower()

        if reranker_type == "cross_encoder":
            from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(
                model_name=config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_n=config.get("rerank_top_n", 5),
            )
        elif reranker_type == "colbert":
            from retrieval.rerankers.colbert_reranker import ColBERTReranker
            reranker = ColBERTReranker(
                model_name=config.get("rerank_model", "colbert-ir/colbertv2.0"),
                top_n=config.get("rerank_top_n", 5),
            )
        elif reranker_type == "llm":
            from retrieval.rerankers.llm_reranker import Reranker
            reranker = Reranker(
                model_name=config.get("rerank_model", None),
                top_n=config.get("rerank_top_n", 5),
                include_reasoning=config.get("rerank_include_reasoning", False),
            )
        else:
            raise ValueError(
                f"Unknown reranker type: {reranker_type}. "
                f"Choose from: 'llm', 'cross_encoder', 'colbert'"
            )

    retriever = SPLADERetriever(
        chunks_path=config["chunks_file"],
        model_name=config.get(
            "model_name", SPLADERetriever.DEFAULT_MODEL
        ),
        device=config.get("device", None),
        max_length=config.get("max_length", 256),
        batch_size=config.get("batch_size", 32),
        index_path=config.get("splade_index_file"),
        doc_topn=config.get("doc_topn", 256),
        query_topn=config.get("query_topn", 128),
        weight_threshold=config.get("weight_threshold", 0.01),
    )

    stats = retriever.get_stats()
    print("=" * 60)
    print("Document RAG Test - SPLADE Sparse Retrieval")
    print("=" * 60)
    print(f"Model          : {stats['model_name']}")
    print(f"Device         : {stats['device']}")
    print(f"Chunks         : {stats['total_chunks']}")
    print(f"Avg nnz dims   : {stats['avg_nonzero_dims']}")
    print(f"Vocab size     : {stats['vocab_size']}")
    print(f"Top K          : {config['top_k']}")
    print("-" * 60)

    openai_client = (
        AzureOpenAIClient()
        if config.get("generate_answer", True)
        else None
    )

    all_results = []
    for i, question in enumerate(config["questions"], 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 60)

        results = retriever.search(question, top_k=config["top_k"])

        # Rerank if enabled
        if reranker and results:
            results = reranker.rerank(
                query=question,
                chunks=results,
                batch_size=config.get("rerank_batch_size", 5),
            )

        if config.get("show_chunks", False):
            for j, r in enumerate(results, 1):
                print(f"  [{j}] score={r['score']:.4f}")
                print(f"      {r['content'][:200]}...")
        else:
            print(f"  [{len(results)} chunks retrieved]")

        # Generate answer if enabled
        answer = None
        if openai_client and results:
            context = "\n\n---\n\n".join(r["content"] for r in results)
            answer = openai_client.generate_rag_response(question, context)
            print(f"\n  Answer: {answer[:300]}...")

        all_results.append({
            "question": question,
            "answer": answer,
            "chunks_used": len(results),
            "chunks": [
                {
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "content": r["content"][:200],
                }
                for r in results
            ],
        })

    # Save results
    os.makedirs("data/test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/test_results/splade_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "search_method": "splade",
        "model_name": config.get("model_name", SPLADERetriever.DEFAULT_MODEL),
        "top_k": config["top_k"],
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    CONFIG = {
        # Input: chunks produced by chunker.py
        "chunks_file": "data/chunks/..........json",

        # Path to SPLADE index. If the file exists it is loaded (fast).
        # If not, the index is built from scratch and saved here.
        # Set to None to always encode fresh without saving.
        "splade_index_file": "data/index/splade/splade_index.json",

        # Model
        "model_name": "naver/splade-cocondenser-ensembledistil",

        # Encoding settings
        "max_length": 256,
        "batch_size": 32,
        "device": None,     # None = auto (cuda if available, else cpu)

        # Pruning
        "doc_topn": 256,           # Max dims per document vector
        "query_topn": 128,         # Max dims per query vector
        "weight_threshold": 0.01,  # Drop weights below this

        # Retrieval
        "top_k": 20,

        # Reranker
        "rerank": False,
        "reranker_type": "cross_encoder",  # "llm", "cross_encoder", "colbert"
        "rerank_top_n": 5,
        "rerank_model": None,
        "rerank_include_reasoning": False,
        "rerank_batch_size": 5,

        # Output
        "show_chunks": False,
        "generate_answer": True,

        # Test Questions
        "questions": [
            "Hoeveel vakantiedagen krijg ik per jaar?",
            "Hoelang mag ik remote werken volgens het document?",
            "Welke kledingnormen moet ik hanteren?"
        ],
    }
    # ========================

    main(CONFIG)
