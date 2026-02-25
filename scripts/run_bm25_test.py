"""Test BM25 retrieval on chunked documents."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieval.bm25 import BM25Retriever  # noqa: E402
from core.azure_openai import AzureOpenAIClient  # noqa: E402


def main(config):
    """Run BM25 retrieval test with optional RAG generation."""
    # Initialize reranker if enabled
    reranker = None
    if config.get("rerank", False):
        from retrieval.reranker import Reranker
        reranker = Reranker(
            model_name=config.get(
                "rerank_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            top_n=config.get("rerank_top_n", 5),
        )

    chunks_path = config["chunks_file"]
    retriever = BM25Retriever(
        chunks_path,
        k1=config.get("k1", 1.5),
        b=config.get("b", 0.75),
    )

    stats = retriever.get_stats()
    print("=" * 60)
    print("Document RAG Test - BM25 (Keyword) Retrieval")
    print("=" * 60)
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Avg chunk length: {stats['avg_chunk_length']} tokens")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"BM25 k1={stats['k1']}, b={stats['b']}")
    print(f"Top K: {config['top_k']}")
    print("-" * 60)

    openai_client = AzureOpenAIClient() if config.get("generate_answer", True) else None

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
    output_path = f"data/test_results/bm25_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "search_method": "bm25",
        "top_k": config["top_k"],
        "k1": config.get("k1", 1.5),
        "b": config.get("b", 0.75),
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    CONFIG = {
        # Chunks file (produced by chunker.py)
        "chunks_file": os.path.join(PROJECT_ROOT, "data/chunks/document_handbook_mei_2024_chunks.json"),

        # BM25 Parameters
        "k1": 1.5,     # Term saturation (higher = more weight to term frequency)
        "b": 0.75,     # Length normalization (0 = no normalization, 1 = full)
        "top_k": 20,           # Retrieve 20 candidates

        # Reranker (LLM-based)
        "rerank": True,
        "rerank_top_n": 5,     # Keep top 5 after reranking
        "rerank_model": "gpt-5",  # Azure OpenAI deployment name
        "rerank_batch_size": 5,  # Chunks per batch for parallel processing

        # Output
        "show_chunks": False,
        "generate_answer": True,

        # Test Questions
        "questions": [
            "Hoeveel vakantiedagen krijg je bij het document en hoe moet je deze aanvragen?",
            "Wat zijn de regels voor remote werken bij het document?",
            "Wat zijn de huisregels en kledingnormen bij het document?"
        ],
    }
    # ========================

    main(CONFIG)
