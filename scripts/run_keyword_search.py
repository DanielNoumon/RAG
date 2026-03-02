"""Test BM25 retrieval on chunked documents."""
import json
import os
from datetime import datetime

from retrieval.bm25 import BM25Retriever
from core.azure_openai import AzureOpenAIClient


def main(config):
    """Run BM25 retrieval test with optional RAG generation."""
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
        "chunks_file": "data/chunks/..........json",

        # BM25 Parameters
        "k1": 1.5,     # Term saturation (higher = more weight to term frequency)
        "b": 0.75,     # Length normalization (0 = no normalization, 1 = full)
        "top_k": 20,

        # Reranker
        "rerank": True,
        "reranker_type": "cross_encoder",  # Options: "llm", "cross_encoder", "colbert"
        "rerank_top_n": 5,     # Keep top 5 after reranking
        "rerank_model": None,  # Model name (None = use default for reranker type)
        "rerank_include_reasoning": False,  # Only for LLM reranker: include reasoning (slower)
        "rerank_batch_size": 5,  # Chunks per batch for parallel processing (LLM only)

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
