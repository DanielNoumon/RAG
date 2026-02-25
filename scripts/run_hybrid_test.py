"""Test hybrid retrieval (vector + BM25) on chunked documents."""
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieval.hybrid import HybridRetriever  # noqa: E402
from core.azure_openai import AzureOpenAIClient  # noqa: E402


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main(config):
    """Run hybrid retrieval test with optional RAG generation."""
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

    retriever = HybridRetriever(
        chunks_path=config["chunks_file"],
        embeddings_path=config["embeddings_file"],
        vector_backend=config.get("vector_backend", "hnsw"),
        bm25_k1=config.get("k1", 1.5),
        bm25_b=config.get("b", 0.75),
    )

    stats = retriever.get_stats()
    fusion = config.get("fusion", "rrf")
    alpha = config.get("alpha", 0.5)

    print("=" * 60)
    print("Document RAG Test - Hybrid (Vector + BM25)")
    print("=" * 60)
    print(f"Fusion: {fusion}")
    if fusion == "weighted":
        print(f"Alpha: {alpha} (vector={alpha}, bm25={1 - alpha})")
    print(f"Vector backend: {config.get('vector_backend', 'hnsw')}")
    print(f"Vector docs: {stats['vector_docs']}")
    print(f"BM25 chunks: {stats['bm25']['total_chunks']}")
    print(f"BM25 vocab: {stats['bm25']['vocabulary_size']}")
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

        results = retriever.search(
            query=question,
            top_k=config["top_k"],
            fusion=fusion,
            alpha=alpha,
            rrf_k=config.get("rrf_k", 60),
            overlap_boost=config.get("overlap_boost"),
            vector_threshold=config.get("vector_threshold", 0.0),
        )

        # Store initial results for JSON output
        initial_results = results.copy()
        
        # Rerank if enabled
        final_results = results
        if reranker and results:
            final_results = reranker.rerank(
                query=question, 
                chunks=results,
                batch_size=config.get("rerank_batch_size", 5),
            )

        if config.get("show_chunks", False):
            score_key = (
                "fusion_score" if fusion == "rrf" else "hybrid_score"
            )
            for j, r in enumerate(final_results, 1):
                v_rank = r.get("vector_search_rank", "-")
                b_rank = r.get("bm25_search_rank", "-")
                
                # Build display string
                display_parts = [
                    f"[{j}] {score_key}={r[score_key]:.6f}",
                    f"vector_search_rank={v_rank}",
                    f"bm25_search_rank={b_rank}"
                ]
                
                # Add reranker scores if available
                if "rerank_score" in r:
                    display_parts.append(f"rerank_score={r['rerank_score']:.3f}")
                if "reasoning" in r:
                    display_parts.append(f"reasoning='{r['reasoning'][:50]}...'")
                
                print("  " + "  ".join(display_parts))
                print(f"      {r['content'][:200]}...")
        else:
            print(f"  [{len(results)} chunks retrieved]")

        # Generate answer if enabled
        answer = None
        if openai_client and final_results:
            context_parts: List[str] = []
            for chunk in final_results:
                chunk_text = chunk["content"]
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                context_parts.append(chunk_text)
            context = "\n\n---\n\n".join(context_parts)
            answer = openai_client.generate_rag_response(question, context)
            print(f"\n  Answer: {answer[:300]}...")

        # Create mapping from content to reranker data for all chunks
        reranker_data_map = {}
        reranked_contents = set()
        
        if config.get("rerank", False):
            for r in final_results:
                reranker_data_map[r["content"]] = {
                    "rerank_score": r.get("rerank_score"),
                    "reasoning": r.get("reasoning")
                }
                reranked_contents.add(r["content"])
            
            # Also map chunks that were reranked but not in top results
            for r in initial_results:
                if r["content"] not in reranker_data_map:
                    reranker_data_map[r["content"]] = {
                        "rerank_score": 0.0,
                        "reasoning": None
                    }
        
        score_key = (
            "fusion_score" if fusion == "rrf" else "hybrid_score"
        )
        
        # Build initial chunks with reranker data
        initial_chunks_data = []
        for r in initial_results:
            chunk_data = {
                "content": r["content"],
                "fusion_score": r[score_key],
                "vector_similarity": r.get("vector_similarity", 0),
                "bm25_score": r.get("bm25_score", 0),
                "vector_search_rank": r.get("vector_search_rank"),
                "bm25_search_rank": r.get("bm25_search_rank"),
            }
            
            # Add reranker data if enabled
            if config.get("rerank", False):
                rerank_data = reranker_data_map.get(r["content"], {})
                chunk_data["rerank_score"] = rerank_data.get("rerank_score")
                chunk_data["reasoning"] = rerank_data.get("reasoning")
                chunk_data["selected_by_reranker"] = r["content"] in reranked_contents
            
            initial_chunks_data.append(chunk_data)
        
        result_data = {
            "question": question,
            "answer": answer,
            "initial_chunks": initial_chunks_data,
        }
        
        # Only add reranked_chunks section if reranking is enabled
        if config.get("rerank", False):
            result_data["reranked_chunks"] = [
                {
                    "content": r["content"],
                    "fusion_score": r[score_key],
                    "vector_similarity": r.get("vector_similarity", 0),
                    "bm25_score": r.get("bm25_score", 0),
                    "vector_search_rank": r.get("vector_search_rank"),
                    "bm25_search_rank": r.get("bm25_search_rank"),
                    "rerank_score": r.get("rerank_score"),
                    "reasoning": r.get("reasoning"),
                }
                for r in final_results
            ]
        
        all_results.append(result_data)

    # Save results
    results_dir = PROJECT_ROOT / "data" / "test_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"hybrid_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "search_method": f"hybrid_{fusion}",
        "fusion": fusion,
        "alpha": alpha if fusion == "weighted" else None,
        "vector_backend": config.get("vector_backend", "hnsw"),
        "top_k": config["top_k"],
        "results": all_results,
    }

    print(f"\nAttempting to save results to: {output_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute path: {os.path.abspath(output_path)}")
    
    try:
        # Convert data first
        converted_output = convert_numpy_types(output)
        print("Data conversion successful")
        
        # Test JSON serialization
        json_test = json.dumps(converted_output, indent=2, ensure_ascii=False)
        print("JSON serialization successful, length:", len(json_test))
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_test)
        
        print(f"Results saved to {output_path}")
        print(f"File exists after save: {os.path.exists(output_path)}")
        
        # Verify file content
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"File size: {len(content)} characters")
                print("First 100 chars:", content[:100])
        
    except Exception as e:
        print(f"Error saving file: {e}")
        import traceback
        traceback.print_exc()
        # Try saving without conversion as fallback
        try:
            fallback_path = output_path.with_name(output_path.stem + "_fallback.json")
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=str)
            print("Fallback file saved")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    CONFIG = {
        # Data files
        "chunks_file": str(PROJECT_ROOT / "data" / "chunks" / "document_handbook_mei_2024_chunks.json"),
        "embeddings_file": str(PROJECT_ROOT / "data" / "embeddings" / "embeddings_hnsw.json"),
        "vector_backend": "hnsw",  # "hnsw" = fast approximate search, "knn" = exact search (slower)

        # Fusion strategy
        "fusion": "rrf",    # "rrf" or "weighted" -- both (0-1 normalized scores)
        "rrf_k": 60,        # RRF smoothing constant (default 60). Higher = flatter ranking, lower = top ranks dominate
        # "alpha": 0.5,       # used in weighted only mode: vector weight. alpha=0.7 → 70% vector, 30% BM25 (trust semantic meaning more)
        # "overlap_boost": 1.2,  # Optional: boost docs found by both methods (e.g. 1.2 = 20%)

        # Vector search threshold
        "vector_threshold": 0.0,  # Minimum vector similarity score (0.0 = no threshold)

        # BM25 parameters
        "k1": 1.5,          # Term saturation (higher = more weight to repeated terms)
        "b": 0.75,           # Length normalization (0 = no penalty, 1 = strong penalty for long chunks)

        # (initial) Retrieval
        "top_k": 20,           # Retrieve 20 candidates -> use lower amount without reranker

        # Final Reranker filter (LLM-based)
        "rerank": True,
        "rerank_top_n": 5,     # Keep top 5 after reranking
        "rerank_model": "gpt-5",  # Azure OpenAI deployment name
        "rerank_batch_size": 5,  # Chunks per batch for parallel processing

        # Output
        "show_chunks": True,
        "generate_answer": True,

        # Test Questions
        "questions": [
            "Hoeveel vakantiedagen staan er in het document en hoe moet je deze aanvragen?",
            "Wat zijn de regels voor remote werken volgens het document?",
            "Welke huisregels en kledingnormen beschrijft het document?"
        ],
    }
    # ========================

    main(CONFIG)
