import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(config):
    """Simple document RAG test - 3 questions (configurable search method)."""
    all_results = []

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

    # Choose RAG pipeline based on config
    model = config.get("embedding_model")
    if config["search_method"] == "hnsw":
        from core.rag_pipeline_hnsw import RAGPipelineHNSW
        rag = RAGPipelineHNSW(storage_file=config["storage_file"])
        search_name = "HNSW (Approximate)"
    elif config["search_method"] == "exhaustive_knn":
        from core.rag_pipeline_knn import RAGPipelineKNN
        rag = RAGPipelineKNN(storage_file="data/embeddings/embeddings_knn.json")
        search_name = "Exhaustive KNN (Exact)"
    else:
        raise ValueError(f"Unknown search_method: {config['search_method']}. Use 'hnsw' or 'exhaustive_knn'.")

    # Override embedding model only if different from current
    if model and model != rag.embedding_manager.model_name:
        rag.embedding_manager.model_name = model
        rag.embedding_manager._load_model()

    print("="*60)
    print(f"Document RAG Test - {search_name} Version - 3 Questions")
    print("="*60)

    # Show stats
    stats = rag.get_stats()
    print(f"Search Method: {config['search_method']}")
    print(f"Embedding Model: {config.get('embedding_model', 'default')}")
    print(f"Documents: {stats['total_documents']}")
    print(f"Storage: {stats['storage_file']}")

    # Show additional stats for HNSW
    if config["search_method"] == "hnsw":
        print(f"Dimension: {stats['dimension']}")

    if config.get("similarity_threshold"):
        print(f"Similarity threshold: {config['similarity_threshold']}")
    print(f"Top K chunks: {config['top_k']}")
    print("-"*60)

    # Test each question
    for i, question in enumerate(config["questions"], 1):
        print(f"\n{i}. Question: {question}")
        print("-"*60)

        try:
            query_kwargs = {"top_k": config["top_k"]}
            if config.get("similarity_threshold"):
                query_kwargs["similarity_threshold"] = config["similarity_threshold"]
            result = rag.query(question, **query_kwargs)

            # Rerank if enabled
            if reranker and result["context_documents"]:
                result["context_documents"] = reranker.rerank(
                    query=question,
                    chunks=result["context_documents"],
                    batch_size=config.get("rerank_batch_size", 5),
                )

            print(f"Answer: {result['answer']}")

            all_results.append({
                "question": question,
                "answer": result["answer"],
                "chunks_used": len(result["context_documents"]),
                "chunks": [
                    {
                        "similarity": round(float(d["similarity"]), 3),
                        "content": d["content"]
                    }
                    for d in result["context_documents"]
                ]
            })

            if config.get("show_chunks", False):
                if result['context_documents']:
                    print(f"\n--- Used {len(result['context_documents'])} relevant chunks ({search_name} Search) ---")
                    for j, doc in enumerate(result['context_documents'], 1):
                        print(f"\nChunk {j}:")
                        print(f"Similarity: {doc['similarity']:.3f}")
                        print(f"Content: {doc['content']}")
                        print("-" * 40)
                else:
                    print("\n--- No relevant chunks found ---")
            else:
                n = len(result['context_documents'])
                print(f"\n[{n} chunks retrieved]")

        except Exception as e:
            print(f"Error: {e}")
            all_results.append({"question": question, "error": str(e)})

        print()

    # Save results to JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "search_method": config["search_method"],
        "top_k": config["top_k"],
        "similarity_threshold": config.get("similarity_threshold"),
        "results": all_results
    }
    results_dir = "data/test_results"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(results_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    CONFIG = {
        # Search Method: "hnsw" or "exhaustive_knn" #hnsw for larger datasets -> efficiency. at smaller datasets, can play around with both (probably similar)
        "search_method": "hnsw",

        # RAG Parameters
        "storage_file": "data/embeddings/embeddings_hnsw.json", # embedded chunks
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "top_k": 20,          # Retrieve 20 candidates
        "show_chunks": False,
        # "similarity_threshold": 0.3,

        # Reranker (LLM-based)
        "rerank": True,
        "rerank_top_n": 5,     # Keep top 5 after reranking
        "rerank_model": "gpt-5",  # Azure OpenAI deployment name
        "rerank_batch_size": 5,  # Chunks per batch for parallel processing

        # Test Questions
        "questions": [
            "Hoeveel vakantiedagen staan er in het document en hoe moet je deze aanvragen?",
            "Wat zijn de regels voor remote werken volgens het document?",
            "Welke huisregels en kledingnormen beschrijft het document?"
        ]
    }
    # ========================

    main(CONFIG)
