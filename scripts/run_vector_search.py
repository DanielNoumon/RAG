import json
import os
from datetime import datetime


def main(config):
    """Simple document RAG test - 3 questions (configurable search method)."""
    all_results = []

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

    # Choose RAG pipeline based on config
    model = config.get("embedding_model")
    if config["search_method"] == "hnsw":
        from core.vector_search_pipeline_hnsw import VectorSearchPipelineHNSW
        rag = VectorSearchPipelineHNSW(storage_file=config["storage_file"])
        search_name = "HNSW (Approximate)"
    elif config["search_method"] == "exhaustive_knn":
        from core.vector_search_pipeline_knn import VectorSearchPipelineKNN
        rag = VectorSearchPipelineKNN(storage_file="data/embeddings/embeddings_knn.json")
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

        # Reranker
        "rerank": False,
        "reranker_type": "cross_encoder",  # Options: "llm", "cross_encoder", "colbert"
        "rerank_top_n": 5,     # Keep top 5 after reranking
        "rerank_model": None,  # Model name (None = use default for reranker type)
        "rerank_include_reasoning": False,  # Only for LLM reranker: include reasoning (slower)
        "rerank_batch_size": 5,  # Chunks per batch for parallel processing (LLM only)

        # Test Questions
        "questions": [
            "Hoeveel vakantiedagen krijg ik per jaar?",
            "Hoelang mag ik remote werken volgens het document?",
            "Welke kledingnormen moet ik hanteren?"
        ]
    }
    # ========================

    main(CONFIG)
