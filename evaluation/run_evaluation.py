"""Compare retrieval methods on a ground-truth test set.

Runs BM25, SPLADE, Vector (HNSW), Vector (KNN), and Hybrid search
on every question in the test set, then computes Recall@k,
Precision@k, MRR, and MAP for each method and prints a comparison
table.

Usage (from project root):
    uv run python -m evaluation.run_evaluation
"""
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.bm25 import BM25Retriever
from retrieval.splade import SPLADERetriever
from core.embedding_manager import EmbeddingManager
from core.model_embedding_manager import ModelEmbeddingManager
from core.hnsw_storage import HNSWStorageManager
from core.json_storage import JSONStorageManager


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _chunk_id_from_result(result: Dict[str, Any]) -> int:
    """Extract a comparable chunk_id from a retrieval result dict.

    Different methods return the id under different keys
    ('chunk_id', 'id') and sometimes as int, sometimes str.
    """
    cid = result.get("chunk_id", result.get("id"))
    if cid is None:
        return -1
    return int(cid)


def _build_content_to_chunk_id(
    chunks_path: str,
) -> Dict[str, int]:
    """Build a mapping from chunk content to chunk_id.

    This resolves the mismatch between vector stores (1-based id)
    and the chunks file (0-based chunk_id).
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        ch["content"]: ch.get("chunk_id", idx)
        for idx, ch in enumerate(data["chunks"])
    }


def _run_vector_search(
    query: str,
    embedding_mgr,
    store,
    top_k: int,
    content_to_cid: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Run vector search and map results to chunk_ids.

    Supports both EmbeddingManager (embed_text) and
    ModelEmbeddingManager (embed_query).
    Deduplicates by chunk_id (keeps best score per chunk).
    """
    if hasattr(embedding_mgr, "embed_query"):
        query_emb = embedding_mgr.embed_query(query)
    else:
        query_emb = embedding_mgr.embed_text(query)
    raw = store.search_similar(
        query_emb, limit=top_k * 3, threshold=0.0
    )
    seen_cids: set = set()
    results = []
    for r in raw:
        content = r["content"]
        cid = content_to_cid.get(content, r.get("id", -1))
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        results.append({
            "chunk_id": cid,
            "content": content,
            "score": r["similarity"],
        })
        if len(results) >= top_k:
            break
    return results


def _rrf_fuse(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of two ranked result lists.

    Each result dict must have 'chunk_id' and 'content' keys.
    """
    scores: Dict[int, Dict[str, Any]] = {}

    for rank, doc in enumerate(results_a, start=1):
        cid = _chunk_id_from_result(doc)
        if cid not in scores:
            scores[cid] = {
                "chunk_id": cid,
                "content": doc["content"],
                "score": 0.0,
            }
        scores[cid]["score"] += 1.0 / (rrf_k + rank)

    for rank, doc in enumerate(results_b, start=1):
        cid = _chunk_id_from_result(doc)
        if cid not in scores:
            scores[cid] = {
                "chunk_id": cid,
                "content": doc["content"],
                "score": 0.0,
            }
        scores[cid]["score"] += 1.0 / (rrf_k + rank)

    fused = sorted(
        scores.values(), key=lambda x: x["score"], reverse=True,
    )
    return fused[:top_k]


# ------------------------------------------------------------------
# Per-method runners
# ------------------------------------------------------------------

def evaluate_method(
    method_name: str,
    search_fn,
    questions: List[Dict],
    k_values: List[int],
) -> Dict[str, Any]:
    """Run a single retrieval method on all questions and compute metrics.

    Parameters
    ----------
    method_name : str
        Human-readable label.
    search_fn : callable(query, top_k) -> list[dict]
        Must return dicts with at least a 'chunk_id' key.
    questions : list[dict]
        Test set questions with 'relevant_chunk_ids'.
    k_values : list[int]
        Values of k at which to compute Recall and Precision.

    Returns a dict with per-question details and aggregated metrics.
    """
    per_question = []
    all_mrr = []
    all_ap = []
    recall_sums = {k: 0.0 for k in k_values}
    prec_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}

    total_time = 0.0
    max_k = max(k_values)

    for q in questions:
        query = q["question"]
        relevant = set(q["relevant_chunk_ids"])

        t0 = time.perf_counter()
        results = search_fn(query, max_k)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        retrieved_ids = [_chunk_id_from_result(r) for r in results]

        mrr = reciprocal_rank(retrieved_ids, relevant)
        ap = average_precision(retrieved_ids, relevant)
        all_mrr.append(mrr)
        all_ap.append(ap)

        q_metrics = {
            "question_id": q["id"],
            "question": query,
            "type": q.get("type", ""),
            "relevant_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": retrieved_ids[:max_k],
            "mrr": round(mrr, 4),
            "ap": round(ap, 4),
            "latency_s": round(elapsed, 4),
        }

        for k in k_values:
            r = recall_at_k(retrieved_ids, relevant, k)
            p = precision_at_k(retrieved_ids, relevant, k)
            n = ndcg_at_k(retrieved_ids, relevant, k)
            recall_sums[k] += r
            prec_sums[k] += p
            ndcg_sums[k] += n
            q_metrics[f"recall@{k}"] = round(r, 4)
            q_metrics[f"precision@{k}"] = round(p, 4)
            q_metrics[f"ndcg@{k}"] = round(n, 4)

        per_question.append(q_metrics)

    n = len(questions)
    aggregated = {
        "method": method_name,
        "num_questions": n,
        "mean_mrr": round(sum(all_mrr) / n, 4) if n else 0,
        "map": round(sum(all_ap) / n, 4) if n else 0,
        "total_latency_s": round(total_time, 3),
        "mean_latency_s": round(total_time / n, 4) if n else 0,
    }
    for k in k_values:
        aggregated[f"mean_ndcg@{k}"] = round(ndcg_sums[k] / n, 4) if n else 0
        aggregated[f"mean_recall@{k}"] = round(recall_sums[k] / n, 4) if n else 0
        aggregated[f"mean_precision@{k}"] = round(prec_sums[k] / n, 4) if n else 0

    return {"aggregated": aggregated, "per_question": per_question}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(config: Dict[str, Any]):
    test_set_path = config["test_set"]
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    questions = test_set["questions"]

    k_values = config.get("k_values", [1, 3, 5, 10])
    results_all = {}

    # ---- BM25 ----
    if config.get("run_bm25", True):
        print("=" * 60)
        print("Evaluating: BM25")
        print("=" * 60)
        bm25 = BM25Retriever(config["chunks_file"])

        def bm25_search(query, top_k):
            return bm25.search(query, top_k)

        results_all["bm25"] = evaluate_method(
            "BM25", bm25_search, questions, k_values,
        )

    # ---- SPLADE ----
    if config.get("run_splade", True):
        print("=" * 60)
        print("Evaluating: SPLADE")
        print("=" * 60)
        splade = SPLADERetriever(
            chunks_path=config["chunks_file"],
            index_path=config.get("splade_index_file"),
        )

        def splade_search(query, top_k):
            return splade.search(query, top_k)

        results_all["splade"] = evaluate_method(
            "SPLADE", splade_search, questions, k_values,
        )

    # ---- SPLADE-NL (Dutch) ----
    if config.get("run_splade_nl", False):
        print("=" * 60)
        print("Evaluating: SPLADE-NL")
        print("=" * 60)
        splade_nl = SPLADERetriever(
            chunks_path=config["chunks_file"],
            model_name=config.get(
                "splade_nl_model",
                "sparse-encoder/splade-robbert-dutch-base-v1",
            ),
            index_path=config.get("splade_nl_index_file"),
        )

        def splade_nl_search(query, top_k):
            return splade_nl.search(query, top_k)

        results_all["splade_nl"] = evaluate_method(
            "SPLADE-NL", splade_nl_search, questions, k_values,
        )

    # ---- shared embedding manager for vector methods ----
    needs_vector = any(
        config.get(k, True) for k in [
            "run_vector_hnsw", "run_vector_knn",
            "run_vector_e5nl_hnsw", "run_vector_e5nl_knn",
            "run_hybrid_bm25_hnsw", "run_hybrid_bm25_knn",
            "run_hybrid_splade_hnsw", "run_hybrid_splade_knn",
            "run_hybrid_splade_nl_hnsw",
            "run_hybrid_splade_nl_knn",
        ]
    )
    emb_mgr = None
    content_to_cid = None
    if needs_vector:
        emb_mgr = EmbeddingManager()
        content_to_cid = _build_content_to_chunk_id(
            config["chunks_file"]
        )

    # ---- Vector (HNSW) ----
    if config.get("run_vector_hnsw", True):
        print("=" * 60)
        print("Evaluating: Vector (HNSW)")
        print("=" * 60)
        hnsw = HNSWStorageManager(config["embeddings_hnsw"])

        def hnsw_search(query, top_k):
            return _run_vector_search(
                query, emb_mgr, hnsw, top_k, content_to_cid,
            )

        results_all["vector_hnsw"] = evaluate_method(
            "Vector (HNSW)", hnsw_search, questions, k_values,
        )

    # ---- Vector (KNN) ----
    if config.get("run_vector_knn", True):
        print("=" * 60)
        print("Evaluating: Vector (KNN)")
        print("=" * 60)
        knn = JSONStorageManager(config["embeddings_knn"])

        def knn_search(query, top_k):
            return _run_vector_search(
                query, emb_mgr, knn, top_k, content_to_cid,
            )

        results_all["vector_knn"] = evaluate_method(
            "Vector (KNN)", knn_search, questions, k_values,
        )

    # ---- Hybrid: BM25 + HNSW ----
    if config.get("run_hybrid_bm25_hnsw", True):
        print("=" * 60)
        print("Evaluating: Hybrid (BM25+HNSW)")
        print("=" * 60)
        if "bm25" not in dir():
            bm25 = BM25Retriever(config["chunks_file"])
        if "hnsw" not in dir():
            hnsw = HNSWStorageManager(config["embeddings_hnsw"])

        def hybrid_bm25_hnsw(query, top_k):
            sparse = bm25.search(query, top_k)
            dense = _run_vector_search(
                query, emb_mgr, hnsw, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_bm25_hnsw"] = evaluate_method(
            "BM25+HNSW", hybrid_bm25_hnsw, questions, k_values,
        )

    # ---- Hybrid: BM25 + KNN ----
    if config.get("run_hybrid_bm25_knn", True):
        print("=" * 60)
        print("Evaluating: Hybrid (BM25+KNN)")
        print("=" * 60)
        if "bm25" not in dir():
            bm25 = BM25Retriever(config["chunks_file"])
        if "knn" not in dir():
            knn = JSONStorageManager(config["embeddings_knn"])

        def hybrid_bm25_knn(query, top_k):
            sparse = bm25.search(query, top_k)
            dense = _run_vector_search(
                query, emb_mgr, knn, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_bm25_knn"] = evaluate_method(
            "BM25+KNN", hybrid_bm25_knn, questions, k_values,
        )

    # ---- Hybrid: SPLADE + HNSW ----
    if config.get("run_hybrid_splade_hnsw", True):
        print("=" * 60)
        print("Evaluating: Hybrid (SPLADE+HNSW)")
        print("=" * 60)
        if "splade" not in dir():
            splade = SPLADERetriever(
                chunks_path=config["chunks_file"],
                index_path=config.get("splade_index_file"),
            )
        if "hnsw" not in dir():
            hnsw = HNSWStorageManager(config["embeddings_hnsw"])

        def hybrid_splade_hnsw(query, top_k):
            sparse = splade.search(query, top_k)
            dense = _run_vector_search(
                query, emb_mgr, hnsw, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_splade_hnsw"] = evaluate_method(
            "SPLADE+HNSW", hybrid_splade_hnsw, questions, k_values,
        )

    # ---- Hybrid: SPLADE + KNN ----
    if config.get("run_hybrid_splade_knn", True):
        print("=" * 60)
        print("Evaluating: Hybrid (SPLADE+KNN)")
        print("=" * 60)
        if "splade" not in dir():
            splade = SPLADERetriever(
                chunks_path=config["chunks_file"],
                index_path=config.get("splade_index_file"),
            )
        if "knn" not in dir():
            knn = JSONStorageManager(config["embeddings_knn"])

        def hybrid_splade_knn(query, top_k):
            sparse = splade.search(query, top_k)
            dense = _run_vector_search(
                query, emb_mgr, knn, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_splade_knn"] = evaluate_method(
            "SPLADE+KNN", hybrid_splade_knn, questions, k_values,
        )

    # ---- Hybrid: SPLADE-NL + HNSW ----
    if config.get("run_hybrid_splade_nl_hnsw", False):
        print("=" * 60)
        print("Evaluating: Hybrid (SPLADE-NL+HNSW)")
        print("=" * 60)
        if "splade_nl" not in dir():
            splade_nl = SPLADERetriever(
                chunks_path=config["chunks_file"],
                model_name=config.get(
                    "splade_nl_model",
                    "sparse-encoder/splade-robbert-dutch-base-v1",
                ),
                index_path=config.get("splade_nl_index_file"),
            )
        if "hnsw" not in dir():
            hnsw = HNSWStorageManager(config["embeddings_hnsw"])

        def hybrid_splade_nl_hnsw(query, top_k):
            sparse = splade_nl.search(query, top_k)
            dense = _run_vector_search(
                query, emb_mgr, hnsw, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_splade_nl_hnsw"] = evaluate_method(
            "SPLADE-NL+HNSW",
            hybrid_splade_nl_hnsw, questions, k_values,
        )

    # ---- Hybrid: SPLADE-NL + KNN ----
    if config.get("run_hybrid_splade_nl_knn", False):
        print("=" * 60)
        print("Evaluating: Hybrid (SPLADE-NL+KNN)")
        print("=" * 60)
        if "splade_nl" not in dir():
            splade_nl = SPLADERetriever(
                chunks_path=config["chunks_file"],
                model_name=config.get(
                    "splade_nl_model",
                    "sparse-encoder/splade-robbert-dutch-base-v1",
                ),
                index_path=config.get("splade_nl_index_file"),
            )
        if "knn" not in dir():
            knn = JSONStorageManager(config["embeddings_knn"])

        def hybrid_splade_nl_knn(query, top_k):
            sparse = splade_nl.search(query, top_k)
            dense = _run_vector_search(
                query, emb_mgr, knn, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_splade_nl_knn"] = evaluate_method(
            "SPLADE-NL+KNN",
            hybrid_splade_nl_knn, questions, k_values,
        )

    # ---- E5-NL Vector (HNSW) ----
    if config.get("run_vector_e5nl_hnsw", False):
        print("=" * 60)
        print("Evaluating: Vector E5-NL (HNSW)")
        print("=" * 60)
        e5nl_emb = ModelEmbeddingManager(
            config["e5nl_model"]
        )
        e5nl_hnsw = HNSWStorageManager(
            config["embeddings_e5nl_hnsw"],
            dim=e5nl_emb.get_embedding_dimension(),
        )

        def e5nl_hnsw_search(query, top_k):
            return _run_vector_search(
                query, e5nl_emb, e5nl_hnsw,
                top_k, content_to_cid,
            )

        results_all["vector_e5nl_hnsw"] = evaluate_method(
            "E5-NL (HNSW)",
            e5nl_hnsw_search, questions, k_values,
        )

    # ---- E5-NL Vector (KNN) ----
    if config.get("run_vector_e5nl_knn", False):
        print("=" * 60)
        print("Evaluating: Vector E5-NL (KNN)")
        print("=" * 60)
        if "e5nl_emb" not in dir():
            e5nl_emb = ModelEmbeddingManager(
                config["e5nl_model"]
            )
        e5nl_knn = JSONStorageManager(
            config["embeddings_e5nl_knn"]
        )

        def e5nl_knn_search(query, top_k):
            return _run_vector_search(
                query, e5nl_emb, e5nl_knn,
                top_k, content_to_cid,
            )

        results_all["vector_e5nl_knn"] = evaluate_method(
            "E5-NL (KNN)",
            e5nl_knn_search, questions, k_values,
        )

    # ---- Hybrid: SPLADE-NL + E5-NL ----
    if config.get("run_hybrid_splade_nl_e5nl", False):
        print("=" * 60)
        print("Evaluating: Hybrid (SPLADE-NL+E5-NL)")
        print("=" * 60)
        if "splade_nl" not in dir():
            splade_nl = SPLADERetriever(
                chunks_path=config["chunks_file"],
                model_name=config.get(
                    "splade_nl_model",
                    "sparse-encoder/splade-robbert-dutch-base-v1",
                ),
                index_path=config.get("splade_nl_index_file"),
            )
        if "e5nl_emb" not in dir():
            e5nl_emb = ModelEmbeddingManager(
                config["e5nl_model"]
            )
        if "e5nl_hnsw" not in dir():
            e5nl_hnsw = HNSWStorageManager(
                config["embeddings_e5nl_hnsw"],
                dim=e5nl_emb.get_embedding_dimension(),
            )

        def hybrid_splade_nl_e5nl(query, top_k):
            sparse = splade_nl.search(query, top_k)
            dense = _run_vector_search(
                query, e5nl_emb, e5nl_hnsw, top_k,
                content_to_cid,
            )
            return _rrf_fuse(sparse, dense, top_k)

        results_all["hybrid_splade_nl_e5nl"] = evaluate_method(
            "SPLADE-NL+E5-NL",
            hybrid_splade_nl_e5nl, questions, k_values,
        )

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("  RETRIEVAL METHOD COMPARISON")
    print("=" * 80)

    methods = list(results_all.keys())
    header_parts = [f"{'Metric':<25}"]
    for m in methods:
        label = results_all[m]["aggregated"]["method"]
        header_parts.append(f"{label:>18}")
    print("".join(header_parts))
    print("-" * 80)

    # Rows to display
    metric_keys = ["mean_mrr", "map"]
    for k in k_values:
        metric_keys.append(f"mean_ndcg@{k}")
    for k in k_values:
        metric_keys.append(f"mean_recall@{k}")
    for k in k_values:
        metric_keys.append(f"mean_precision@{k}")
    metric_keys.append("mean_latency_s")

    for mk in metric_keys:
        row = [f"{mk:<25}"]
        for m in methods:
            val = results_all[m]["aggregated"].get(mk, "")
            if isinstance(val, float):
                row.append(f"{val:>18.4f}")
            else:
                row.append(f"{str(val):>18}")
        print("".join(row))

    print("=" * 80)

    # ------------------------------------------------------------------
    # Per-question type breakdown
    # ------------------------------------------------------------------
    types = sorted(set(q.get("type", "") for q in questions))
    if len(types) > 1:
        print("\n")
        print("=" * 80)
        print("  BREAKDOWN BY QUESTION TYPE")
        print("=" * 80)
        for qtype in types:
            type_qs = [q for q in questions if q.get("type") == qtype]
            if not type_qs:
                continue
            print(f"\n  Type: {qtype} ({len(type_qs)} questions)")
            print(f"  {'Method':<20} {'MRR':>8} {'Recall@3':>10} {'Recall@5':>10}")
            print(f"  {'-'*50}")
            for m in methods:
                pq = results_all[m]["per_question"]
                subset = [
                    p for p in pq if p["type"] == qtype
                ]
                if not subset:
                    continue
                avg_mrr = sum(p["mrr"] for p in subset) / len(subset)
                avg_r3 = sum(p.get("recall@3", 0) for p in subset) / len(subset)
                avg_r5 = sum(p.get("recall@5", 0) for p in subset) / len(subset)
                label = results_all[m]["aggregated"]["method"]
                print(f"  {label:<20} {avg_mrr:>8.4f} {avg_r3:>10.4f} {avg_r5:>10.4f}")

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    os.makedirs("data/eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/eval_results/eval_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "test_set": test_set_path,
        "k_values": k_values,
        "methods": {
            m: results_all[m] for m in methods
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    CONFIG = {
        # Test set
        "test_set": "data/test_sets/splade_vs_dense_testset.json",

        # Data files
        "chunks_file": "data/chunks/DSL_handboek_mei_2024_chunks.json",
        "embeddings_hnsw": "data/index/embeddings_hnsw.json",
        "embeddings_knn": "data/index/embeddings_knn.json",
        "splade_index_file": "data/index/splade_index.json",
        "splade_nl_model": "sparse-encoder/splade-robbert-dutch-base-v1",
        "splade_nl_index_file": "data/index/splade_dutch_index.json",
        "e5nl_model": "clips/e5-large-trm-nl",
        "embeddings_e5nl_hnsw": "data/index/embeddings_e5nl_hnsw.json",
        "embeddings_e5nl_knn": "data/index/embeddings_e5nl_knn.json",

        # Which methods to evaluate
        "run_bm25": True,
        "run_splade": True,
        "run_splade_nl": True,
        "run_vector_hnsw": True,
        "run_vector_knn": True,
        "run_vector_e5nl_hnsw": True,
        "run_vector_e5nl_knn": True,
        "run_hybrid_bm25_hnsw": True,
        "run_hybrid_bm25_knn": True,
        "run_hybrid_splade_hnsw": True,
        "run_hybrid_splade_knn": True,
        "run_hybrid_splade_nl_hnsw": True,
        "run_hybrid_splade_nl_knn": True,
        "run_hybrid_splade_nl_e5nl": True,

        # Evaluation k values
        "k_values": [1, 3, 5, 10],
    }

    main(CONFIG)
