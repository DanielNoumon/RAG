"""Evaluate rerankers on a ground-truth test set.

Each reranker receives *all* chunks from the corpus (unordered)
and must score / rank them by relevance to the query.  The output
is then truncated to ``top_n`` and evaluated with Recall@k,
Precision@k, MRR, MAP, and NDCG@k.

This measures pure reranking quality — no first-stage retriever
is involved.

Usage (from project root):
    uv run python -m evaluation.run_reranker_evaluation
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _chunk_id_from_result(result: Dict[str, Any]) -> int:
    cid = result.get("chunk_id", result.get("id"))
    if cid is None:
        return -1
    return int(cid)


def _load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


# ------------------------------------------------------------------
# Evaluation core
# ------------------------------------------------------------------

def evaluate_reranker(
    method_name: str,
    reranker,
    all_chunks: List[Dict[str, Any]],
    questions: List[Dict],
    k_values: List[int],
    top_n: int = 10,
) -> Dict[str, Any]:
    """Evaluate a single reranker.

    Parameters
    ----------
    method_name : str
        Human-readable label.
    reranker : object
        Must have ``.rerank(query, chunks, top_n=...)``.
    all_chunks : list[dict]
        The full corpus of chunks (given unordered to the
        reranker for every question).
    questions : list[dict]
        Test set with ``relevant_chunk_ids``.
    k_values : list[int]
        Values of k for Recall/Precision/NDCG.
    top_n : int
        Number of chunks the reranker should keep.
    """
    per_question: List[Dict[str, Any]] = []
    all_mrr: List[float] = []
    all_ap: List[float] = []
    recall_sums = {k: 0.0 for k in k_values}
    prec_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}

    total_time = 0.0
    max_k = max(k_values)

    for q in questions:
        query = q["question"]
        relevant = set(q["relevant_chunk_ids"])

        t0 = time.perf_counter()
        reranked = reranker.rerank(
            query, all_chunks,
            top_n=top_n,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        retrieved_ids = [
            _chunk_id_from_result(r) for r in reranked
        ]

        mrr = reciprocal_rank(retrieved_ids, relevant)
        ap = average_precision(retrieved_ids, relevant)
        all_mrr.append(mrr)
        all_ap.append(ap)

        q_metrics: Dict[str, Any] = {
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

    n_q = len(questions)
    aggregated: Dict[str, Any] = {
        "method": method_name,
        "num_questions": n_q,
        "mean_mrr": round(
            sum(all_mrr) / n_q, 4
        ) if n_q else 0,
        "map": round(
            sum(all_ap) / n_q, 4
        ) if n_q else 0,
        "total_latency_s": round(total_time, 3),
        "latency_per_query_s": round(
            total_time / n_q, 4
        ) if n_q else 0,
    }
    for k in k_values:
        aggregated[f"mean_ndcg@{k}"] = (
            round(ndcg_sums[k] / n_q, 4) if n_q else 0
        )
        aggregated[f"mean_recall@{k}"] = (
            round(recall_sums[k] / n_q, 4) if n_q else 0
        )
        aggregated[f"mean_precision@{k}"] = (
            round(prec_sums[k] / n_q, 4) if n_q else 0
        )

    return {
        "aggregated": aggregated,
        "per_question": per_question,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(config: Dict[str, Any]):
    test_set_path = config["test_set"]
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    questions = test_set["questions"]

    chunks_path = config["chunks_file"]
    all_chunks = _load_chunks(chunks_path)
    print(f"Loaded {len(all_chunks)} chunks from {chunks_path}")

    k_values = config.get("k_values", [1, 3, 5, 10])
    top_n = config.get("top_n", 10)
    results_all: Dict[str, Any] = {}

    # -- Cross-Encoder --
    if config.get("run_cross_encoder", True):
        from retrieval.rerankers.cross_encoder_reranker import (
            CrossEncoderReranker,
        )
        print("=" * 60)
        print("Evaluating: Cross-Encoder")
        print("=" * 60)
        ce = CrossEncoderReranker(
            model_name=config.get(
                "cross_encoder_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            top_n=top_n,
        )
        results_all["cross_encoder"] = evaluate_reranker(
            "CrossEncoder",
            ce, all_chunks, questions, k_values, top_n,
        )

    # -- ColBERT (colbertv2.0) --
    if config.get("run_colbert", True):
        from retrieval.rerankers.colbert_reranker import (
            ColBERTReranker,
        )
        print("=" * 60)
        print("Evaluating: ColBERT (v2.0)")
        print("=" * 60)
        cb = ColBERTReranker(
            model_name=config.get(
                "colbert_model",
                "colbert-ir/colbertv2.0",
            ),
            cache_path=config.get(
                "colbert_cache",
                "data/colbert/colbertv2_doc_embeddings.pkl",
            ),
            top_n=top_n,
        )
        results_all["colbert"] = evaluate_reranker(
            "ColBERT v2",
            cb, all_chunks, questions, k_values, top_n,
        )

    # -- Reason-ColBERT (Reason-ModernColBERT) --
    if config.get("run_reason_colbert", True):
        from retrieval.rerankers.reason_colbert_reranker import (
            ReasonColBERTReranker,
        )
        print("=" * 60)
        print("Evaluating: Reason-ColBERT")
        print("=" * 60)
        rc = ReasonColBERTReranker(
            model_name=config.get(
                "reason_colbert_model",
                "lightonai/Reason-ModernColBERT",
            ),
            cache_path=config.get(
                "reason_colbert_cache",
                "data/colbert/doc_embeddings.pkl",
            ),
            top_n=top_n,
        )
        results_all["reason_colbert"] = evaluate_reranker(
            "R-ColBERT",
            rc, all_chunks, questions, k_values, top_n,
        )

    # -- LLM Reranker (Azure OpenAI) --
    if config.get("run_llm_reranker", False):
        from retrieval.rerankers.llm_reranker import Reranker
        print("=" * 60)
        print("Evaluating: LLM reranker")
        print("=" * 60)
        try:
            llm = Reranker(
                model_name=config.get("llm_reranker_model"),
                top_n=top_n,
            )
            results_all["llm"] = evaluate_reranker(
                "LLM",
                llm, all_chunks, questions,
                k_values, top_n,
            )
        except Exception as exc:
            print(f"  LLM reranker failed: {exc}")
            print("  Skipping -- continuing with "
                  "remaining rerankers.")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print(
        f"  RERANKER COMPARISON  "
        f"(25 chunks -> top {top_n})"
    )
    print("=" * 80)

    methods = list(results_all.keys())
    header = [f"{'Metric':<25}"]
    for m in methods:
        label = results_all[m]["aggregated"]["method"]
        header.append(f"{label:>18}")
    print("".join(header))
    print("-" * (25 + 18 * len(methods)))

    metric_keys = ["mean_mrr", "map"]
    for k in k_values:
        metric_keys.append(f"mean_ndcg@{k}")
    for k in k_values:
        metric_keys.append(f"mean_recall@{k}")
    for k in k_values:
        metric_keys.append(f"mean_precision@{k}")
    metric_keys.append("latency_per_query_s")

    for mk in metric_keys:
        row = [f"{mk:<25}"]
        for m in methods:
            val = results_all[m]["aggregated"].get(mk, "")
            if isinstance(val, float):
                row.append(f"{val:>18.4f}")
            else:
                row.append(f"{str(val):>18}")
        print("".join(row))

    print("=" * (25 + 18 * len(methods)))
    print(
        "\n  Note: R-ColBERT and ColBERT v2 use "
        "pre-cached doc embeddings (only query encoded"
        "\n  at runtime). CrossEncoder re-encodes all "
        "docs per query (no caching). Latency is"
        "\n  not directly comparable across "
        "architectures."
    )

    # ------------------------------------------------------------------
    # Per-question type breakdown
    # ------------------------------------------------------------------
    types = sorted(
        set(q.get("type", "") for q in questions)
    )
    if len(types) > 1:
        print("\n")
        print("=" * 80)
        print("  BREAKDOWN BY QUESTION TYPE")
        print("=" * 80)
        for qtype in types:
            type_qs = [
                q for q in questions
                if q.get("type") == qtype
            ]
            if not type_qs:
                continue
            print(
                f"\n  Type: {qtype} "
                f"({len(type_qs)} questions)"
            )
            hdr = f"  {'Method':<20} {'MRR':>8}"
            hdr += f" {'Recall@3':>10} {'Recall@5':>10}"
            print(hdr)
            print(f"  {'-' * 50}")
            for m in methods:
                pq = results_all[m]["per_question"]
                subset = [
                    p for p in pq if p["type"] == qtype
                ]
                if not subset:
                    continue
                avg_mrr = (
                    sum(p["mrr"] for p in subset)
                    / len(subset)
                )
                avg_r3 = (
                    sum(
                        p.get("recall@3", 0)
                        for p in subset
                    ) / len(subset)
                )
                avg_r5 = (
                    sum(
                        p.get("recall@5", 0)
                        for p in subset
                    ) / len(subset)
                )
                label = (
                    results_all[m]["aggregated"]["method"]
                )
                print(
                    f"  {label:<20} {avg_mrr:>8.4f}"
                    f" {avg_r3:>10.4f} {avg_r5:>10.4f}"
                )

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    os.makedirs("data/eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        f"data/eval_results/reranker_eval_{timestamp}.json"
    )

    output = {
        "timestamp": timestamp,
        "test_set": test_set_path,
        "num_chunks": len(all_chunks),
        "top_n": top_n,
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
        "test_set": (
            "data/test_sets/splade_vs_dense_testset.json"
        ),

        # All chunks (fed to every reranker)
        "chunks_file": (
            "data/chunks/DSL_handboek_mei_2024_chunks.json"
        ),

        # Reranker keeps top_n from the 25 chunks
        "top_n": 10,

        # Which rerankers to evaluate
        "run_cross_encoder": True,
        "run_colbert": True,
        "run_reason_colbert": True,
        "run_llm_reranker": True,
        "llm_reranker_model": "gpt5-nano",

        # Reranker model overrides (optional)
        # "cross_encoder_model": "...",
        # "colbert_model": "...",
        # "reason_colbert_model": "...",
        # "reason_colbert_cache": "...",

        # Evaluation k values
        "k_values": [1, 3, 5, 10],
    }

    main(CONFIG)
