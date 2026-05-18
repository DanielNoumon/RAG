"""Retrieval evaluation metrics."""
import math
from typing import List, Set


def ndcg_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).
    NDCG@k = DCG@k / IDCG@k
    """
    if not relevant_ids or k == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-based

    # Ideal DCG: all relevant docs ranked at the top
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """Fraction of relevant chunks found in the top-k results.

    recall@k = |relevant ∩ retrieved[:k]| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(relevant_ids & top_k) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """Fraction of top-k results that are relevant.

    precision@k = |relevant ∩ retrieved[:k]| / k
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(relevant_ids & top_k) / k


def reciprocal_rank(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
) -> float:
    """Average precision across all recall points."""
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / len(relevant_ids)
