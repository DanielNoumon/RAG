"""Hybrid retrieval combining vector search, BM25, and optionally SPLADE."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from retrieval.bm25 import BM25Retriever
from core.embedding_manager import EmbeddingManager
from core.hnsw_storage import HNSWStorageManager
from core.json_storage import JSONStorageManager


class HybridRetriever:
    """Combines dense vector search, BM25, and optionally SPLADE.

    Supports three fusion strategies:

    - **rrf** (Reciprocal Rank Fusion): merges ranked lists via
      score = sum(1 / (k + rank)).  Robust default, no score
      normalisation needed.
    - **weighted**: min-max normalises scores to [0, 1], then combines
      as  alpha * vector + (1 - alpha) * bm25.
    - **pool**: takes the top-K results from each retriever independently
      and unions them into a single candidate set.  Preserves the distinct
      strengths of each method — semantic recall (dense), exact keyword
      matches (BM25), and vocabulary expansion (SPLADE) — without collapsing
      them into a single ranked list.  Best used upstream of a reranker that
      handles final ordering.

    Parameters
    ----------
    chunks_path : str
        Path to chunks JSON (produced by chunker.py).
    embeddings_path : str
        Path to vector embeddings file (KNN json or HNSW json).
    vector_backend : str
        ``"hnsw"`` or ``"knn"`` — which storage to use.
    bm25_k1, bm25_b : float
        BM25 tuning parameters forwarded to BM25Retriever.
    splade_index_path : str | None
        Path to a pre-built SPLADE index.  When provided, SPLADE is loaded
        and becomes available as a third source in ``fusion="pool"`` (and
        also in ``"rrf"``).
    splade_model : str | None
        HuggingFace model id for SPLADE.  Defaults to
        ``SPLADERetriever.DEFAULT_MODEL`` when ``None``.
    """

    def __init__(
        self,
        chunks_path: str,
        embeddings_path: str,
        vector_backend: str = "hnsw",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        overlap_boost: Optional[float] = None,
        splade_index_path: Optional[str] = None,
        splade_model: Optional[str] = None,
    ):
        self.overlap_boost = overlap_boost

        # BM25 (keyword)
        self.bm25 = BM25Retriever(chunks_path, k1=bm25_k1, b=bm25_b)

        # Dense vector (semantic)
        self.embedding_mgr = EmbeddingManager()
        if vector_backend == "hnsw":
            self.vector_store = HNSWStorageManager(embeddings_path)
        elif vector_backend == "knn":
            self.vector_store = JSONStorageManager(embeddings_path)
        else:
            raise ValueError(
                f"Unknown vector_backend: {vector_backend}. "
                "Use 'hnsw' or 'knn'."
            )
        self.vector_backend = vector_backend

        # SPLADE — optional third source
        self.splade = None
        if splade_index_path:
            from retrieval.splade import SPLADERetriever
            self.splade = SPLADERetriever(
                chunks_path=chunks_path,
                model_name=splade_model or SPLADERetriever.DEFAULT_MODEL,
                index_path=splade_index_path,
            )
            print(f"HybridRetriever: SPLADE loaded from {splade_index_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        fusion: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        splade_top_k: Optional[int] = None,
        pool_k_per_source: Optional[Any] = None,
        overlap_boost: Optional[float] = None,
        vector_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks using hybrid search.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Maximum number of results returned by this method.  Applied
            consistently across all fusion modes.  For ``fusion="pool"``
            this caps the deduplicated pool; set it >= sum of
            ``pool_k_per_source`` values if you want the full pool to
            reach a downstream reranker.
        fusion : str
            ``"rrf"`` — Reciprocal Rank Fusion across all active sources.
            ``"weighted"`` — normalised score combination (dense + BM25 only).
            ``"pool"`` — take top-K from each source independently and union
            them.  Intended as the candidate set for a downstream reranker.
        alpha : float
            Weight for vector score in ``fusion="weighted"``.
        rrf_k : int
            Smoothing constant for ``fusion="rrf"`` (default 60).
        vector_top_k, bm25_top_k, splade_top_k : int | None
            Per-source candidate counts for rrf/weighted.  Default: top_k * 3.
        pool_k_per_source : int | dict[str, int] | None
            Controls how many candidates each source contributes in
            ``fusion="pool"``.

            - **int** — same budget for every source.
              E.g. ``20`` → top-20 from dense, top-20 from BM25,
              top-20 from SPLADE (if active).
            - **dict** — per-source budgets, enabling weighted
              contribution.  Keys are ``"dense"``, ``"bm25"``,
              ``"splade"``.  Missing sources fall back to ``top_k``.
              E.g. ``{"dense": 30, "bm25": 10, "splade": 10}`` yields
              a 60% / 20% / 20% split over a 50-chunk candidate pool.

            Default (``None``): ``top_k`` per source.  The total pool
            size before reranking is up to the sum of all budgets
            (reduced by duplicate chunks found by multiple sources).
        overlap_boost : float | None
            Multiplicative boost for docs found by multiple methods (rrf /
            weighted only).  E.g. 1.2 = 20% boost.
        """
        candidate_k = top_k * 3
        v_k = vector_top_k or candidate_k
        b_k = bm25_top_k or candidate_k
        s_k = splade_top_k or candidate_k
        boost = overlap_boost if overlap_boost is not None else self.overlap_boost

        # --- Retrieve from all sources ---
        futures = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures["dense"] = executor.submit(
                self._vector_search, query, v_k, vector_threshold
            )
            futures["bm25"] = executor.submit(self.bm25.search, query, b_k)
            if self.splade is not None:
                futures["splade"] = executor.submit(
                    self.splade.search, query, s_k
                )

        vector_results = futures["dense"].result()
        bm25_results = futures["bm25"].result()
        splade_results = futures["splade"].result() if "splade" in futures else []

        # --- Fuse ---
        if fusion == "rrf":
            fused = self._rrf_fusion(
                vector_results, bm25_results, splade_results,
                k=rrf_k, overlap_boost=boost,
            )
            return fused[:top_k]
        elif fusion == "weighted":
            fused = self._weighted_fusion(
                vector_results, bm25_results,
                alpha=alpha, overlap_boost=boost,
            )
            return fused[:top_k]
        elif fusion == "pool":
            budgets = self._resolve_pool_budgets(
                pool_k_per_source, top_k,
                has_splade=bool(splade_results),
            )
            sources = {
                "dense": vector_results[:budgets["dense"]],
                "bm25": bm25_results[:budgets["bm25"]],
            }
            if splade_results:
                sources["splade"] = splade_results[:budgets["splade"]]
            pool = self._pool_fusion(sources)
            return pool[:top_k]
        else:
            raise ValueError(
                f"Unknown fusion: {fusion!r}. Use 'rrf', 'weighted', or 'pool'."
            )

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics from all active retrieval backends."""
        stats = {
            "bm25": self.bm25.get_stats(),
            "vector_backend": self.vector_backend,
            "vector_docs": self.vector_store.get_stats().get("total_documents", 0),
            "embedding_dim": self.embedding_mgr.get_embedding_dimension(),
            "splade": self.splade.get_stats() if self.splade else None,
        }
        return stats

    # ------------------------------------------------------------------
    # Vector search helper
    # ------------------------------------------------------------------

    def _vector_search(
        self, query: str, top_k: int, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Run vector similarity search and return unified format."""
        query_emb = self.embedding_mgr.embed_text(query)
        results = self.vector_store.search_similar(
            query_emb, limit=top_k, threshold=threshold
        )
        # Normalise output format
        return [
            {
                "content": r["content"],
                "metadata": r.get("metadata", {}),
                "score": r["similarity"],
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Dict],
        bm25_results: List[Dict],
        splade_results: Optional[List[Dict]] = None,
        k: int = 60,
        overlap_boost: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (Cormack et al., 2009).

        score(d) = sum over methods of 1 / (k + rank_i).
        Supports two or three sources (dense, BM25, SPLADE).
        If overlap_boost is set, docs found by multiple methods get a
        multiplicative boost (e.g. 1.2 = 20% boost).

        RRF uses ranks only, not score magnitudes.
        """
        scores: Dict[str, Dict[str, Any]] = {}

        sources = [
            ("dense", vector_results, "vector_search_rank", "vector_similarity"),
            ("bm25", bm25_results, "bm25_search_rank", "bm25_score"),
        ]
        if splade_results:
            sources.append(("splade", splade_results, "splade_rank", "splade_score"))

        all_rank_keys = [rk for _, _, rk, _ in sources]

        for _name, results, rank_key, score_key in sources:
            for rank, doc in enumerate(results, start=1):
                key = doc["content"]
                if key not in scores:
                    scores[key] = {
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "fusion_score": 0.0,
                        **{rk: None for rk in all_rank_keys},
                        "vector_similarity": 0.0,
                        "bm25_score": 0.0,
                        "splade_score": 0.0,
                    }
                scores[key]["fusion_score"] += 1.0 / (k + rank)
                scores[key][rank_key] = rank
                scores[key][score_key] = doc.get("score", 0.0)

        if overlap_boost is not None:
            for doc in scores.values():
                n_sources_found = sum(
                    1 for rk in all_rank_keys if doc.get(rk) is not None
                )
                if n_sources_found > 1:
                    doc["fusion_score"] *= overlap_boost

        fused = sorted(scores.values(), key=lambda x: x["fusion_score"], reverse=True)
        for doc in fused:
            doc["fusion_score"] = round(doc["fusion_score"], 6)
        return fused

    @staticmethod
    def _resolve_pool_budgets(
        pool_k_per_source: Any,
        default: int,
        has_splade: bool,
    ) -> Dict[str, int]:
        """Resolve pool budgets to a per-source dict.

        Accepts:
        - ``None``  → ``default`` for every active source.
        - ``int``   → that value for every active source.
        - ``dict``  → used as-is; missing keys fall back to ``default``.

        Returns a dict with keys ``"dense"``, ``"bm25"``, ``"splade"``.
        """
        sources = ["dense", "bm25"] + (["splade"] if has_splade else [])
        if pool_k_per_source is None or isinstance(pool_k_per_source, int):
            k = pool_k_per_source or default
            return {s: k for s in sources}
        if isinstance(pool_k_per_source, dict):
            return {s: pool_k_per_source.get(s, default) for s in sources}
        raise TypeError(
            f"pool_k_per_source must be int, dict, or None — got {type(pool_k_per_source)}"
        )

    @staticmethod
    def _pool_fusion(
        sources: Dict[str, List[Dict]],
    ) -> List[Dict[str, Any]]:
        """Multi-source candidate pooling.

        Unions the already-budgeted results from each source into a single
        deduplicated candidate set.  Chunks found by multiple sources are
        merged into one entry and annotated with every source name that
        retrieved them.

        The returned list is not ranked — it is intended as the input to
        a downstream reranker that produces the final ordering.
        """
        seen: Dict[str, Dict[str, Any]] = {}

        for name, results in sources.items():
            for doc in results:
                key = doc["content"]
                if key not in seen:
                    seen[key] = {
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0.0),
                        "sources": [name],
                    }
                elif name not in seen[key]["sources"]:
                    seen[key]["sources"].append(name)

        return list(seen.values())

    @staticmethod
    def _weighted_fusion(
        vector_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float = 0.5,
        overlap_boost: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Weighted score combination with min-max normalisation.

        final_score = alpha * norm(vector) + (1 - alpha) * norm(bm25)
        If overlap_boost is set, docs found by both methods get a
        multiplicative boost (e.g. 1.2 = 20% boost).
        """

        def _min_max(vals):
            lo, hi = min(vals), max(vals)
            span = hi - lo if hi != lo else 1.0
            return [(v - lo) / span for v in vals]

        # Normalise vector scores
        if vector_results:
            v_scores = _min_max([r["score"] for r in vector_results])
        else:
            v_scores = []

        # Normalise BM25 scores
        if bm25_results:
            b_scores = _min_max([r["score"] for r in bm25_results])
        else:
            b_scores = []

        scores: Dict[str, Dict[str, Any]] = {}

        for doc, norm_s in zip(vector_results, v_scores):
            key = doc["content"]
            scores[key] = {
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "hybrid_score": alpha * norm_s,
                "vector_score": doc["score"],
                "bm25_score": 0.0,
            }

        for doc, norm_s in zip(bm25_results, b_scores):
            key = doc["content"]
            if key in scores:
                scores[key]["hybrid_score"] += (1 - alpha) * norm_s
                scores[key]["bm25_score"] = doc["score"]
            else:
                scores[key] = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "hybrid_score": (1 - alpha) * norm_s,
                    "vector_score": 0.0,
                    "bm25_score": doc["score"],
                }

        # Apply overlap boost to docs found by both methods
        if overlap_boost is not None:
            for doc in scores.values():
                if doc["vector_score"] > 0 and doc["bm25_score"] > 0:
                    doc["hybrid_score"] *= overlap_boost

        fused = sorted(
            scores.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )
        for doc in fused:
            doc["hybrid_score"] = round(doc["hybrid_score"], 6)
        return fused
