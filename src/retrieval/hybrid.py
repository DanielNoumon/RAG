"""Hybrid retrieval combining vector search and BM25 keyword search."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from retrieval.bm25 import BM25Retriever
from core.embedding_manager import EmbeddingManager
from core.hnsw_storage import HNSWStorageManager
from core.json_storage import JSONStorageManager


class HybridRetriever:
    """Combines vector similarity search with BM25 keyword search.

    Supports two fusion strategies:
      - **RRF** (Reciprocal Rank Fusion): merges ranked lists using
        score = sum(1 / (k + rank)) per method.  Robust, no score
        normalisation needed.  Good default.
      - **weighted**: min-max normalises both score sets to [0, 1],
        then combines as  alpha * vector + (1 - alpha) * bm25.

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
    """

    def __init__(
        self,
        chunks_path: str,
        embeddings_path: str,
        vector_backend: str = "hnsw",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        overlap_boost: Optional[float] = None,
    ):
        self.overlap_boost = overlap_boost
        # BM25 (keyword)
        self.bm25 = BM25Retriever(
            chunks_path, k1=bm25_k1, b=bm25_b
        )

        # Vector (semantic)
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
        overlap_boost: Optional[float] = None,
        vector_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks using hybrid search.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Number of final results to return.
        fusion : str
            ``"rrf"`` for Reciprocal Rank Fusion, ``"weighted"`` for
            normalised score combination.
        alpha : float
            Only used when fusion="weighted".  Weight for vector score;
            (1 - alpha) is used for BM25.
        rrf_k : int
            Only used when fusion="rrf".  Smoothing constant (default 60,
            the standard value from the RRF paper).
        vector_top_k, bm25_top_k : int | None
            How many candidates to fetch from each method before fusion.
            Defaults to ``top_k * 3`` to give the fuser enough candidates.
        overlap_boost : float | None
            Multiplicative boost for docs found by both methods.
            Overrides the instance-level setting.  E.g. 1.2 = 20%% boost.
            ``None`` means no boost.
        """
        candidate_k = top_k * 3
        v_k = vector_top_k or candidate_k
        b_k = bm25_top_k or candidate_k
        boost = overlap_boost if overlap_boost is not None else self.overlap_boost

        # Retrieve from both in parallel
        vector_results = []
        bm25_results = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            v_future = pool.submit(self._vector_search, query, v_k, vector_threshold)
            b_future = pool.submit(self.bm25.search, query, b_k)

            for future in as_completed([v_future, b_future]):
                if future is v_future:
                    vector_results = future.result()
                else:
                    bm25_results = future.result()

        # Fuse
        if fusion == "rrf":
            fused = self._rrf_fusion(
                vector_results, bm25_results,
                k=rrf_k, overlap_boost=boost,
            )
        elif fusion == "weighted":
            fused = self._weighted_fusion(
                vector_results, bm25_results,
                alpha=alpha, overlap_boost=boost,
            )
        else:
            raise ValueError(
                f"Unknown fusion: {fusion}. Use 'rrf' or 'weighted'."
            )

        return fused[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics from both retrieval backends."""
        return {
            "bm25": self.bm25.get_stats(),
            "vector_backend": self.vector_backend,
            "vector_docs": self.vector_store.get_stats().get(
                "total_documents", 0
            ),
            "embedding_dim": self.embedding_mgr.get_embedding_dimension(),
        }

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
        k: int = 60,
        overlap_boost: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (Cormack et al., 2009).

        score(d) = sum over methods of  1 / (k + rank_i)
        where rank_i is the 1-based rank of document d in method i.
        If overlap_boost is set, docs found by both methods get a
        multiplicative boost (e.g. 1.2 = 20% boost).
        
        NOTE: 
        - RRF uses RANKS only, not score magnitudes
        - Raw RRF scores kept (not normalized) - this is a rank fusion heuristic
        - Higher fusion scores indicate better rank consensus across methods
        - Use scores for sorting, not as absolute relevance measures
        """
        scores: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(vector_results, start=1):
            key = doc["content"]
            if key not in scores:
                scores[key] = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "fusion_score": 0.0,
                    "vector_search_rank": None,
                    "bm25_search_rank": None,
                    "vector_similarity": 0.0,
                    "bm25_score": 0.0,
                }
            scores[key]["fusion_score"] += 1.0 / (k + rank)
            scores[key]["vector_search_rank"] = rank
            scores[key]["vector_similarity"] = doc["score"]

        for rank, doc in enumerate(bm25_results, start=1):
            key = doc["content"]
            if key not in scores:
                scores[key] = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "fusion_score": 0.0,
                    "vector_search_rank": None,
                    "bm25_search_rank": None,
                    "vector_similarity": 0.0,
                    "bm25_score": 0.0,
                }
            scores[key]["fusion_score"] += 1.0 / (k + rank)
            scores[key]["bm25_search_rank"] = rank
            scores[key]["bm25_score"] = doc["score"]

        # Apply overlap boost to docs found by both methods
        if overlap_boost is not None:
            for doc in scores.values():
                if doc["vector_search_rank"] is not None and doc["bm25_search_rank"] is not None:
                    doc["fusion_score"] *= overlap_boost

        fused = sorted(
            scores.values(),
            key=lambda x: x["fusion_score"],
            reverse=True,
        )
        
        # Round scores for readability (keep raw RRF values)
        for doc in fused:
            doc["fusion_score"] = round(doc["fusion_score"], 6)
        return fused

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
