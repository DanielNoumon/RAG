"""BM25 retrieval for chunk-based search."""
import json
import os
import math
from typing import List, Dict, Any
from collections import Counter


class BM25Retriever:
    """BM25-based keyword retrieval over chunked documents.

    Loads chunks from the same JSON format produced by chunker.py
    and ranks them using the Okapi BM25 scoring function.
    """

    def __init__(self, chunks_path: str, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            chunks_path: Path to chunks JSON file.
            k1: Term saturation parameter (default 1.5).
            b: Length normalization parameter (default 0.75).
        """
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.avg_dl: float = 0.0
        self.n_docs: int = 0

        self._load_chunks(chunks_path)
        self._build_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_chunks(self, path: str) -> None:
        """Load chunks from JSON produced by chunker.py."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunks file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        print(f"BM25: Loaded {len(self.chunks)} chunks from {path}")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def _build_index(self) -> None:
        """Compute document frequencies and average document length."""
        self.tokenized_corpus = [
            self._tokenize(chunk["content"]) for chunk in self.chunks
        ]
        self.n_docs = len(self.tokenized_corpus)
        self.avg_dl = (
            sum(len(doc) for doc in self.tokenized_corpus) / self.n_docs
            if self.n_docs > 0
            else 0.0
        )

        # Document frequency: how many docs contain each term
        df: Dict[str, int] = {}
        for doc_tokens in self.tokenized_corpus:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1
        self.doc_freqs = df

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue

            df = self.doc_freqs[term]
            idf = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1.0
            )
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avg_dl)
            )
            score += idf * (numerator / denominator)

        return score

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks ranked by BM25 relevance.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: chunk_id, content, metadata, score.
        """
        query_tokens = self._tokenize(query)

        scored = []
        for idx in range(self.n_docs):
            s = self._score(query_tokens, idx)
            if s > 0:
                chunk = self.chunks[idx]
                scored.append({
                    "chunk_id": chunk.get("chunk_id", idx),
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                    "score": round(s, 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "total_chunks": self.n_docs,
            "avg_chunk_length": round(self.avg_dl, 1),
            "vocabulary_size": len(self.doc_freqs),
            "k1": self.k1,
            "b": self.b,
        }
