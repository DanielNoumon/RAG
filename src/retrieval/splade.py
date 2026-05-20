"""SPLADE sparse retrieval for chunk-based search.

SPLADE (Sparse Lexical and Expansion) uses a transformer to produce
learned sparse vectors over the full vocabulary.  Each token in the
vocabulary receives a non-negative weight via:

    w_t = max_{i} log(1 + ReLU(h_i · E_t))

where h_i are the contextual token representations and E_t is the
embedding column for vocabulary token t.  The result is a very sparse
dict {token_id: weight} that supports semantic term expansion while
remaining dot-product compatible with classical inverted indexes.

Default model: naver/splade-cocondenser-ensembledistil
"""
import json
import os
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class SPLADERetriever:
    """SPLADE-based sparse retrieval over chunked documents.

    Loads chunks from the same JSON format produced by chunker.py and
    ranks them using SPLADE sparse vector dot-product scoring.

    Parameters
    ----------
    chunks_path : str
        Path to chunks JSON file.
    model_name : str
        HuggingFace model id for a SPLADE-compatible MLM model.
    device : str | None
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    max_length : int
        Maximum token length for the transformer encoder.
    batch_size : int
        Number of chunks to encode per batch during index build.
    index_path : str | None
        Path to a pre-built SPLADE index JSON.  If the file exists
        it is loaded directly (skipping the expensive encode step).
        If ``None`` the index is built from scratch.
    doc_topn : int
        Keep at most this many dimensions per document vector.
    query_topn : int
        Keep at most this many dimensions per query vector.
    weight_threshold : float
        Drop dimensions with weight below this value.
    """

    DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"

    def __init__(
        self,
        chunks_path: str,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 32,
        index_path: Optional[str] = None,
        doc_topn: int = 256,
        query_topn: int = 128,
        weight_threshold: float = 0.01,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.doc_topn = doc_topn
        self.query_topn = query_topn
        self.weight_threshold = weight_threshold

        # Device
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.chunks: List[Dict[str, Any]] = []
        # Sparse index: list aligned with self.chunks
        # Each entry is a dict {token_id_str: weight}
        self._sparse_index: List[Dict[str, float]] = []

        print(f"SPLADE: Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("SPLADE: Model loaded.")

        self._load_chunks(chunks_path)

        # Load pre-built index if available, otherwise build fresh
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            self._build_index()
            if index_path:
                self.save_index(index_path)

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
        print(f"SPLADE: Loaded {len(self.chunks)} chunks from {path}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_batch(
        self,
        texts: List[str],
        topn: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Encode a batch of texts into SPLADE sparse vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        topn : int | None
            Max dimensions to keep per vector.  Falls back to
            ``self.doc_topn`` when ``None``.

        Returns a list of dicts mapping token_id (as str) → weight.
        """
        topn = topn if topn is not None else self.doc_topn

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {
            k: v.to(self.device) for k, v in encoded.items()
        }

        output = self.model(**encoded)
        # logits: (batch, seq_len, vocab_size)
        logits = output.logits

        # SPLADE aggregation: log(1 + relu(logits))
        values = torch.log1p(torch.relu(logits))

        # Mask padding tokens so they don't leak into max-pool
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        values = values.masked_fill(attention_mask == 0, 0.0)

        # Max-pool over sequence → (batch, vocab_size)
        sparse_vecs = values.max(dim=1).values

        results = []
        for vec in sparse_vecs:
            results.append(
                self._prune_vec(vec, topn, self.weight_threshold)
            )
        return results

    @staticmethod
    def _prune_vec(
        vec: torch.Tensor,
        topn: int,
        threshold: float,
    ) -> Dict[str, float]:
        """Threshold + top-N pruning of a sparse vector."""
        vec = torch.where(
            vec >= threshold, vec, torch.zeros_like(vec)
        )
        nonzero = vec.nonzero(as_tuple=True)[0]
        if len(nonzero) == 0:
            return {}
        weights = vec[nonzero]
        if len(nonzero) > topn:
            top_w, top_pos = torch.topk(weights, topn)
            nonzero = nonzero[top_pos]
            weights = top_w
        return {
            str(int(tid)): float(w)
            for tid, w in zip(nonzero.tolist(), weights.tolist())
        }

    def encode(self, text: str) -> Dict[str, float]:
        """Encode a single text into a SPLADE sparse vector.

        Uses ``query_topn`` for pruning (queries are typically
        pruned more aggressively than documents).
        """
        return self._encode_batch([text], topn=self.query_topn)[0]

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Encode all chunks and store sparse vectors in memory."""
        texts = [chunk["content"] for chunk in self.chunks]
        n = len(texts)
        self._sparse_index = []

        print(f"SPLADE: Encoding {n} chunks (batch_size={self.batch_size})...")
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_vecs = self._encode_batch(texts[start:end])
            self._sparse_index.extend(batch_vecs)
            if (start // self.batch_size) % 5 == 0:
                print(f"  SPLADE: encoded {end}/{n} chunks")

        print(f"SPLADE: Index built — {len(self._sparse_index)} sparse vectors.")

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """Save the sparse index to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``data/index/splade_index.json``).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_chunks": len(self.chunks),
            "vectors": self._sparse_index,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"SPLADE: Index saved to {path} ({size_mb:.1f} MB)")

    def load_index(self, path: str) -> None:
        """Load a previously saved sparse index from JSON.

        The number of vectors must match the number of loaded chunks.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SPLADE index not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        saved_model = payload.get("model_name", "")
        if saved_model and saved_model != self.model_name:
            print(
                f"SPLADE WARNING: index was built with '{saved_model}', "
                f"current model is '{self.model_name}'."
            )

        vectors = payload["vectors"]
        if len(vectors) != len(self.chunks):
            raise ValueError(
                f"Index has {len(vectors)} vectors but {len(self.chunks)} chunks loaded. "
                "Rebuild the index."
            )
        self._sparse_index = vectors
        print(f"SPLADE: Index loaded from {path} ({len(vectors)} vectors).")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _dot_product(
        self, query_vec: Dict[str, float], doc_vec: Dict[str, float]
    ) -> float:
        """Compute sparse dot product between query and document vectors."""
        score = 0.0
        # Iterate over the smaller vector for efficiency
        if len(query_vec) > len(doc_vec):
            query_vec, doc_vec = doc_vec, query_vec
        for tid, qw in query_vec.items():
            dw = doc_vec.get(tid, 0.0)
            if dw:
                score += qw * dw
        return score

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks ranked by SPLADE dot-product similarity.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of results to return.

        Returns
        -------
        List of dicts with keys: chunk_id, content, metadata, score.
        """
        query_vec = self.encode(query)

        scored: List[Tuple[float, int]] = []
        for idx, doc_vec in enumerate(self._sparse_index):
            s = self._dot_product(query_vec, doc_vec)
            if s > 0:
                scored.append((s, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for s, idx in scored[:top_k]:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", idx),
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": round(s, 4),
            })
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        if not self._sparse_index:
            avg_nnz = 0.0
        else:
            avg_nnz = sum(len(v) for v in self._sparse_index) / len(self._sparse_index)

        return {
            "total_chunks": len(self.chunks),
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "avg_nonzero_dims": round(avg_nnz, 1),
            "vocab_size": self.tokenizer.vocab_size,
        }
