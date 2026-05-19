"""Reason-ModernColBERT reranker with pre-cached document embeddings.

Uses lightonai/Reason-ModernColBERT (ModernBERT backbone + Dense 768→128
projection) with MaxSim scoring.  Document token embeddings are computed
once and cached to disk; only the query is encoded at rerank time.

Loads the model directly via ``transformers`` / ``huggingface_hub``
(bypasses SentenceTransformer and pylate, which both have compatibility
issues with this model on Windows).
"""
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "lightonai/Reason-ModernColBERT"
DEFAULT_CACHE_PATH = "data/colbert/doc_embeddings.pkl"


class ReasonColBERTReranker:
    """Reranks retrieved chunks using Reason-ModernColBERT MaxSim.

    Document embeddings are pre-computed (one array of shape
    ``(num_tokens, 128)`` per chunk) and stored as a pickle file.
    At rerank time only the query is encoded, then MaxSim is computed
    against the cached candidate embeddings.

    The public ``rerank()`` method follows the same signature as the
    other rerankers so they can be used interchangeably in retrieval
    pipelines.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: str = DEFAULT_CACHE_PATH,
        top_n: int = 5,
        device: Optional[str] = None,
        max_doc_length: int = 512,
        max_query_length: int = 128,
        batch_size: int = 4,
    ):
        """Initialise the Reason-ModernColBERT reranker.

        Args:
            model_name: HuggingFace model identifier.
            cache_path: Path to save/load pre-computed doc embeddings.
            top_n: Default number of chunks to keep after reranking.
            device: Torch device (``"cpu"``, ``"cuda"``).  ``None``
                    for auto-detect.
            max_doc_length: Max token length when encoding documents.
            max_query_length: Max token length when encoding queries.
            batch_size: Encoding batch size for document pre-encoding.
        """
        self.model_name = model_name
        self.cache_path = cache_path
        self.top_n = top_n
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.batch_size = batch_size

        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # chunk_id → np.ndarray (num_tokens, 128)
        self._cache: Dict[str, np.ndarray] = {}

        self._load_model()
        self._load_cache()

        print(
            f"Reason-ColBERT Reranker loaded: {model_name} "
            f"on {self.device}  "
            f"({len(self._cache)} cached doc embeddings)"
        )

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> None:
        """Load tokenizer, backbone, and Dense projection head."""
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        self._backbone = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._backbone.to(self.device)
        self._backbone.eval()

        # Dense projection head (768 → 128, no bias)
        dense_cfg_file = hf_hub_download(
            self.model_name, "1_Dense/config.json"
        )
        with open(dense_cfg_file, "r") as f:
            dense_cfg = json.load(f)

        in_feat = dense_cfg["in_features"]
        out_feat = dense_cfg["out_features"]
        self._proj = torch.nn.Linear(in_feat, out_feat, bias=False)

        try:
            wf = hf_hub_download(
                self.model_name, "1_Dense/pytorch_model.bin"
            )
            state = torch.load(
                wf, map_location="cpu", weights_only=True,
            )
        except Exception:
            from safetensors.torch import load_file
            wf = hf_hub_download(
                self.model_name, "1_Dense/model.safetensors"
            )
            state = load_file(wf, device="cpu")

        key = next(iter(state.keys()))
        self._proj.weight = torch.nn.Parameter(state[key])
        self._proj.to(self.device)
        self._proj.eval()

    # ----------------------------------------------------------
    # Embedding cache
    # ----------------------------------------------------------
    def _load_cache(self) -> None:
        """Load cached document embeddings from disk (if exists)."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, dict):
                self._cache = {
                    str(k): v for k, v in raw.items()
                }
            elif isinstance(raw, list):
                # Legacy format (list of arrays, index-keyed)
                self._cache = {
                    str(i): v for i, v in enumerate(raw)
                }
            else:
                self._cache = {}

    def _save_cache(self) -> None:
        os.makedirs(
            os.path.dirname(self.cache_path), exist_ok=True
        )
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    def build_cache(
        self,
        chunks: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> None:
        """Pre-encode documents and persist to disk.

        Call this once after chunking to pre-compute all document
        token embeddings.  Subsequent reranker calls will look up
        embeddings from the cache instead of re-encoding.

        Args:
            chunks: List of chunk dicts (must have ``chunk_id``
                    and *content_key*).
            content_key: Key in each chunk dict holding the text.
        """
        texts = [ch[content_key] for ch in chunks]
        ids = [str(ch["chunk_id"]) for ch in chunks]
        embeddings = self._encode(texts, is_query=False)
        for cid, emb in zip(ids, embeddings):
            self._cache[cid] = emb
        self._save_cache()
        print(
            f"Reason-ColBERT: cached {len(ids)} doc embeddings "
            f"→ {self.cache_path}"
        )

    # ----------------------------------------------------------
    # Encoding helpers
    # ----------------------------------------------------------
    @torch.no_grad()
    def _encode(
        self,
        texts: List[str],
        is_query: bool,
    ) -> List[np.ndarray]:
        """Encode texts into L2-normalised (num_tokens, 128) arrays."""
        max_len = (
            self.max_query_length if is_query
            else self.max_doc_length
        )
        results: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._backbone(**enc)
            embs = self._proj(out.last_hidden_state)  # (B,T,128)
            embs = F.normalize(embs, p=2, dim=-1)

            mask = enc["attention_mask"]
            for j in range(len(batch)):
                length = mask[j].sum().item()
                results.append(
                    embs[j, :length].cpu().numpy()
                )
        return results

    def _encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True)[0]

    def _get_doc_embedding(
        self, chunk: Dict[str, Any], content_key: str,
    ) -> np.ndarray:
        """Return cached embedding or encode on-the-fly."""
        cid = str(chunk.get("chunk_id", ""))
        if cid in self._cache:
            return self._cache[cid]
        # Fallback: encode now (not cached)
        return self._encode(
            [chunk[content_key]], is_query=False
        )[0]

    # ----------------------------------------------------------
    # MaxSim scoring
    # ----------------------------------------------------------
    @staticmethod
    def _maxsim(
        query_emb: np.ndarray,
        doc_emb: np.ndarray,
    ) -> float:
        """MaxSim: per-query-token max cosine sim, then sum."""
        sim = query_emb @ doc_emb.T  # (Tq, Td)
        return float(sim.max(axis=1).sum())

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        content_key: str = "content",
        verbose: bool = True,
        batch_size: int = 1,  # unused, kept for interface compat
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by Reason-ModernColBERT MaxSim score.

        Args:
            query: The user query.
            chunks: Retrieved chunks; each dict must contain
                    *content_key* (and ideally ``chunk_id``).
            top_n: How many to keep (defaults to ``self.top_n``).
            content_key: Key holding document text.
            verbose: Whether to print progress.
            batch_size: Unused (interface compatibility).

        Returns:
            Top-n chunks sorted descending by MaxSim score.
        """
        n = top_n or self.top_n
        if not chunks:
            return []

        if verbose:
            print(
                f"Reranking {len(chunks)} chunks with "
                f"Reason-ColBERT ({self.model_name})..."
            )

        q_emb = self._encode_query(query)

        scored = []
        for chunk in chunks:
            d_emb = self._get_doc_embedding(chunk, content_key)
            score = self._maxsim(q_emb, d_emb)
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = score
            scored.append(chunk_copy)

        scored.sort(
            key=lambda x: x["rerank_score"], reverse=True
        )

        if verbose:
            print(
                f"  Reranking complete. "
                f"Keeping top {n}/{len(chunks)} chunks."
            )

        return scored[:n]


if __name__ == "__main__":
    # Demo: build cache from chunks, then rerank sample
    import json as _json

    CHUNKS_FILE = (
        "data/chunks/DSL_handboek_mei_2024_chunks.json"
    )

    print("=== Reason-ColBERT Reranker Demo ===\n")
    reranker = ReasonColBERTReranker()

    # Build/refresh cache from the full chunk set
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = _json.load(f)["chunks"]
    reranker.build_cache(all_chunks)

    # Take a subset as "first-stage candidates"
    candidates = all_chunks[:10]
    query = "Hoeveel vakantiedagen krijg ik?"

    print(f"\nQuery: {query}")
    print(f"Candidates: {len(candidates)}")

    results = reranker.rerank(query, candidates, top_n=5)
    print("\n=== Top 5 after reranking ===")
    for i, r in enumerate(results, 1):
        print(
            f"{i}. [ID:{r['chunk_id']}] "
            f"Score: {r['rerank_score']:.4f} - "
            f"{r['content'][:60]}..."
        )
    print("\n=== Demo Complete ===")
