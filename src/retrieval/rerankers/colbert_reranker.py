"""ColBERT-based reranker using late interaction with pre-cached doc embeddings.

Uses colbert-ir/colbertv2.0 with MaxSim scoring.  Document token
embeddings are computed once and cached to disk; only the query is
encoded at rerank time.
"""
import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEFAULT_CACHE_PATH = "data/index/colbert/colbertv2_doc_embeddings.pkl"


class ColBERTReranker:
    """Reranks retrieved chunks using ColBERT late interaction.

    ColBERT (Contextualized Late Interaction over BERT) uses a novel
    late interaction architecture where query and document embeddings
    are computed independently, then combined via MaxSim operation.
    This provides better accuracy than bi-encoders while being more
    efficient than cross-encoders.

    Default model: ``colbert-ir/colbertv2.0`` -- state-of-the-art
    retrieval model trained on MS MARCO.

    Document embeddings are pre-computed and cached to disk.  At
    rerank time only the query is encoded, then MaxSim is computed
    against cached candidate embeddings.

    The public ``rerank()`` method follows the same signature as the
    other rerankers so they can be used interchangeably in retrieval
    pipelines.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        cache_path: str = DEFAULT_CACHE_PATH,
        top_n: int = 5,
        device: Optional[str] = None,
        max_doc_length: int = 512,
        max_query_length: int = 32,
    ):
        """Initialise the ColBERT reranker.

        Args:
            model_name: HuggingFace ColBERT model identifier.
            cache_path: Path to save/load pre-computed doc embeddings.
            top_n: Default number of chunks to keep after reranking.
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
                    ``None`` lets torch pick automatically.
            max_doc_length: Max token length when encoding documents.
            max_query_length: Max token length when encoding queries.
        """
        self.model_name = model_name
        self.cache_path = cache_path
        self.top_n = top_n
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # chunk_id -> np.ndarray (num_tokens, hidden_dim)
        self._cache: Dict[str, np.ndarray] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

        # ColBERT v2 has a linear projection (768→128) baked into its
        # checkpoint but AutoModel (BertModel) silently drops it.
        # We load it separately from the raw state dict.
        self._proj: Optional[torch.nn.Linear] = (
            self._load_projection()
        )

        self._load_cache()

        print(
            f"ColBERT Reranker loaded: {model_name} on "
            f"{self.device}  "
            f"({len(self._cache)} cached doc embeddings)"
        )

    # ----------------------------------------------------------
    # Projection loading
    # ----------------------------------------------------------
    def _load_projection(self) -> Optional[torch.nn.Linear]:
        """Extract the ColBERT linear projection from the checkpoint.

        AutoModel maps colbert-ir/colbertv2.0 to BertModel, which has
        no linear layer, so the projection weight is silently dropped.
        We load the raw state dict from the hub and pull it out
        manually — identical to how ReasonColBERTReranker loads its
        Dense head.
        """
        from huggingface_hub import hf_hub_download

        try:
            wf = hf_hub_download(
                self.model_name, "pytorch_model.bin"
            )
            state = torch.load(
                wf, map_location="cpu", weights_only=True,
            )
        except Exception:
            try:
                from safetensors.torch import load_file
                wf = hf_hub_download(
                    self.model_name, "model.safetensors"
                )
                state = load_file(wf, device="cpu")
            except Exception:
                print(
                    "  ColBERT v2: could not load projection "
                    "weights — using raw BERT output (768-dim)."
                )
                return None

        if "linear.weight" not in state:
            return None

        w = state["linear.weight"]  # (128, 768)
        proj = torch.nn.Linear(
            w.shape[1], w.shape[0], bias=False
        )
        proj.weight = torch.nn.Parameter(w)
        proj.to(self.device)
        proj.eval()
        return proj

    # ----------------------------------------------------------
    # Embedding cache
    # ----------------------------------------------------------
    def _load_cache(self) -> None:
        """Load cached document embeddings from disk (if exists).

        Validates that the cached embedding dimension matches the
        expected output dimension (128 when projection is loaded,
        768 otherwise).  Clears the cache if stale so that
        build_cache() will be called before reranking.
        """
        if not os.path.exists(self.cache_path):
            return

        with open(self.cache_path, "rb") as f:
            raw = pickle.load(f)

        if isinstance(raw, dict):
            loaded = {str(k): v for k, v in raw.items()}
        elif isinstance(raw, list):
            loaded = {str(i): v for i, v in enumerate(raw)}
        else:
            return

        # Check embedding dimension matches current model config.
        expected_dim = (
            128 if self._proj is not None else
            self.model.config.hidden_size
        )
        if loaded:
            sample = next(iter(loaded.values()))
            if sample.shape[-1] != expected_dim:
                print(
                    f"  ColBERT v2: cache dimension "
                    f"{sample.shape[-1]} != expected "
                    f"{expected_dim} — cache cleared, "
                    f"rebuild with build_cache()."
                )
                return

        self._cache = loaded

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
        embeddings = self._encode_docs(texts)
        for cid, emb in zip(ids, embeddings):
            self._cache[cid] = emb
        self._save_cache()
        print(
            f"ColBERT v2: cached {len(ids)} doc embeddings "
            f"-> {self.cache_path}"
        )

    # ----------------------------------------------------------
    # Encoding helpers
    # ----------------------------------------------------------
    @torch.no_grad()
    def _encode_docs(
        self, texts: List[str],
    ) -> List[np.ndarray]:
        """Encode documents into L2-normalised token embeddings."""
        results: List[np.ndarray] = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_doc_length,
            ).to(self.device)
            out = self.model(**inputs).last_hidden_state
            if self._proj is not None:
                out = self._proj(out)
            emb = F.normalize(out, p=2, dim=-1)
            length = inputs["attention_mask"][0].sum().item()
            results.append(
                emb[0, :length].cpu().numpy()
            )
        return results

    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query into L2-normalised token embeddings."""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
        ).to(self.device)
        out = self.model(**inputs).last_hidden_state
        if self._proj is not None:
            out = self._proj(out)
        emb = F.normalize(out, p=2, dim=-1)
        length = inputs["attention_mask"][0].sum().item()
        return emb[0, :length].cpu().numpy()

    def _get_doc_embedding(
        self, chunk: Dict[str, Any], content_key: str,
    ) -> np.ndarray:
        """Return cached embedding or encode on-the-fly."""
        cid = str(chunk.get("chunk_id", ""))
        if cid in self._cache:
            return self._cache[cid]
        # Fallback: encode now (not cached)
        return self._encode_docs([chunk[content_key]])[0]

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
        """Rerank chunks by ColBERT late interaction relevance.

        Args:
            query: The user query.
            chunks: Retrieved chunks; each dict must contain
                    *content_key* (and ideally ``chunk_id``).
            top_n: How many to keep (defaults to ``self.top_n``).
            content_key: Key in each chunk dict holding the text.
            verbose: Whether to print progress information.
            batch_size: Unused (interface compatibility).

        Returns:
            Top-n chunks sorted descending by ColBERT MaxSim score.
        """
        n = top_n or self.top_n
        if not chunks:
            return []

        if verbose:
            print(
                f"Reranking {len(chunks)} chunks with ColBERT "
                f"({self.model_name})..."
            )

        q_emb = self._encode_query(query)

        scored_chunks = []
        for chunk in chunks:
            d_emb = self._get_doc_embedding(
                chunk, content_key
            )
            score = self._maxsim(q_emb, d_emb)
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = score
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(
            key=lambda x: x["rerank_score"], reverse=True
        )

        if verbose:
            print(
                f"  Reranking complete. "
                f"Keeping top {n}/{len(chunks)} chunks."
            )

        return scored_chunks[:n]


if __name__ == "__main__":
    # Demo: build cache from chunks, then rerank sample
    import json as _json

    CHUNKS_FILE = (
        "data/chunks/DSL_handboek_mei_2024_chunks.json"
    )

    print("=== ColBERT v2 Reranker Demo ===\n")
    reranker = ColBERTReranker()

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
