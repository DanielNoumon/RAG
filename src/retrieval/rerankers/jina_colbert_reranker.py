"""Jina ColBERT v2 reranker with pre-cached document embeddings.

Uses jinaai/jina-colbert-v2 (XLM-RoBERTa-Large backbone with ColBERT
late interaction).  Requires trust_remote_code=True — the projection
head (to 128-dim) is baked into the custom HF_ColBERT class, so no
separate Dense folder is needed.

Queries are prefixed with ``[QueryMarker]`` and documents with
``[DocumentMarker]`` as required by the model's training setup.
"""
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "jinaai/jina-colbert-v2"
DEFAULT_CACHE_PATH = (
    "data/colbert/jina_colbertv2_doc_embeddings.pkl"
)
EXPECTED_DIM = 128


class JinaColBERTReranker:
    """Reranks retrieved chunks using Jina ColBERT v2 MaxSim.

    Document embeddings are pre-computed (one array of shape
    ``(num_tokens, 128)`` per chunk) and stored as a pickle file.
    At rerank time only the query is encoded, then MaxSim is computed
    against the cached candidate embeddings.

    The public ``rerank()`` method follows the same signature as the
    other rerankers so they can be used interchangeably.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: str = DEFAULT_CACHE_PATH,
        top_n: int = 5,
        device: Optional[str] = None,
        max_doc_length: int = 512,
        max_query_length: int = 32,
        batch_size: int = 4,
    ):
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

        self._cache: Dict[str, np.ndarray] = {}

        self._load_model()
        self._load_cache()

        print(
            f"Jina ColBERT v2 Reranker loaded: {model_name} "
            f"on {self.device}  "
            f"({len(self._cache)} cached doc embeddings)"
        )

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> None:
        """Load tokenizer, backbone, and linear projection.

        The custom HF_ColBERT class maps to XLMRobertaModel and
        silently drops ``linear.weight`` (1024→128) from the
        checkpoint.  We extract it from the raw state dict and
        apply it ourselves — identical to the ColBERT v2 fix.
        """
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

        self._proj: Optional[torch.nn.Linear] = (
            self._load_projection()
        )
        # Cast projection to match backbone dtype (e.g. BFloat16).
        if self._proj is not None:
            model_dtype = next(self._model.parameters()).dtype
            self._proj = self._proj.to(model_dtype)

    def _load_projection(self) -> Optional[torch.nn.Linear]:
        """Extract linear.weight from the raw checkpoint."""
        from huggingface_hub import hf_hub_download

        try:
            from safetensors.torch import load_file
            wf = hf_hub_download(
                self.model_name, "model.safetensors"
            )
            state = load_file(wf, device="cpu")
        except Exception:
            try:
                wf = hf_hub_download(
                    self.model_name, "pytorch_model.bin"
                )
                state = torch.load(
                    wf, map_location="cpu", weights_only=True,
                )
            except Exception:
                print(
                    "  Jina ColBERT v2: could not load projection"
                    " weights — using raw backbone output."
                )
                return None

        if "linear.weight" not in state:
            return None

        w = state["linear.weight"]  # (128, 1024)
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
        """Load cached embeddings, clearing on dimension mismatch."""
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

        if loaded:
            sample = next(iter(loaded.values()))
            if sample.shape[-1] != EXPECTED_DIM:
                print(
                    f"  Jina ColBERT v2: cache dimension "
                    f"{sample.shape[-1]} != {EXPECTED_DIM} "
                    f"— cache cleared, rebuild with build_cache()."
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
        """Pre-encode documents and persist to disk."""
        texts = [ch[content_key] for ch in chunks]
        ids = [str(ch["chunk_id"]) for ch in chunks]
        embeddings = self._encode(texts, is_query=False)
        for cid, emb in zip(ids, embeddings):
            self._cache[cid] = emb
        self._save_cache()
        print(
            f"Jina ColBERT v2: cached {len(ids)} doc embeddings "
            f"-> {self.cache_path}"
        )

    # ----------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------
    @torch.no_grad()
    def _encode(
        self,
        texts: List[str],
        is_query: bool,
    ) -> List[np.ndarray]:
        """Encode texts to L2-normalised (num_tokens, 128) arrays.

        Prepends ``[QueryMarker]`` or ``[DocumentMarker]`` as the
        model requires.
        """
        prefix = (
            "[QueryMarker] " if is_query else "[DocumentMarker] "
        )
        max_len = (
            self.max_query_length if is_query
            else self.max_doc_length
        )
        prefixed = [prefix + t for t in texts]

        results: List[np.ndarray] = []
        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i: i + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc)
            embs = out.last_hidden_state  # (B, T, 1024 or 128)
            if self._proj is not None:
                embs = self._proj(embs)  # (B, T, 128)
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
        cid = str(chunk.get("chunk_id", ""))
        if cid in self._cache:
            return self._cache[cid]
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
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by Jina ColBERT v2 MaxSim score."""
        n = top_n or self.top_n
        if not chunks:
            return []

        if verbose:
            print(
                f"Reranking {len(chunks)} chunks with "
                f"Jina ColBERT v2 ({self.model_name})..."
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
