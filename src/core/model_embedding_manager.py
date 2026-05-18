"""Flexible embedding manager supporting multiple models with asymmetric encoding.

Handles query vs document encoding differences across model families:
- Symmetric models: same encode() for queries and documents
- Asymmetric models: encode_query() / encode_document() or prompt_name
"""
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# Registry of supported models and their configurations
MODEL_REGISTRY = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "short_name": "MiniLM",
        "dims": 384,
        "encoding": "symmetric",
        "model_kwargs": {},
        "tokenizer_kwargs": {},
    },
    "clips/e5-large-trm-nl": {
        "short_name": "E5-NL",
        "dims": 1024,
        "encoding": "encode_query_document",
        "model_kwargs": {},
        "tokenizer_kwargs": {},
    },
    "Octen/Octen-Embedding-8B": {
        "short_name": "Octen-8B",
        "dims": 4096,
        "encoding": "symmetric",
        "model_kwargs": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "tokenizer_kwargs": {"padding_side": "left"},
    },
    "Qwen/Qwen3-Embedding-8B": {
        "short_name": "Qwen3-8B",
        "dims": 4096,
        "encoding": "prompt_name",
        "query_prompt_name": "query",
        "model_kwargs": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "tokenizer_kwargs": {"padding_side": "left"},
    },
    "nvidia/llama-embed-nemotron-8b": {
        "short_name": "Nemotron-8B",
        "dims": 4096,
        "encoding": "encode_query_document",
        "trust_remote_code": True,
        "model_kwargs": {
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16",
        },
        "tokenizer_kwargs": {"padding_side": "left"},
    },
}


class ModelEmbeddingManager:
    """Unified embedding manager for multiple model architectures."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.config = MODEL_REGISTRY.get(model_name, {})
        self.short_name = self.config.get("short_name", model_name)
        self.encoding_style = self.config.get("encoding", "symmetric")

        # Build kwargs for SentenceTransformer
        st_kwargs = {}
        if device:
            st_kwargs["device"] = device
        if self.config.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        if self.config.get("model_kwargs"):
            st_kwargs["model_kwargs"] = self.config["model_kwargs"]
        if self.config.get("tokenizer_kwargs"):
            st_kwargs["tokenizer_kwargs"] = self.config[
                "tokenizer_kwargs"
            ]

        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name, **st_kwargs)
        print(
            f"  Loaded. Dims={self.get_embedding_dimension()}"
        )

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a query text using the appropriate method."""
        if self.encoding_style == "encode_query_document":
            return self.model.encode_query(text, convert_to_numpy=True)
        elif self.encoding_style == "prompt_name":
            prompt = self.config.get("query_prompt_name", "query")
            return self.model.encode(
                text, prompt_name=prompt, convert_to_numpy=True,
            )
        else:
            return self.model.encode(text, convert_to_numpy=True)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """Encode multiple query texts."""
        if self.encoding_style == "encode_query_document":
            return self.model.encode_query(
                texts, convert_to_numpy=True,
            )
        elif self.encoding_style == "prompt_name":
            prompt = self.config.get("query_prompt_name", "query")
            return self.model.encode(
                texts, prompt_name=prompt, convert_to_numpy=True,
            )
        else:
            return self.model.encode(texts, convert_to_numpy=True)

    def embed_document(self, text: str) -> np.ndarray:
        """Encode a document text using the appropriate method."""
        if self.encoding_style == "encode_query_document":
            return self.model.encode_document(
                text, convert_to_numpy=True,
            )
        else:
            # Symmetric and prompt_name: documents use plain encode
            return self.model.encode(text, convert_to_numpy=True)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Encode multiple document texts."""
        if self.encoding_style == "encode_query_document":
            return self.model.encode_document(
                texts, convert_to_numpy=True,
            )
        else:
            return self.model.encode(texts, convert_to_numpy=True)
