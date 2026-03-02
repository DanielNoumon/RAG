"""BGE-reranker-v2-gemma LLM-based reranker - DISABLED.

WARNING: This reranker requires downloading a ~10GB model.
The model must be downloaded locally before use, which can take significant
time and disk space. It also requires substantial RAM (~16GB+) to run.

For most use cases, use the cross-encoder or ColBERT rerankers instead,
which are much lighter and faster while still providing good accuracy.

To enable this reranker, you would need to:
1. Install FlagEmbedding: pip install -U FlagEmbedding
2. Ensure sufficient disk space for the 10GB model download
3. Ensure sufficient RAM (16GB+ recommended)
4. Uncomment and implement the BGEReranker class below

This file is currently a stub to prevent accidental heavy downloads.
"""


# Stub class to prevent import errors
class BGEReranker:
    """Reranks retrieved chunks using BGE-reranker-v2-gemma (2B LLM).

    This is a powerful LLM-based reranker built on Google's Gemma 2B model,
    trained by BAAI (Beijing Academy of Artificial Intelligence) for
    multilingual reranking tasks. It uses a causal language model approach
    where the model predicts "Yes" or "No" for query-passage relevance.

    The model supports multilingual content and can handle longer contexts
    than traditional cross-encoders.

    Default model: ``BAAI/bge-reranker-v2-gemma`` — 2B parameter LLM
    reranker with strong multilingual performance.

    The public ``rerank()`` method follows the same signature as the
    other rerankers so they can be used interchangeably in retrieval
    pipelines.
    """

    def __init__(self, **kwargs):
        """BGE reranker is disabled.
        
        This reranker requires a 10GB model download and 16GB+ RAM.
        Use cross-encoder or ColBERT rerankers instead.
        """
        raise NotImplementedError(
            "BGE reranker is disabled due to heavy resource requirements.\n"
            "The model requires ~10GB download and 16GB+ RAM.\n"
            "Please use CrossEncoderReranker or ColBERTReranker instead."
        )

    def rerank(self, *args, **kwargs):
        """BGE reranker is disabled."""
        raise NotImplementedError(
            "BGE reranker is disabled. Use CrossEncoderReranker or "
            "ColBERTReranker instead."
        )
