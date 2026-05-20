# Dutch RAG Experimentation Toolkit

A modular toolkit for experimenting with Retrieval-Augmented Generation on **Dutch documents**. It provides a systematic way to compare retrieval strategies and measure their effectiveness on Dutch-language queries using a curated evaluation set and standard IR metrics (MRR, MAP, NDCG@k, Recall@k).

Retrieval is structured as a two-stage pipeline: a **first-pass retriever** narrows the corpus to a candidate set, followed by an optional **reranker** that re-scores candidates for higher precision.

## Project Structure

```
├── scripts/                    # Thin CLI entry points that bootstrap src modules
│   ├── run_vector_search.py     # Vector search test (HNSW / exhaustive KNN)
│   ├── run_keyword_search.py    # BM25-only keyword retrieval test
│   ├── run_hybrid_search.py     # Hybrid (vector + BM25) test + optional reranker
│   └── run_splade_search.py     # SPLADE sparse retrieval test
├── evaluation/                 # Retrieval evaluation framework
│   ├── run_evaluation.py        # Main evaluation: runs all methods, computes metrics
│   ├── metrics.py               # IR metrics (Recall@k, Precision@k, MRR, MAP, NDCG@k)
│   └── _print_results.py       # Pretty-print latest eval results
├── src/                        # Production-ready packages imported by scripts
│   ├── core/                   # Core RAG orchestration (embedding, storage, prompts, config)
│   │   ├── embedding_manager.py          # Default MiniLM embedding manager
│   │   ├── model_embedding_manager.py    # Multi-model manager (E5-NL, 8B models)
│   │   ├── hnsw_storage.py              # HNSW approximate nearest neighbor index
│   │   ├── json_storage.py              # KNN exhaustive search storage
│   │   ├── vector_search_pipeline_hnsw.py
│   │   ├── vector_search_pipeline_knn.py
│   │   ├── azure_openai.py
│   │   ├── config.py
│   │   └── prompts.py
│   ├── preprocessing/          # Chunking + embedding build scripts
│   │   ├── chunker.py
│   │   ├── build_embeddings.py          # Rebuild MiniLM KNN/HNSW stores
│   │   ├── build_all_embeddings.py      # Build stores for all embedding models
│   │   └── build_splade_index.py        # Build SPLADE sparse index
│   ├── retrieval/              # Retrieval strategies
│   │   ├── bm25.py                      # BM25 keyword retrieval
│   │   ├── splade.py                    # SPLADE sparse neural retrieval
│   │   ├── hybrid.py                    # Hybrid fusion (RRF / weighted)
│   │   └── rerankers/                   # Multiple reranking implementations
│   │       ├── cross_encoder_reranker.py
│   │       ├── colbert_reranker.py
│   │       ├── llm_reranker.py
│   │       └── compare_rerankers.py
│   └── utils/
├── data/
│   ├── documents/              # Source documents (PDFs, TXT)
│   ├── chunks/                 # Chunked documents (JSON)
│   ├── index/                  # All search indexes
│   │   ├── embeddings/         # Dense vector indexes (KNN/HNSW per model)
│   │   ├── splade/             # SPLADE sparse neural indexes
│   │   └── colbert/            # ColBERT pre-cached document embeddings (pkl)
│   ├── eval_results/           # Evaluation output (JSON, timestamped)
│   └── test_sets/              # Curated evaluation test sets
├── pyproject.toml              # Project config + dependencies (uv)
├── .env / .env.example
```

## Retrieval Methods

### Stage 1 — First-pass Retrieval

| Method | Model | MRR | Notes |
|--------|-------|-----|-------|
| Dense — bi-encoder | `clips/e5-large-trm-nl` (Dutch, 1024-dim) | 0.85 | Best performing |
| Dense — bi-encoder | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | 0.47 | Fast baseline |
| Sparse — SPLADE | `splade-robbert-dutch-base-v1` (Dutch) | 0.82 | Best sparse for Dutch |
| Sparse — SPLADE | `splade-cocondenser-ensembledistil` (English) | 0.60 | English baseline |
| Sparse — BM25 | — | 0.40 | TF-IDF keyword retrieval |
| Hybrid — RRF | SPLADE-NL + E5-NL | 0.85 | Fuses two strong Dutch retrievers |

Dense indexes support both HNSW (approximate, fast) and exhaustive KNN backends.

Three fusion strategies are available in `HybridRetriever`:

| Strategy | Description |
|----------|-------------|
| `rrf` | Reciprocal Rank Fusion — merges ranked lists by position; works with 2 or 3 sources |
| `weighted` | Normalised score combination — `alpha * dense + (1-alpha) * bm25` |
| `pool` | Takes top-K from each source independently and unions them into a single deduplicated candidate set; intended as input to a reranker that handles final ordering |

**Pool mode** preserves the distinct strengths of each retriever rather than collapsing them into one ranked list: dense covers semantic recall, BM25 covers exact keyword matches, SPLADE covers vocabulary expansion. The candidate budget per source is controlled by `pool_k_per_source`:

```python
# Equal budget — every source contributes the same number of candidates
"pool_k_per_source": 20          # → up to 60 unique chunks (3 sources × 20)

# Weighted budget — control the mix explicitly
"pool_k_per_source": {"dense": 30, "bm25": 10, "splade": 10}  # → up to 50 unique chunks, 60/20/20 split
```

The pool is deduplicated (chunks found by multiple sources are merged and annotated with all source names) and returned unranked — the downstream reranker handles final ordering.

**Important:** `top_k` is applied as a cap to the final pool, consistent with `rrf` and `weighted`. To pass the full pool to the reranker, set `top_k >= sum(pool_k_per_source)`. For the example above with budgets `{30, 10, 10}`, use `top_k=50`.

Add SPLADE as a third source via `splade_index_path` in the run script config.

### Stage 2 — Reranking

| Reranker | Type | Speed | Notes |
|----------|------|-------|-------|
| Cross-Encoder | Pointwise scoring | ~2.5s | Best speed/accuracy balance |
| ColBERT v2 | Late-interaction | ~4s | MaxSim token-level scoring |
| Reason-ModernColBERT | Late-interaction | ~4s | Dutch ModernBERT backbone |
| Agent-ModernColBERT | Late-interaction | ~4s | Dutch ModernBERT backbone |
| Jina ColBERT v2 | Late-interaction | ~4s | Multilingual, 128-dim |
| LLM (GPT-4o-mini) | Generative | ~40s | Highest accuracy, supports reasoning |

## Setup

1. Install [uv](https://docs.astral.sh/uv/) (fast Python package manager):
   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install dependencies and the package in editable mode:
   ```bash
   uv sync
   ```

3. Configure `.env` (copy from `.env.example`):
   ```
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

## Usage

### Building indexes

```bash
uv run python -m preprocessing.build_embeddings       # MiniLM KNN + HNSW
uv run python -m preprocessing.build_splade_index     # SPLADE sparse indexes
uv run python -m preprocessing.build_all_embeddings   # All embedding models
```

### Running retrieval

```bash
uv run python scripts/run_vector_search.py    # Vector search (HNSW/KNN)
uv run python scripts/run_keyword_search.py   # BM25 keyword search
uv run python scripts/run_splade_search.py    # SPLADE sparse retrieval
uv run python scripts/run_hybrid_search.py    # Hybrid fusion + reranker
```

### Running evaluation

```bash
uv run python -m evaluation.run_evaluation
```

This benchmarks all retrieval methods on the curated test set and reports:
- **MRR**, **MAP**, **NDCG@k**, **Recall@k**, **Precision@k**
- Per-question-type breakdown (factual, paraphrase, multi-aspect)
- Results saved to `data/eval_results/`

### Reranking

Configure in any run script:
```python
"rerank": True,
"reranker_type": "cross_encoder",  # "cross_encoder" | "colbert" | "llm"
"rerank_top_n": 5,
```

## Output

Evaluation results are saved to `data/eval_results/` as timestamped JSON files containing per-question metrics, retrieved chunk IDs, and aggregated scores across all methods.
