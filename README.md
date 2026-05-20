# Dutch RAG Experimentation Toolkit

A modular toolkit for experimenting with Retrieval-Augmented Generation (RAG) on **Dutch documents**. The goal is to systematically compare retrieval strategies — from simple keyword search to neural sparse retrieval and late-interaction reranking — and measure their effectiveness on Dutch-language queries using a curated evaluation framework.

The pipeline is structured around two retrieval stages:

**Stage 1 — First-pass retrieval** narrows the corpus to a candidate set:
- **Dense bi-encoder** — semantic vector search (HNSW / exhaustive KNN) using Dutch-tuned or multilingual embedding models
- **Sparse: BM25** — classical TF-IDF keyword retrieval
- **Sparse: SPLADE** — learned sparse neural retrieval with vocabulary expansion; Dutch and English variants
- **Hybrid** — reciprocal rank fusion (RRF) of any sparse + dense pair

**Stage 2 — Reranking** re-scores the candidate set for precision:
- **Cross-encoder** — pointwise query-document scoring; fast and accurate
- **Late-interaction (ColBERT)** — MaxSim token-level interaction; multiple variants including Dutch-tuned ModernColBERT models
- **LLM reranker** — GPT-4o-mini scores and reasons over candidates; highest accuracy, highest cost

Results across all configurations are benchmarked with standard IR metrics (MRR, MAP, NDCG@k, Recall@k) on a curated Dutch test set.

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

### Stage 1 — First-pass Retrieval

#### Dense bi-encoder

| Model | Dim | Language | MRR | Notes |
|-------|-----|----------|-----|-------|
| `clips/e5-large-trm-nl` | 1024 | Dutch | 0.85 | Best performing |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual | 0.47 | Default, fast |

Both models support HNSW (approximate, fast) and exhaustive KNN backends.

#### Sparse: BM25

| Method | MRR | Notes |
|--------|-----|-------|
| BM25 | 0.40 | TF-IDF keyword retrieval, no index build required |

#### Sparse: SPLADE

| Model | Language | MRR | Notes |
|-------|----------|-----|-------|
| `sparse-encoder/splade-robbert-dutch-base-v1` | Dutch | 0.82 | Best sparse model for Dutch |
| `naver/splade-cocondenser-ensembledistil` | English | 0.60 | Default SPLADE |

#### Hybrid fusion

| Combination | MRR | Notes |
|-------------|-----|-------|
| SPLADE-NL + E5-NL (RRF) | 0.85 | Fusing two strong Dutch retrievers |
| Any sparse + dense pair | varies | Configurable via RRF or weighted fusion |

### Stage 2 — Reranking

| Reranker | Type | Speed | MRR boost | Notes |
|----------|------|-------|-----------|-------|
| **Cross-Encoder** | Pointwise | Fast (~2.5s) | +moderate | Best speed/accuracy balance |
| **ColBERT v2** | Late-interaction | Medium (~4s) | +moderate | MaxSim token scoring |
| **Reason-ModernColBERT** | Late-interaction | Medium | +good | Dutch ModernBERT backbone |
| **Agent-ModernColBERT** | Late-interaction | Medium | +good | Dutch ModernBERT backbone |
| **Jina ColBERT v2** | Late-interaction | Medium | +moderate | Multilingual, 128-dim |
| **LLM (GPT-4o-mini)** | Generative | Slow (~40s) | +excellent | Reasoning support, expensive |

Configure reranking in run scripts via:
```python
"rerank": True,
"reranker_type": "cross_encoder",  # Options: "llm", "cross_encoder", "colbert"
"rerank_top_n": 5,
```

## Output

Evaluation results are saved to `data/eval_results/` as timestamped JSON files containing per-question metrics, retrieved chunk IDs, and aggregated scores across all methods.
