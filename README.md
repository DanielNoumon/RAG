# Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying Dutch reference documents using Azure OpenAI, multiple retrieval strategies (vector search, BM25, SPLADE sparse retrieval), hybrid fusion (RRF), rerankers, and a systematic evaluation framework to compare retrieval effectiveness.

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
│   ├── embeddings/             # Vector embeddings (KNN/HNSW per model)
│   ├── splade/                 # SPLADE sparse indexes
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

### Search Methods

| Method | Description | MRR |
|--------|-------------|-----|
| **E5-NL (HNSW/KNN)** | Dutch-finetuned E5-large dense retrieval | 0.85 |
| **SPLADE-NL** | Dutch SPLADE sparse neural retrieval | 0.82 |
| **SPLADE-NL+E5-NL** | RRF hybrid of two strong retrievers | 0.85 |
| **SPLADE** | English SPLADE (naver/splade-cocondenser) | 0.60 |
| **Vector (HNSW/KNN)** | MiniLM multilingual dense retrieval | 0.47 |
| **BM25** | TF-IDF keyword retrieval | 0.40 |
| **Hybrid (sparse+dense)** | RRF fusion of any sparse + dense pair | varies |

### Embedding Models

| Model | Dim | Language | Notes |
|-------|-----|----------|-------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual | Default, fast |
| `clips/e5-large-trm-nl` | 1024 | Dutch | Best performing |
| `sparse-encoder/splade-robbert-dutch-base-v1` | sparse | Dutch | SPLADE variant |
| `naver/splade-cocondenser-ensembledistil` | sparse | English | Default SPLADE |

### Rerankers

| Reranker | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| **Cross-Encoder** | Fast (~2.5s) | Good | Default choice, best speed/accuracy balance |
| **ColBERT** | Medium (~4s) | Good | Late interaction, good for longer contexts |
| **LLM** | Slow (~40s) | Excellent | Most accurate, supports reasoning, expensive |

Configure in run scripts via:
```python
"rerank": True,
"reranker_type": "cross_encoder",  # Options: "llm", "cross_encoder", "colbert"
"rerank_top_n": 5,
```

## Output

Evaluation results are saved to `data/eval_results/` as timestamped JSON files containing per-question metrics, retrieved chunk IDs, and aggregated scores across all methods.
