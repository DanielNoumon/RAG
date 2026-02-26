# Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying a reference document using Azure OpenAI, vector search, BM25 and a reranker. Also an open-source embedder and PostgreSQL connection.

## Project Structure

```
├── scripts/                    # Thin CLI entry points that bootstrap src modules
│   ├── run_vector_search.py     # Vector search test (HNSW / exhaustive KNN)
│   ├── run_keyword_search.py    # BM25-only keyword retrieval test
│   ├── run_hybrid_search.py     # Hybrid (vector + BM25) test + optional reranker
│   └── debug_json.py           # Simple JSON serialization checks (moved to src/utils)
├── src/                        # Production-ready packages imported by scripts
│   ├── core/                   # Core RAG orchestration (embedding, storage, prompts, config)
│   │   ├── vector_search_pipeline_hnsw.py
│   │   ├── vector_search_pipeline_knn.py
│   │   ├── hnsw_storage.py
│   │   ├── json_storage.py
│   │   ├── embedding_manager.py
│   │   ├── azure_openai.py
│   │   ├── config.py
│   │   └── prompts.py
│   ├── preprocessing/          # Chunking + embedding helper scripts
│   │   ├── chunker.py
│   │   └── build_embeddings.py
│   ├── retrieval/              # Retrieval strategies (vector / BM25 / reranker)
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   └── utils/                  # Supporting utilities (inspect/debug helpers)
│       ├── inspect_results.py
│       └── debug_json.py
├── data/
│   ├── documents/              # Source documents (PDFs, TXT)
│   ├── chunks/                 # Chunked documents (JSON)
│   ├── embeddings/             # Vector embeddings
│   └── test_results/           # Saved test results (JSON, timestamped)
├── requirements.txt
├── .env / .env.example
```

## Setup

1. Create and activate a conda environment:
   ```
   conda create -n test_rag python=3.11 -y
   conda activate test_rag
   ```

2. Install dependencies (optional when you install editable mode, but useful if you need the packages without the editable install):
   ```
   pip install -r requirements.txt
   ```

3. Install the package in editable mode so CLI scripts can import `core`/`retrieval` directly; this is the only required step for dependency setup as it installs everything listed above:
   ```
   pip install -e .
   ```

4. Configure `.env` (copy from `.env.example`):
   ```
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

## Usage

### Full pipeline (from scratch)

```bash
python scripts/chunker.py           # Chunk source document (delegates to src/preprocessing.chunker)
python scripts/run_hybrid_search.py # Run hybrid retrieval + reranker test flows
python scripts/run_vector_search.py # Run vector-only retrieval test (HNSW/knn)
```

> **Note:** The CLI scripts now expect the repository to be installed (editable install recommended above) so that `core.*` and `retrieval.*` import paths resolve without modifying `sys.path` manually.

### Configuration

Each script has a `CONFIG` section under `if __name__ == "__main__"` where parameters can be adjusted.

**chunker.py**
- `file_path` — Source file (.pdf or .txt)
- `chunk_size` — Words per chunk (default: 500)
- `overlap` — Overlapping words between chunks (default: 100)

**run_vector_search.py**
- `storage_file` — Path to embeddings file
- `embedding_model` — Sentence transformer model name
- `top_k` — Number of chunks to retrieve per query
- `show_chunks` — Print full chunk content in terminal (default: False)
- `similarity_threshold` — Optional minimum similarity filter
- `questions` — List of test questions
- `chunks_file`: "data/chunks/document_handbook_mei_2024_chunks.json"

### Search Methods

| Method | Best for | How it works |
|--------|----------|-------------|
| **HNSW** | Large datasets (100s+ docs) | Approximate nearest neighbor via graph traversal, O(log n) |
| **Exhaustive KNN** | Small datasets | Exact cosine similarity over all documents, O(n) |
| **BM25** | Keyword-heavy queries | TF-IDF based scoring with length normalisation, no embeddings needed |
| **Hybrid** | Best overall quality | Combines vector + BM25 via Reciprocal Rank Fusion (RRF) or weighted scores |

## Output

Test results are automatically saved to `data/test_results/` as timestamped JSON files containing questions, answers, retrieved chunks, and similarity scores.
