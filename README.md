# Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying a reference document using Azure OpenAI, vector search, BM25 and a reranker. Also an open-source embedder and PostgreSQL connection.

## Project Structure

```
├── scripts/                    # Thin CLI entry points that bootstrap src modules
│   ├── run_rag_test.py         # Drive the RAG query flow (imports src/core & src/retrieval)
│   ├── run_bm25_test.py        # Run BM25-only retrieval tests via src/retrieval
│   ├── run_hybrid_test.py      # Run hybrid (vector + BM25) retrieval tests via src/retrieval
│   └── debug_json.py           # Simple JSON serialization checks (moved to src/utils)
├── src/                        # Production-ready packages imported by scripts
│   ├── core/                   # Core RAG orchestration (embedding, storage, prompts, config)
│   │   ├── rag_pipeline_hnsw.py
│   │   ├── rag_pipeline_knn.py
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

2. Install dependencies:
   ```
   pip install -r requirements.txt
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

### Full pipeline (from scratch)

```bash
python scripts/chunker.py           # Chunk source document (delegates to src/preprocessing.chunker)
python scripts/run_hybrid_test.py   # Run retrieval + optional reranker answer generation
python scripts/run_rag_test.py      # Run RAG answer generation pipeline
```

### Configuration

Each script has a `CONFIG` section under `if __name__ == "__main__"` where parameters can be adjusted.

**chunker.py**
- `file_path` — Source file (.pdf or .txt)
- `chunk_size` — Words per chunk (default: 500)
- `overlap` — Overlapping words between chunks (default: 100)

**run_rag_test.py**
- `search_method` — `"hnsw"` (approximate, fast) or `"exhaustive_knn"` (exact)
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
