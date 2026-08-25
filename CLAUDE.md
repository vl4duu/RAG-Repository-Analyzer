# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Repository Analyzer: a system that indexes GitHub repositories and answers natural language questions about them using Retrieval Augmented Generation. Users submit a GitHub repo path, the system indexes it into ChromaDB, and questions are answered using semantic search + GPT.

## Commands

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest test_api.py
pytest test_analyze_and_query.py
pytest tests/test_rag_service_integration.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev       # Development at http://localhost:3000
npm run build     # Static export build
```

### Deployment
The project deploys to Render via `render.yaml`. The `build.sh` script handles full deployment builds (installs Python deps + builds Next.js static export).

## Architecture

### Request Flow
1. **Index** (`POST /index`): GitHub repo path → GitHub API fetches files → LangChain text splitter chunks them → embeddings generated → stored in ChromaDB
2. **Query** (`POST /query`): question → embed query → ChromaDB similarity search → retrieved chunks + question → GPT-3.5-turbo → answer

### Two Indexing Pipelines
- **Default pipeline**: fetches all files upfront, embeds all chunks, persists to ChromaDB
- **Lazy pipeline** (`USE_LAZY_PIPELINE=1` env var): builds lightweight metadata index only, parses/embeds files on-demand per query with LRU caching

### Key Source Files
- `backend/main.py` — FastAPI app: REST endpoints, CORS config, static file serving, error handling
- `src/rag_service.py` — Core orchestrator: manages both indexing pipelines, ChromaDB collections, conversation memory
- `src/github_parser.py` — GitHub API integration: fetches files, chunks into text/code segments
- `src/embedding.py` — Embedding generation: OpenAI `text-embedding-ada-002` for text, CodeBERT for code; falls back to hash-based embeddings if unavailable
- `src/chromaDB_setup.py` — ChromaDB collection creation/management (telemetry disabled)
- `src/metadata_index.py` — Lightweight in-memory metadata index for the lazy pipeline
- `src/file_selector.py` — Query-driven file candidate selection (lazy pipeline)
- `src/lazy_parser.py` — On-demand file parsing with LRU cache (capacity 100)
- `src/models.py` — Pydantic request/response models

### Query Classification
Queries are classified as `text`, `code`, or `both`, routing to separate ChromaDB collections. Results are weighted differently per type.

### Graceful Degradation
All external services have fallbacks: GitHub API unavailable → synthetic content; OpenAI unavailable → local hash-based embeddings; CodeBERT unavailable → fallback embeddings.

## Environment Variables
```
OPENAI_API_KEY      # OpenAI API key
GITHUB_API_KEY      # GitHub personal access token
ALLOWED_ORIGINS     # CORS allowed origins (comma-separated)
PORT                # API port (default: 10000)
USE_LAZY_PIPELINE   # Set to 1 to enable lazy indexing pipeline
```

## API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/index` | Index a GitHub repository |
| POST | `/query` | Query an indexed repository |
| GET | `/repositories` | List all indexed repos |
| GET | `/status/{repo_path}` | Check if repo is indexed |
| DELETE | `/repository/{repo_path}` | Remove indexed repository |
| POST | `/clear-memory/{repo_path}` | Clear conversation memory |
| GET | `/health` | Health check |

## Frontend
Next.js 13+ App Router with Tailwind CSS, configured for static export (`output: 'export'` in `next.config.js`). The FastAPI backend serves the static export from its root, with `/undefined/*` catch-all handlers to detect frontend API misconfiguration.
