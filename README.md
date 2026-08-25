# RAG Repository Analyzer

Retrieval-Augmented Generation system for answering natural-language questions about a GitHub repository. Indexes the repo into ChromaDB and answers via semantic search + GPT.

## Architecture

Two indexing pipelines sit behind a common `Pipeline` protocol (`src/pipeline.py`):

- **`DefaultPipeline`** (`src/default_pipeline.py`) — fetch → chunk → embed → persist to ChromaDB. Routes queries between text and code collections at retrieval time.
- **`LazyPipeline`** (`src/lazy_pipeline.py`) — builds an in-memory metadata index only; embeds and scores files on demand per query.

`RAGService` (`src/rag_service.py`) is a thin orchestrator: it takes a `Pipeline` and owns prompt construction, the OpenAI call, and source formatting.

Other modules:
- `src/github_parser.py` — GitHub fetch + LangChain text splitting
- `src/embedding.py` — OpenAI `text-embedding-ada-002` (text), CodeBERT (code), deterministic hash fallback when offline
- `src/chromaDB_setup.py` — persistent ChromaDB collections
- `src/metadata_index.py`, `src/file_selector.py`, `src/lazy_parser.py` — lazy pipeline components
- `backend/main.py` — FastAPI REST endpoints + static frontend serving

All external services degrade gracefully: GitHub unavailable → synthetic repo, OpenAI unavailable → hash embeddings, CodeBERT unavailable → fallback.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY="..."
GITHUB_API_KEY="..."
USE_LAZY_PIPELINE=0   # set to 1 to enable the lazy pipeline
```

Run the backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Frontend

Next.js 13+ App Router + Tailwind, configured for static export. Lives in `./frontend`. Requires Node 18.18+.

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
npm run build    # static export to frontend/out
```

The FastAPI backend serves the static export from its root in production.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/index` | Index a repository |
| POST | `/query` | Query an indexed repository |
| GET | `/repositories` | List indexed repos |
| GET | `/status/{repo_path}` | Repo status |
| DELETE | `/repository/{repo_path}` | Remove indexed repo |
| GET | `/health` | Health check |

## Tests

```bash
pytest tests/                          # unit + integration, runs offline
python test_api.py                     # manual HTTP probe (requires running server)
python test_analyze_and_query.py       # manual end-to-end probe
```

`tests/test_default_pipeline.py` and `tests/test_lazy_pipeline.py` cover each pipeline in isolation. `tests/test_rag_service_integration.py` exercises both pipelines through `RAGService` end-to-end.

## Roadmap

- [ ] Embedding cache (Redis or local file) to skip repeat API calls
- [ ] Semantic chunking via tree-sitter for code-aware boundaries
- [ ] Hybrid retrieval: BM25 alongside vector similarity
- [ ] Cross-encoder re-ranking of retrieved chunks
- [ ] Multi-turn conversation memory persistence
- [ ] Observability: structured logs, tracing, error monitoring
- [ ] GitLab / Bitbucket / on-prem repository providers

## Contributing

Issues and pull requests welcome.
