# src/ — RAG core engine

This package does the actual work behind `POST /index` and `POST /query` in `backend/main.py`:
fetch repo files → chunk → embed → store in ChromaDB → retrieve → prompt GPT. `backend/main.py`
never touches these modules directly except through `RAGService`.

## Architecture: Pipeline protocol

`pipeline.py` defines the `Pipeline` protocol (`analyze`, `retrieve_chunks`, `get_status`, `cleanup`)
and the `RetrievedChunk` dataclass. `rag_service.py`'s `RAGService` is a thin orchestrator that owns
the OpenAI chat call and prompt construction, and delegates all indexing/retrieval to whichever
`Pipeline` implementation it's given. Two implementations exist:

- **`default_pipeline.py` (`DefaultPipeline`)** — fetches all repo files upfront, chunks and embeds
  everything, persists to ChromaDB. Used by default.
- **`lazy_pipeline.py` (`LazyPipeline`)** — builds only a lightweight `MetadataIndex` at analyze-time
  (first N lines per file, extracted symbols). On each query, `FileSelector` scores candidate files by
  keyword/symbol match, `LazyFileParser` fetches/caches their shallow content (LRU, capacity 100), and
  chunks are embedded and cosine-scored on the fly. No ChromaDB — everything is in-memory per query.
  Enabled via `USE_LAZY_PIPELINE=1` (selected in `rag_service._create_pipeline`).

When adding a new pipeline variant, implement the `Pipeline` protocol and wire it into
`_create_pipeline()` — don't change `RAGService`'s public surface.

Both pipelines report progress via a `_status` dict (`stage`, `message`, `counters`, `durations`)
returned by `get_status()`; this is what powers indexing-progress UI in the frontend.

## Key modules

- **`github_parser.py`** — `get_repo_files()` pulls file contents via PyGithub; `chunk_repository_files()`
  splits them into `textual_chunks` / `code_chunks` using LangChain's `RecursiveCharacterTextSplitter`
  (language-aware for code via `Language` enum, token-counted via `tiktoken`). Chunk size is volume-aware
  (`analyze_repository_volume`): smaller repos get bigger chunks, larger repos get smaller ones.
- **`embedding.py`** — `embed_textual_metadata()` (OpenAI `text-embedding-ada-002`, 1536-dim) and
  `generate_code_embedding()` (CodeBERT via `transformers`, 768-dim). Both fall back to a deterministic
  local hash-based embedding (`_fallback_embed`) when the real model/API is unavailable — dimension is
  passed through as `target_dim` so fallback vectors match whatever dimension a collection was created with.
  `DISABLE_HF=1` skips loading CodeBERT entirely (used in tests/offline mode).
- **`chromaDB_setup.py`** — creates/reuses persistent Chroma collections. Collection names are suffixed
  with embedding dimension (e.g. `text_collection_d1536`) so a dimension change (e.g. switching embedding
  fallback on/off) creates a fresh collection instead of erroring on a mismatch. Telemetry is
  force-disabled (`_disable_chroma_telemetry`) to silence noisy PostHog errors from some chromadb versions.
  IDs are stable hashes of `file_name:chunk_index` (mmh3) so re-indexing the same repo doesn't duplicate rows.
- **`metadata_index.py` / `file_selector.py` / `lazy_parser.py`** — lazy-pipeline-only. Regex-based symbol
  extraction (no real AST parsing) and keyword/path-hint scoring for candidate file selection.

## Query routing

`DefaultPipeline._classify_query()` buckets a question into `text`, `code`, or `both` based on keyword
lists, querying the corresponding Chroma collection(s). In `both` mode, textual and code scores are each
normalized to [0,1] and combined 50/50 (`weights`) before re-ranking. `LazyPipeline` doesn't classify —
it scores every candidate file by cosine similarity regardless of type, then splits results into
text/code buckets by file extension only for display.

## Gotchas

- `src/models.py` (Pydantic request/response models) is **not imported anywhere** in the current backend
  — `backend/main.py` defines its own `IndexRequest`/`QueryResponse`/etc. inline. Treat `models.py` as
  legacy/unused; don't assume it reflects the live API contract.
- Every external call (GitHub API, OpenAI, CodeBERT/HF) has a synthetic/local fallback so the whole
  pipeline runs offline. When testing changes here, set `DISABLE_HF=1` and unset `OPENAI_API_KEY` /
  `GITHUB_API_KEY` to exercise the degraded paths deterministically (see `tests/test_rag_service_integration.py`).
- `CHROMA_PERSIST_DIR` controls where Chroma writes data (defaults to `./data/chroma`); tests override
  it per-test via `monkeypatch` + `tmp_path` to avoid crosstalk.
