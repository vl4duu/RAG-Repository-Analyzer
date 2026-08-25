# backend/ — FastAPI app

Single-file FastAPI app (`main.py`) that exposes the REST API described in the root `CLAUDE.md`. It's
a thin HTTP layer over `src.rag_service.RAGService` — request handling, CORS, and static frontend
serving live here; indexing/retrieval logic does not.

## State model

`indexed_repositories: Dict[str, Dict]` and `rag_instances: Dict[str, RAGService]` are plain in-process
dicts — there is no database. State is lost on restart, and nothing is shared across worker processes
(don't run this with multiple uvicorn workers unless that in-memory state is made external first). Each
`POST /index` creates a fresh `RAGService(_create_pipeline())`; `_create_pipeline()` reads
`USE_LAZY_PIPELINE` at call time, so different repos indexed in the same process could in principle use
different pipelines if the env var changes mid-run.

## CORS and origin resolution

CORS is intentionally layered and stricter in deployed environments than locally — see the top of
`main.py`:
1. `API_ALLOW_ALL_ORIGINS=true` → wildcard, no credentials.
2. `FRONTEND_ORIGINS` / `FRONTEND_URL` set → strict allow-list, no implicit localhost.
3. Nothing configured + `RENDER`/`RENDER_EXTERNAL_URL` set (i.e. deployed) → allow **nothing** (fails
   closed) and logs a warning.
4. Nothing configured + not on Render → dev-friendly: localhost + private-network IP ranges via regex.

If you change CORS behavior, keep this closed-by-default-in-prod property — it exists specifically so a
misconfigured Render deploy doesn't silently become world-readable.

## Root path / static frontend / 404 handling

This app can optionally serve the Next.js static export (`frontend/out`, overridable via
`FRONTEND_STATIC_DIR`) directly from FastAPI (see `render.yaml`/`build.sh` — the frontend is normally a
separate Render static site, but `main.py` supports serving it inline too). Because of that, there's
nontrivial logic to avoid redirect loops and 404 amplification:

- `/` serves `frontend/out/index.html` if present, else redirects to `FRONTEND_URL`, else shows a
  fallback landing page. `_same_origin_url` guards against `FRONTEND_URL` accidentally pointing back at
  this same backend (which would loop).
- The global `StarletteHTTPException` handler rewrites unmatched GET/HEAD 404s into either a static
  file serve or a redirect to the frontend — this is an SPA-style catch-all, not a bug. Non-GET/HEAD and
  overly long paths (>4096 chars) fall through to a normal 404 to avoid abuse.
- `/undefined` and `/undefined/*` return an explicit 400 explaining the likely cause: the frontend was
  built without `NEXT_PUBLIC_API_URL` set, so it's calling `<host>/undefined/...`. If you see this in
  logs, it's a frontend build config problem, not a backend bug.

## Adding an endpoint

Request/response models are defined inline in `main.py` (not `src/models.py`, which is unused — see
`src/CLAUDE.md`). Follow the existing pattern: Pydantic model per request/response, `try/except` wrapping
the handler body that re-raises `HTTPException` as-is and converts anything else to a 500 with
`logger.exception`.

## Running / testing

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires `OPENAI_API_KEY` / `GITHUB_API_KEY` in `.env` for real embeddings/GitHub access — both are
optional, since `src/` falls back to synthetic/local behavior without them (useful for local dev without
API keys). `test_api.py` and `test_analyze_and_query.py` at the repo root are manual smoke-test scripts
that hit a **running** server (`requests`-based, not pytest fixtures) — start the server first. The
pytest suite in `tests/` exercises `RAGService`/pipelines directly without a live server.
