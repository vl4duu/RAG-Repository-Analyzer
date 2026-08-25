# tests/ — pytest suite

Tests `src/` (pipelines, `RAGService`) directly, with no live server and no real external calls.

## Offline-mode convention

Every test that exercises indexing/querying calls a `_force_offline(monkeypatch)` helper (repeated per
file, not shared) that:
- sets `DISABLE_HF=1` so CodeBERT is never loaded,
- deletes `OPENAI_API_KEY` and monkeypatches `_openai_client` to `None` in both `src.embedding` and
  `src.rag_service`,

which routes everything through the deterministic local fallbacks (synthetic repo content, hash-based
embeddings — see `src/CLAUDE.md`). Follow this pattern for any new pipeline/service test rather than
mocking GitHub/OpenAI at the network level.

Each test also uses an `isolated_chroma_dir` fixture (`tmp_path` + `monkeypatch.setenv("CHROMA_PERSIST_DIR", ...)`)
so Chroma state doesn't leak between tests or runs — always use it (or an equivalent) for any test that
calls `DefaultPipeline.analyze()`.

`test_openai_mock.py` is the exception: it mocks the `openai` module itself (via `unittest.mock` +
`sys.modules` patch) to assert the OpenAI code path is called correctly, rather than exercising the
fallback.

## Running

```bash
pytest tests/
```

## Not part of this suite

`test_api.py` and `test_analyze_and_query.py` live at the repo root, not here — they're manual
`requests`-based smoke scripts that expect a running server on `localhost:8000`, not pytest tests. Don't
add them to CI as-is.
