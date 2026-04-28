"""Unit tests for DefaultPipeline in isolation.

These tests run offline (no GitHub API, no OpenAI) using the same degraded-mode
fallbacks as the integration tests: synthetic repo from github_parser and
hash-based embeddings from src.embedding.
"""
import asyncio

import pytest


def _force_offline(monkeypatch):
    monkeypatch.setenv("DISABLE_HF", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import src.embedding
    monkeypatch.setattr(src.embedding, "_openai_client", None, raising=False)


@pytest.fixture()
def isolated_chroma_dir(tmp_path, monkeypatch):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    return persist_dir


def test_analyze_sets_collections(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.default_pipeline import DefaultPipeline

    pipeline = DefaultPipeline()
    assert pipeline.collections is None

    asyncio.run(pipeline.analyze("example/repo"))

    assert pipeline.collections is not None
    assert set(pipeline.collections.keys()) == {"textual_collection", "code_collection"}
    pipeline.cleanup()


def test_analyze_status_progresses_to_done(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.default_pipeline import DefaultPipeline

    pipeline = DefaultPipeline()
    asyncio.run(pipeline.analyze("example/repo"))

    status = pipeline.get_status()
    assert status["stage"] == "done"
    durations = status.get("durations", {})
    for stage in ("fetch", "chunk", "embed", "persist"):
        assert stage in durations
    pipeline.cleanup()


def test_retrieve_chunks_returns_retrieved_chunk_objects(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.default_pipeline import DefaultPipeline
    from src.pipeline import RetrievedChunk

    pipeline = DefaultPipeline()
    asyncio.run(pipeline.analyze("example/repo"))
    chunks = asyncio.run(pipeline.retrieve_chunks("what does this repo do?", top_k=2))

    assert isinstance(chunks, list)
    for chunk in chunks:
        assert isinstance(chunk, RetrievedChunk)
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.score, float)
        assert "content_type" in chunk.metadata
    pipeline.cleanup()


def test_retrieve_chunks_routing_returns_both_types(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.default_pipeline import DefaultPipeline

    pipeline = DefaultPipeline()
    asyncio.run(pipeline.analyze("example/repo"))
    # Query with both text and code cues
    chunks = asyncio.run(pipeline.retrieve_chunks(
        "What does the README say about the function hello?", top_k=3
    ))
    content_types = {c.metadata.get("content_type") for c in chunks}
    assert "text" in content_types
    assert "code" in content_types
    pipeline.cleanup()


@pytest.mark.parametrize("query,expected_route", [
    ("show me the readme documentation", "text"),
    ("show me the function definition", "code"),
    ("what does this repository contain?", "both"),
])
def test_classify_query_routes_correctly(query, expected_route, monkeypatch):
    _force_offline(monkeypatch)
    from src.default_pipeline import DefaultPipeline

    pipeline = DefaultPipeline()
    route, _ = pipeline._classify_query(query)
    assert route == expected_route
