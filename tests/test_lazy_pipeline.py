"""Unit tests for LazyPipeline in isolation (previously 0% coverage).

These tests run offline using the same fallbacks as the integration tests.
LazyPipeline never touches ChromaDB — it uses MetadataIndex + on-demand embedding.
"""
import asyncio

import pytest


def _force_offline(monkeypatch):
    monkeypatch.setenv("DISABLE_HF", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import src.embedding
    monkeypatch.setattr(src.embedding, "_openai_client", None, raising=False)


def test_analyze_initializes_lazy_components(monkeypatch):
    _force_offline(monkeypatch)
    from src.lazy_pipeline import LazyPipeline

    pipeline = LazyPipeline()
    assert pipeline.metadata is None
    assert pipeline.file_selector is None
    assert pipeline.lazy_parser is None

    asyncio.run(pipeline.analyze("example/repo"))

    assert pipeline.metadata is not None
    assert pipeline.file_selector is not None
    assert pipeline.lazy_parser is not None
    pipeline.cleanup()


def test_analyze_does_not_create_chromadb(monkeypatch):
    _force_offline(monkeypatch)
    from src.lazy_pipeline import LazyPipeline

    pipeline = LazyPipeline()
    asyncio.run(pipeline.analyze("example/repo"))
    # LazyPipeline has no collections attribute — getattr should return None
    assert getattr(pipeline, "collections", None) is None
    pipeline.cleanup()


def test_analyze_status_progresses_to_done(monkeypatch):
    _force_offline(monkeypatch)
    from src.lazy_pipeline import LazyPipeline

    pipeline = LazyPipeline()
    asyncio.run(pipeline.analyze("example/repo"))

    status = pipeline.get_status()
    assert status["stage"] == "done"
    assert "metadata" in status.get("durations", {})
    pipeline.cleanup()


def test_retrieve_chunks_returns_retrieved_chunk_objects(monkeypatch):
    _force_offline(monkeypatch)
    from src.lazy_pipeline import LazyPipeline
    from src.pipeline import RetrievedChunk

    pipeline = LazyPipeline()
    asyncio.run(pipeline.analyze("example/repo"))
    chunks = asyncio.run(pipeline.retrieve_chunks("what does this repo do?", top_k=2))

    assert isinstance(chunks, list)
    for chunk in chunks:
        assert isinstance(chunk, RetrievedChunk)
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.score, float)
        assert "content_type" in chunk.metadata
        assert chunk.metadata["content_type"] in ("text", "code")
    pipeline.cleanup()


def test_retrieve_chunks_without_analyze_raises(monkeypatch):
    _force_offline(monkeypatch)
    from src.lazy_pipeline import LazyPipeline

    pipeline = LazyPipeline()
    with pytest.raises(ValueError, match="not initialized"):
        asyncio.run(pipeline.retrieve_chunks("any question", top_k=3))
    pipeline.cleanup()
