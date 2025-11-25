import asyncio
import os
from typing import Set

import pytest


# Integration tests for the core RAGService without hitting external services.
# These tests rely on the offline/degraded behavior implemented in:
# - src/github_parser.get_repo_files (returns synthetic repo when GitHub unavailable)
# - src/embedding (falls back to deterministic local embeddings)


def _force_offline(monkeypatch):
    """Force the system into offline-friendly mode for deterministic tests."""
    # Ensure heavy HF models are not loaded
    monkeypatch.setenv("DISABLE_HF", "1")
    # Ensure OpenAI key is absent so embeddings fallback is used
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Force ChatCompletion path to be disabled so we do not attempt network calls
    try:
        import openai  # type: ignore
        monkeypatch.setattr(openai, "ChatCompletion", None, raising=False)
    except Exception:
        # If openai is not importable or already disabled, that's fine
        pass


@pytest.fixture()
def isolated_chroma_dir(tmp_path, monkeypatch):
    """
    Provide an isolated Chroma persistence directory per test to avoid
    crosstalk between tests (and across runs).
    """
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    return persist_dir


def test_analyze_and_query_flow_working(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)

    from src.rag_service import RAGService

    service = RAGService()

    # Analyze a synthetic repository (any non-"nonexistent" path)
    repo = "example/repo"
    result = asyncio.run(service.analyze_repository(repo))

    assert result["status"] == "success"
    assert service.is_ready is True
    assert service.collections is not None
    assert set(service.collections.keys()) == {"textual_collection", "code_collection"}

    # Now query the repository
    question = "What does this repository contain?"
    qr = asyncio.run(service.query_repository(question))

    assert isinstance(qr, dict)
    assert isinstance(qr.get("answer"), str) and len(qr["answer"]) > 0
    assert isinstance(qr.get("sources"), list) and len(qr["sources"]) > 0

    # Validate source schema for the first item
    src0 = qr["sources"][0]
    assert set(["file_name", "content_type", "score", "content"]) <= set(src0.keys())
    assert src0["content_type"] in ("text", "code")
    assert isinstance(src0["score"], float)
    assert isinstance(src0["content"], str)

    service.cleanup()


def test_query_without_analyze_raises(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.rag_service import RAGService

    service = RAGService()
    with pytest.raises(ValueError):
        asyncio.run(service.query_repository("Hello?"))

    service.cleanup()


def test_invalid_repository_raises(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.rag_service import RAGService

    service = RAGService()
    # RAGService is designed to raise for clearly invalid repo names containing "nonexistent"
    with pytest.raises(ValueError):
        asyncio.run(service.analyze_repository("nonexistent/repo"))

    service.cleanup()


def test_routing_returns_both_text_and_code(isolated_chroma_dir, monkeypatch):
    _force_offline(monkeypatch)
    from src.rag_service import RAGService

    service = RAGService()
    asyncio.run(service.analyze_repository("example/repo"))

    # This contains cues for both text ("readme") and code ("function")
    question = "What does the README say about the function hello?"
    qr = asyncio.run(service.query_repository(question))

    content_types: Set[str] = {src.get("content_type") for src in qr.get("sources", [])}
    # Expect both text and code sources to be present in the merged result
    assert "text" in content_types
    assert "code" in content_types

    service.cleanup()
