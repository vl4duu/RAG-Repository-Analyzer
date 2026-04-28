from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class RetrievedChunk:
    content: str
    score: float
    metadata: dict[str, Any]


@runtime_checkable
class Pipeline(Protocol):
    async def analyze(self, repo_path: str) -> None: ...
    async def retrieve_chunks(self, question: str, top_k: int) -> list[RetrievedChunk]: ...
    def get_status(self) -> dict: ...
    def cleanup(self) -> None: ...
