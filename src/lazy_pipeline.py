from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedding import embed_textual_metadata
from .file_selector import FileSelector
from .lazy_parser import LazyFileParser
from .metadata_index import MetadataIndex
from .pipeline import RetrievedChunk

logger = logging.getLogger(__name__)


class LazyPipeline:
    """Lightweight pipeline: build metadata index only, parse and score files on demand per query."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metadata: Optional[MetadataIndex] = None
        self.file_selector: Optional[FileSelector] = None
        self.lazy_parser: Optional[LazyFileParser] = None
        self._status: Dict[str, Any] = {
            "stage": "idle",
            "message": "",
            "counters": {},
            "started_at": None,
            "updated_at": None,
            "durations": {},
        }

    # ------------------------------------------------------------------
    # Pipeline protocol implementation
    # ------------------------------------------------------------------

    async def analyze(self, repo_path: str) -> None:
        self._status["started_at"] = time.time()
        self._update_status("metadata", "Building metadata index")
        t_meta = time.perf_counter()
        metadata = MetadataIndex(repo_path, head_lines=20).build()
        dur_meta = time.perf_counter() - t_meta
        self._status.setdefault("durations", {})["metadata"] = dur_meta
        self._update_status(
            "metadata",
            f"Metadata index ready in {dur_meta:.1f}s",
            files=len(metadata.by_path),
        )
        self.metadata = metadata
        self.file_selector = FileSelector(metadata)
        self.lazy_parser = LazyFileParser(metadata, cache_size=100)
        total = time.time() - (self._status.get("started_at") or time.time())
        self._update_status("done", f"Analysis complete in {total:.1f}s")

    async def retrieve_chunks(self, question: str, top_k: int) -> List[RetrievedChunk]:
        if not (self.metadata and self.file_selector and self.lazy_parser):
            raise ValueError("Lazy pipeline is not initialized. Analyze repository first.")

        candidates = self.file_selector.select_files(question, max_files=max(20, top_k * 4))
        parsed = self.lazy_parser.parse_files(candidates)
        q_emb = await asyncio.get_event_loop().run_in_executor(
            self.executor, embed_textual_metadata, question
        )

        def cosine(a: List[float], b: List[float]) -> float:
            va = np.array(a, dtype=float)
            vb = np.array(b, dtype=float)
            if va.size == 0 or vb.size == 0:
                return 0.0
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

        scored_text: List[Tuple[float, str, Dict[str, Any]]] = []
        scored_code: List[Tuple[float, str, Dict[str, Any]]] = []

        for f in parsed:
            content = f.get("content", "")
            emb = await asyncio.get_event_loop().run_in_executor(
                self.executor, embed_textual_metadata, content
            )
            sim = cosine(q_emb, emb)
            path = f.get("path", "")
            is_text = path.lower().endswith((".md", ".txt", ".rst", ".adoc"))
            meta: Dict[str, Any] = {
                "file_name": path,
                "content_type": "text" if is_text else "code",
            }
            if is_text:
                scored_text.append((sim, content, meta))
            else:
                scored_code.append((sim, content, meta))

        top_text = sorted(scored_text, key=lambda x: x[0], reverse=True)[:top_k]
        top_code = sorted(scored_code, key=lambda x: x[0], reverse=True)[:top_k]

        chunks: List[RetrievedChunk] = []
        for score, text, meta in top_text:
            chunks.append(RetrievedChunk(content=text, score=float(score), metadata=meta))
        for score, text, meta in top_code:
            chunks.append(RetrievedChunk(content=text, score=float(score), metadata=meta))
        return chunks

    def get_status(self) -> Dict[str, Any]:
        return dict(self._status)

    def cleanup(self) -> None:
        self.executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_status(self, stage: str, message: str = "", **counters: Any) -> None:
        now = time.time()
        self._status.update({"stage": stage, "message": message, "updated_at": now})
        if counters:
            c = self._status.get("counters", {})
            c.update(counters)
            self._status["counters"] = c
        if message:
            logger.info(f"[{stage}] {message} | counters={self._status.get('counters', {})}")
