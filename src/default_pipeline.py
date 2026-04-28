from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chromaDB_setup import setup_chroma_collections
from .embedding import embed_textual_metadata, generate_code_embedding
from .github_parser import chunk_repository_files, get_repo_files
from .pipeline import RetrievedChunk

logger = logging.getLogger(__name__)


class DefaultPipeline:
    """Full-indexing pipeline: fetch → chunk → embed → persist to ChromaDB."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.collections: Optional[Dict] = None
        self.text_embedding_dim: Optional[int] = None
        self.code_embedding_dim: Optional[int] = None
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
        self.collections = None
        self.text_embedding_dim = None
        self.code_embedding_dim = None
        self._status["started_at"] = time.time()
        self._update_status("start", f"Begin analysis for {repo_path}")

        # Step 1: Fetch
        t_fetch = time.perf_counter()
        self._update_status("fetch", "Fetching repository files")
        repo_files = await asyncio.get_event_loop().run_in_executor(
            self.executor, get_repo_files, repo_path
        )
        dur_fetch = time.perf_counter() - t_fetch
        self._status.setdefault("durations", {})["fetch"] = dur_fetch
        self._update_status("fetch", f"Fetched {len(repo_files)} files in {dur_fetch:.1f}s", files=len(repo_files))

        if not repo_files:
            raise ValueError(f"No files found in repository {repo_path}")

        # Step 2: Chunk
        t_chunk = time.perf_counter()
        self._update_status("chunk", "Chunking repository files")
        chunked_docs = await asyncio.get_event_loop().run_in_executor(
            self.executor, chunk_repository_files, repo_files
        )
        dur_chunk = time.perf_counter() - t_chunk
        self._status.setdefault("durations", {})["chunk"] = dur_chunk
        self._update_status(
            "chunk",
            f"Chunked files in {dur_chunk:.1f}s",
            textual_chunks=len(chunked_docs.get("textual_chunks", [])),
            code_chunks=len(chunked_docs.get("code_chunks", [])),
        )

        # Step 3: Embed
        self._update_status("embed", "Generating embeddings")
        t_embed = time.perf_counter()
        embedded_chunks = await self._generate_embeddings(chunked_docs)
        dur_embed = time.perf_counter() - t_embed
        self._status.setdefault("durations", {})["embed"] = dur_embed
        self._update_status(
            "embed",
            f"Generated embeddings in {dur_embed:.1f}s",
            textual_embeddings=len(embedded_chunks.get("textual_embeddings", [])),
            code_embeddings=len(embedded_chunks.get("code_embeddings", [])),
        )

        # Step 4: Persist
        self._update_status("persist", "Setting up ChromaDB collections")
        t_db = time.perf_counter()
        self.collections = await asyncio.get_event_loop().run_in_executor(
            self.executor, setup_chroma_collections, chunked_docs, embedded_chunks
        )
        dur_db = time.perf_counter() - t_db
        self._status.setdefault("durations", {})["persist"] = dur_db
        self._update_status("persist", f"Chroma collections ready in {dur_db:.1f}s")

        try:
            tcol = self.collections.get("textual_collection") if self.collections else None
            ccol = self.collections.get("code_collection") if self.collections else None
            if tcol is not None:
                self.text_embedding_dim = self._infer_collection_dim(tcol)
            if ccol is not None:
                self.code_embedding_dim = self._infer_collection_dim(ccol)
        except Exception:
            pass

        total = time.time() - (self._status.get("started_at") or time.time())
        self._update_status("done", f"Analysis complete in {total:.1f}s")

    async def retrieve_chunks(self, question: str, top_k: int) -> List[RetrievedChunk]:
        route, weights = self._classify_query(question)

        textual_results = None
        code_results = None

        if route in ("text", "both"):
            target_dim = self.text_embedding_dim
            if target_dim is None and self.collections and self.collections.get("textual_collection") is not None:
                try:
                    target_dim = self._infer_collection_dim(self.collections["textual_collection"])
                    self.text_embedding_dim = target_dim
                except Exception:
                    target_dim = None
            textual_embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, embed_textual_metadata, question, target_dim
            )
            textual_results = self.collections["textual_collection"].query(
                query_embeddings=[textual_embedding],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"],
            )

        if route in ("code", "both"):
            target_dim = self.code_embedding_dim
            if target_dim is None and self.collections and self.collections.get("code_collection") is not None:
                try:
                    target_dim = self._infer_collection_dim(self.collections["code_collection"])
                    self.code_embedding_dim = target_dim
                except Exception:
                    target_dim = None
            code_embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, generate_code_embedding, question, target_dim
            )
            code_results = self.collections["code_collection"].query(
                query_embeddings=[code_embedding],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"],
            )

        top_textual = self._process_results(textual_results, top_k) if textual_results else []
        top_code = self._process_results(code_results, top_k) if code_results else []

        if route == "both":
            wt_text = weights.get("text", 0.5)
            wt_code = weights.get("code", 0.5)

            def normalize(chunks: List[Tuple]) -> List[Tuple]:
                if not chunks:
                    return []
                try:
                    raw_scores = [float(s) for s, _, _ in chunks]
                    max_s = max(raw_scores)
                    if max_s == 0:
                        return [(0.0, d, m) for (_, d, m) in chunks]
                    return [(float(s / max_s), d, m) for (s, d, m) in chunks]
                except (ValueError, TypeError, ZeroDivisionError):
                    return [(0.0, d, m) for (_, d, m) in chunks]

            ntext = normalize(top_textual)
            ncode = normalize(top_code)
            weighted_text = [(s * wt_text, d, m) for (s, d, m) in ntext]
            weighted_code = [(s * wt_code, d, m) for (s, d, m) in ncode]
            top_textual = sorted(weighted_text, key=lambda x: x[0], reverse=True)[:top_k]
            top_code = sorted(weighted_code, key=lambda x: x[0], reverse=True)[:top_k]

        chunks: List[RetrievedChunk] = []
        for score, text, meta in top_textual:
            chunks.append(RetrievedChunk(
                content=text,
                score=float(score),
                metadata={**meta, "content_type": meta.get("content_type", "text")},
            ))
        for score, text, meta in top_code:
            chunks.append(RetrievedChunk(
                content=text,
                score=float(score),
                metadata={**meta, "content_type": meta.get("content_type", "code")},
            ))
        return chunks

    def get_status(self) -> Dict[str, Any]:
        return dict(self._status)

    def cleanup(self) -> None:
        self.executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_collection_dim(self, collection) -> Optional[int]:
        try:
            name = getattr(collection, "name", "")
            if "_d" in name:
                try:
                    return int(name.rsplit("_d", 1)[-1])
                except Exception:
                    pass
            try:
                res = collection.get(limit=1, include=["embeddings"])
                embs = res.get("embeddings") if isinstance(res, dict) else None
                if embs and len(embs) > 0 and isinstance(embs[0], (list, tuple)):
                    return int(len(embs[0]))
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _update_status(self, stage: str, message: str = "", **counters: Any) -> None:
        now = time.time()
        self._status.update({"stage": stage, "message": message, "updated_at": now})
        if counters:
            c = self._status.get("counters", {})
            c.update(counters)
            self._status["counters"] = c
        if message:
            logger.info(f"[{stage}] {message} | counters={self._status.get('counters', {})}")

    async def _generate_embeddings(self, chunked_docs: Dict) -> Dict[str, Any]:
        textual_embeddings: List[List[float]] = []
        code_embeddings: List[List[float]] = []

        text_chunks = chunked_docs.get("textual_chunks", [])
        code_chunks = chunked_docs.get("code_chunks", [])

        for i, doc in enumerate(text_chunks, start=1):
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, embed_textual_metadata, doc["content"]
            )
            textual_embeddings.append(embedding)
            if i % 200 == 0 or i == len(text_chunks):
                self._update_status("embed", f"Text embeddings: {i}/{len(text_chunks)} done")

        for j, doc in enumerate(code_chunks, start=1):
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, generate_code_embedding, doc["content"], None
            )
            code_embeddings.append(embedding)
            if j % 200 == 0 or j == len(code_chunks):
                self._update_status("embed", f"Code embeddings: {j}/{len(code_chunks)} done")

        try:
            self.text_embedding_dim = int(len(textual_embeddings[0])) if textual_embeddings else self.text_embedding_dim
            self.code_embedding_dim = int(len(code_embeddings[0])) if code_embeddings else self.code_embedding_dim
        except Exception:
            pass

        return {
            "textual_embeddings": textual_embeddings,
            "code_embeddings": code_embeddings,
        }

    def _process_results(self, results: Optional[Dict], top_k: int) -> List[Tuple]:
        if not results or "distances" not in results or not results.get("distances"):
            logger.warning("'distances' key missing or empty in results")
            return []
        distances = np.array(results["distances"][0])
        scores = 1 - distances
        combined = list(zip(scores, results["documents"][0], results["metadatas"][0]))
        return sorted(combined, key=lambda x: x[0], reverse=True)[:top_k]

    def _classify_query(self, query: str) -> Tuple[str, Dict[str, float]]:
        q = query.lower()
        code_keywords = [
            "function", "class", "method", "variable", "error", "stack trace", "traceback",
            "api", "endpoint", "def ", "return ", "for (", "if (", "compile", "build", "test", "unit test",
        ]
        text_keywords = [
            "readme", "documentation", "docs", "license", "contributing", "overview", "about", "install",
        ]
        code_hits = any(k in q for k in code_keywords)
        text_hits = any(k in q for k in text_keywords)
        if code_hits and not text_hits:
            return "code", {"code": 1.0}
        if text_hits and not code_hits:
            return "text", {"text": 1.0}
        return "both", {"text": 0.5, "code": 0.5}
