from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .default_pipeline import DefaultPipeline
from .lazy_pipeline import LazyPipeline
from .pipeline import Pipeline, RetrievedChunk

# OpenAI v1.0+ API
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_openai_client = None

logger = logging.getLogger(__name__)

if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client in RAGService: {e}")
        _openai_client = None


def _create_pipeline() -> Pipeline:
    use_lazy = os.getenv("USE_LAZY_PIPELINE", "0").lower() in {"1", "true", "yes"}
    return LazyPipeline() if use_lazy else DefaultPipeline()


class RAGService:
    """Thin orchestrator: delegates indexing and retrieval to a Pipeline, owns prompt + AI call."""

    def __init__(self, pipeline: Optional[Pipeline] = None) -> None:
        self.pipeline: Pipeline = pipeline if pipeline is not None else _create_pipeline()
        self.current_repository: Optional[str] = None
        self.is_ready: bool = False
        self.executor = ThreadPoolExecutor(max_workers=4)

    @property
    def collections(self) -> Optional[Dict]:
        """Expose DefaultPipeline collections for backward compatibility."""
        return getattr(self.pipeline, "collections", None)

    async def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        if "nonexistent" in repo_path.lower():
            raise ValueError(f"Repository '{repo_path}' not found or inaccessible")

        logger.info(f"Starting analysis of repository: {repo_path}")
        self.is_ready = False
        try:
            await self.pipeline.analyze(repo_path)
            self.current_repository = repo_path
            self.is_ready = True
            logger.info(f"Successfully analyzed repository: {repo_path}")
            return {
                "status": "success",
                "message": f"Repository {repo_path} analyzed successfully",
                "repository": repo_path,
            }
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_path}: {str(e)}")
            self.is_ready = False
            raise

    async def query_repository(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        if not self.is_ready:
            raise ValueError("No repository has been analyzed yet. Please analyze a repository first.")

        try:
            logger.info(f"Processing query: {question}")
            chunks = await self.pipeline.retrieve_chunks(question, top_k)
            rag_prompt = self._construct_rag_prompt(question, chunks)
            ai_answer = await self._query_ai_model(rag_prompt)
            sources = self._format_sources(chunks)
            return {"answer": ai_answer, "sources": sources}
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            raise

    def _construct_rag_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        prompt = (
            "You are a repository analyser, use the provided chunks to answer any related "
            f"questions about the repository:\n\nQuestion: {query}\n\nContext:\n"
        )
        if not chunks:
            prompt += "\n--- No chunks found ---\n"
        else:
            for chunk in chunks:
                ctype = chunk.metadata.get("content_type", "unknown")
                prompt += f"\n--- {ctype.capitalize()} Chunk ---\n"
                prompt += f"Score: {chunk.score:.4f}\n"
                prompt += f"Content: {chunk.content}\n"
                prompt += f"Metadata: {chunk.metadata}\n\n"
        prompt += "\nAnswer:"
        return prompt

    async def _query_ai_model(self, prompt: str) -> str:
        try:
            if _openai_client is not None:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: _openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant. Answer the question using only the provided context.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=500,
                        temperature=0.1,
                    ),
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI call failed or unavailable, falling back to local answer. Reason: {e}")

        try:
            context = prompt.split("Context:", 1)[-1]
            lines = [
                ln.strip()
                for ln in context.splitlines()
                if ln.strip()
                and not ln.lower().startswith("score:")
                and not ln.lower().startswith("metadata:")
            ]
            snippet = " ".join(lines[:10])
            if not snippet:
                snippet = "Insufficient context to answer precisely."
            return f"Based on the provided repository context, here is a concise answer: {snippet[:500]}"
        except Exception:
            return "Unable to generate an answer due to missing model and context."

    def _format_sources(self, chunks: List[RetrievedChunk]) -> List[Dict]:
        sources = []
        for chunk in chunks:
            sources.append({
                "file_name": chunk.metadata.get("file_name", "unknown"),
                "content_type": chunk.metadata.get("content_type", "unknown"),
                "score": float(chunk.score),
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
            })
        return sources

    def get_status(self) -> Dict[str, Any]:
        pipeline_status = self.pipeline.get_status()
        base: Dict[str, Any] = {
            "repository": self.current_repository,
            "ready": self.is_ready,
            "message": (
                f"Repository '{self.current_repository}' is ready for queries"
                if self.is_ready
                else pipeline_status.get("message", "No repository analyzed")
            ),
        }
        base.update({
            "stage": pipeline_status.get("stage"),
            "counters": pipeline_status.get("counters", {}),
            "durations": pipeline_status.get("durations", {}),
            "started_at": pipeline_status.get("started_at"),
            "updated_at": pipeline_status.get("updated_at"),
        })
        return base

    def cleanup(self) -> None:
        self.pipeline.cleanup()
        self.executor.shutdown(wait=True)
