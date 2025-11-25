"""
FastAPI backend for RAG Repository Analyzer.

Provides endpoints for indexing GitHub repositories and querying them.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Add src directory to path (project root) so `src` package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Use the consolidated RAG service implementation
from src.rag_service import RAGService

load_dotenv()

# Basic logging config (respect LOG_LEVEL env). Uvicorn will also manage logs, but this ensures
# our modules use a consistent formatter when imported outside uvicorn too.
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Repository Analyzer API",
    description="API for analyzing GitHub repositories using RAG",
    version="1.0.0"
)

# Configure CORS (dev-friendly defaults + env override)
frontend_origins_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN")
allow_all = os.getenv("API_ALLOW_ALL_ORIGINS", "false").lower() in {"1", "true", "yes"}

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Regex to allow common private-network origins by default (http[s]://<ip>:<port>)
private_net_origin_regex = r"^https?://(localhost|127\\.0\\.0\\.1|10(?:\\.\\d{1,3}){3}|192\\.168(?:\\.\\d{1,3}){2}|172\\.(?:1[6-9]|2\\d|3[0-1])(?:\\.\\d{1,3}){2})(?::\\d+)?$"

if allow_all:
    cors_kwargs = dict(
        allow_origins=["*"],
        # credentials cannot be used with wildcard origins
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    env_origins = (
        [o.strip() for o in frontend_origins_env.split(",") if o.strip()]
        if frontend_origins_env
        else []
    )
    cors_kwargs = dict(
        allow_origins=list({*default_origins, *env_origins}),
        allow_origin_regex=private_net_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(CORSMiddleware, **cors_kwargs)

# In-memory storage for indexed repositories and active RAG services
# `indexed_repositories` can carry lightweight metadata (e.g., volume_info) if available.
indexed_repositories: Dict[str, Dict] = {}
rag_instances: Dict[str, RAGService] = {}


# Request/Response Models
class IndexRequest(BaseModel):
    repo_path: str
    use_langchain: bool = True


class IndexResponse(BaseModel):
    status: str
    message: str
    repo_path: str
    volume_info: Optional[Dict] = None


class QueryRequest(BaseModel):
    repo_path: str
    question: str
    use_both_collections: bool = True


class QueryResponse(BaseModel):
    answer: str
    textual_sources: List[Dict] = []
    code_sources: List[Dict] = []
    all_sources: List[Dict] = []


class RepositoryStatus(BaseModel):
    repo_path: str
    indexed: bool
    volume_info: Optional[Dict] = None


# Endpoints
@app.get("/")
async def root():
    """Root endpoint.

    In production, we prefer showing the frontend application rather than the raw API JSON.
    If FRONTEND_URL (or the first value from FRONTEND_ORIGINS) is set, redirect there.
    Otherwise, keep returning a small HTML landing page with links.
    """
    frontend_url = os.getenv("FRONTEND_URL")
    if not frontend_url:
        origins = os.getenv("FRONTEND_ORIGINS", "").split(",")
        origins = [o.strip() for o in origins if o.strip()]
        if origins:
            frontend_url = origins[0]

    if frontend_url:
        # Use 307 to preserve method should someone POST to root by mistake
        return RedirectResponse(url=frontend_url, status_code=307)

    # Fallback lightweight HTML page for local/dev without configured frontend URL
    html = """
    <!doctype html>
    <html lang=\"en\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>RAG Repository Analyzer API</title>
        <style>
          body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 2rem; }
          code { background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 4px; }
          ul { line-height: 1.8; }
        </style>
      </head>
      <body>
        <h1>RAG Repository Analyzer API</h1>
        <p>Set <code>FRONTEND_URL</code> (or <code>FRONTEND_ORIGINS</code>) to redirect this root path to your frontend.</p>
        <h2>Endpoints</h2>
        <ul>
          <li><code>/index</code></li>
          <li><code>/query</code></li>
          <li><code>/repositories</code></li>
          <li><code>/status/{repo_path}</code></li>
          <li><code>/health</code></li>
        </ul>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint for Render and uptime monitors."""
    return JSONResponse({"status": "ok"})


def _resolve_frontend_url() -> Optional[str]:
    """Resolve the configured frontend URL from env vars.

    Priority: FRONTEND_URL, then first entry in FRONTEND_ORIGINS.
    Returns None if neither is configured.
    """
    frontend_url = os.getenv("FRONTEND_URL")
    if not frontend_url:
        origins = os.getenv("FRONTEND_ORIGINS", "").split(",")
        origins = [o.strip() for o in origins if o.strip()]
        if origins:
            frontend_url = origins[0]
    return frontend_url or None


@app.exception_handler(StarletteHTTPException)
async def not_found_redirect_handler(request: Request, exc: StarletteHTTPException):
    """Redirect unknown GET/HEAD paths to the frontend when configured.

    This helps when someone navigates to a deep link on the backend domain or
    when uptime monitors probe random paths; instead of 404, we forward them to
    the actual frontend service. All known API routes continue to be handled by
    FastAPI. For non-GET/HEAD methods or when no frontend is configured, fall
    back to FastAPI's default handler.
    """
    try:
        if exc.status_code == 404 and request.method in {"GET", "HEAD"}:
            frontend_url = _resolve_frontend_url()
            if frontend_url:
                # Preserve the path for potential deep-linking on the frontend
                target = frontend_url.rstrip("/") + request.url.path
                # Include query string if present
                if request.url.query:
                    target += f"?{request.url.query}"
                return RedirectResponse(url=target, status_code=307)
    except Exception:
        # If anything goes wrong, use default behavior
        pass
    return await http_exception_handler(request, exc)


@app.post("/index", response_model=IndexResponse)
async def index_repo(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index a GitHub repository.
    
    Args:
        request: IndexRequest with repo_path and use_langchain flag
        
    Returns:
        IndexResponse with status and volume information
    """
    try:
        repo_path = request.repo_path
        
        # Check if already indexed
        if repo_path in indexed_repositories:
            return IndexResponse(
                status="already_indexed",
                message=f"Repository {repo_path} is already indexed",
                repo_path=repo_path,
                volume_info=indexed_repositories[repo_path].get("volume_info")
            )
        
        logger.info(f"Index request received for repo: {repo_path}")

        # Create and run the RAG service analysis for this repository
        service = RAGService()
        t0 = time.perf_counter()
        result = await service.analyze_repository(repo_path)
        elapsed = time.perf_counter() - t0
        logger.info(f"Indexing completed for {repo_path} in {elapsed:.1f}s")

        # Store service instance in memory for subsequent queries
        rag_instances[repo_path] = service

        # Optionally keep lightweight metadata placeholder (no detailed volume info at this time)
        indexed_repositories[repo_path] = {
            "volume_info": None
        }

        return IndexResponse(
            status=result.get("status", "success"),
            message=result.get("message", f"Successfully indexed repository {repo_path}"),
            repo_path=repo_path,
            volume_info=indexed_repositories[repo_path].get("volume_info")
        )
        
    except Exception as e:
        logger.exception(f"Error indexing repository: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index repository: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    """
    Query an indexed repository.
    
    Args:
        request: QueryRequest with repo_path, question, and options
        
    Returns:
        QueryResponse with answer and sources
    """
    try:
        repo_path = request.repo_path
        
        # Check if repository is indexed (i.e., has an active service)
        if repo_path not in rag_instances:
            raise HTTPException(
                status_code=404,
                detail=f"Repository {repo_path} is not indexed. Please index it first."
            )

        # Get RAG service and execute query
        service = rag_instances[repo_path]
        result = await service.query_repository(
            question=request.question,
            top_k=3
        )

        # Map to existing response model. Put all sources into `all_sources` for frontend summarizer.
        return QueryResponse(
            answer=result.get("answer", ""),
            textual_sources=[],
            code_sources=[],
            all_sources=result.get("sources", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error querying repository: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query repository: {str(e)}"
        )


@app.get("/repositories")
async def list_repositories():
    """
    List all indexed repositories.
    
    Returns:
        List of indexed repository paths with their status
    """
    repositories = []
    for repo_path, data in indexed_repositories.items():
        repositories.append({
            "repo_path": repo_path,
            "indexed": True,
            "volume_info": data.get("volume_info")
        })
    
    return {"repositories": repositories}


@app.get("/status/{repo_path:path}", response_model=RepositoryStatus)
async def get_repository_status(repo_path: str):
    """
    Get the status of a repository.
    
    Args:
        repo_path: GitHub repository path (e.g., "username/repo")
        
    Returns:
        RepositoryStatus with indexing information
    """
    if repo_path in rag_instances:
        return RepositoryStatus(
            repo_path=repo_path,
            indexed=True,
            volume_info=indexed_repositories.get(repo_path, {}).get("volume_info")
        )
    else:
        return RepositoryStatus(
            repo_path=repo_path,
            indexed=False,
            volume_info=None
        )


@app.delete("/repository/{repo_path:path}")
async def delete_repository(repo_path: str):
    """
    Remove a repository from the indexed collection.
    
    Args:
        repo_path: GitHub repository path (e.g., "username/repo")
        
    Returns:
        Success message
    """
    if repo_path in indexed_repositories or repo_path in rag_instances:
        # Clean up active service if present
        service = rag_instances.pop(repo_path, None)
        if service is not None:
            try:
                service.cleanup()
            except Exception:
                pass
        indexed_repositories.pop(repo_path, None)
        return {"message": f"Repository {repo_path} removed successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_path} not found"
        )


@app.post("/clear-memory/{repo_path:path}")
async def clear_conversation_memory(repo_path: str):
    """
    Clear conversation memory for a repository.
    
    Args:
        repo_path: GitHub repository path (e.g., "username/repo")
        
    Returns:
        Success message
    """
    if repo_path in rag_instances:
        # RAGService does not maintain conversation memory; act as a no-op for compatibility
        return {"message": f"No conversation memory to clear for {repo_path}"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_path} not found"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


