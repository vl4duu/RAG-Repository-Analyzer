"""
FastAPI backend for RAG Repository Analyzer.

Provides endpoints for indexing GitHub repositories and querying them.
"""

import os
import sys
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.volume_aware_indexer import index_repository
from src.rag_query import RepositoryRAG, format_response

load_dotenv()

app = FastAPI(
    title="RAG Repository Analyzer API",
    description="API for analyzing GitHub repositories using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for indexed repositories
indexed_repositories: Dict[str, Dict] = {}
rag_instances: Dict[str, RepositoryRAG] = {}


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
    """Root endpoint."""
    return {
        "message": "RAG Repository Analyzer API",
        "version": "1.0.0",
        "endpoints": [
            "/index",
            "/query",
            "/repositories",
            "/status/{repo_path}"
        ]
    }


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
        
        print(f"Indexing repository: {repo_path}")
        
        # Index the repository
        index_result = index_repository(
            repo_path=repo_path,
            use_langchain=request.use_langchain
        )
        
        # Store the collections and create RAG instance
        indexed_repositories[repo_path] = {
            "collections": index_result['collections'],
            "volume_info": index_result.get('volume_info')
        }
        
        # Create RAG instance
        rag_instances[repo_path] = RepositoryRAG(
            collections=index_result['collections'],
            use_conversation=True
        )
        
        return IndexResponse(
            status="success",
            message=f"Successfully indexed repository {repo_path}",
            repo_path=repo_path,
            volume_info=index_result.get('volume_info')
        )
        
    except Exception as e:
        print(f"Error indexing repository: {e}")
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
        
        # Check if repository is indexed
        if repo_path not in indexed_repositories:
            raise HTTPException(
                status_code=404,
                detail=f"Repository {repo_path} is not indexed. Please index it first."
            )
        
        # Get RAG instance
        rag = rag_instances.get(repo_path)
        if not rag:
            # Recreate RAG instance if missing
            rag = RepositoryRAG(
                collections=indexed_repositories[repo_path]['collections'],
                use_conversation=True
            )
            rag_instances[repo_path] = rag
        
        # Query the repository
        result = rag.query(
            question=request.question,
            use_both_collections=request.use_both_collections
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            textual_sources=result.get("textual_sources", []),
            code_sources=result.get("code_sources", []),
            all_sources=result.get("all_sources", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error querying repository: {e}")
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
    if repo_path in indexed_repositories:
        return RepositoryStatus(
            repo_path=repo_path,
            indexed=True,
            volume_info=indexed_repositories[repo_path].get("volume_info")
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
    if repo_path in indexed_repositories:
        del indexed_repositories[repo_path]
        if repo_path in rag_instances:
            del rag_instances[repo_path]
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
        rag_instances[repo_path].clear_memory()
        return {"message": f"Conversation memory cleared for {repo_path}"}
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


