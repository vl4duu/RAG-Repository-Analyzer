import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import (
    AnalyzeRequest, AnalyzeResponse, QueryRequest, QueryResponse,
    AnalyzeAndQueryRequest, AnalyzeAndQueryResponse,
    HealthResponse, StatusResponse, ErrorResponse
)
from .rag_service import RAGService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/rag-api.log') if os.path.exists('/tmp') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global rag_service
    
    # Startup
    logger.info("Starting RAG Repository Analyzer API")
    
    # Verify environment variables (degraded mode allowed for tests)
    required_env_vars = ["OPENAI_API_KEY", "GITHUB_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        # Do not exit in test/dev environments; run in degraded, offline-friendly mode
        logger.warning(
            "Running in degraded mode. Missing environment variables: %s. "
            "Some features (real embeddings, GitHub API) may be disabled.", missing_vars
        )
    
    # Initialize RAG service
    rag_service = RAGService()
    logger.info("RAG service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Repository Analyzer API")
    if rag_service:
        rag_service.cleanup()


# Create FastAPI application
app = FastAPI(
    title="RAG Repository Analyzer",
    description="A REST API for analyzing GitHub repositories using RAG (Retrieval Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
# Note: Wildcard origins cannot be used together with credentials per CORS spec.
# For local/dev usage we default to allowing any origin without credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=False,  # must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(request: AnalyzeRequest):
    """
    Analyze a GitHub repository by fetching files, generating embeddings, and setting up vector storage.
    
    This endpoint processes the entire repository and prepares it for queries.
    It may take several minutes depending on repository size.
    """
    try:
        logger.info(f"Received analyze request for repository: {request.repository}")
        
        # Validate repository format
        if "/" not in request.repository or len(request.repository.split("/")) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Repository must be in format 'username/repo_name'"
            )
        
        # Analyze repository
        result = await rag_service.analyze_repository(request.repository)
        
        return AnalyzeResponse(
            status=result["status"],
            message=result["message"],
            repository=result.get("repository")
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing repository: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze repository: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """
    Query the analyzed repository with a question.
    
    The repository must be analyzed first using the /analyze endpoint.
    Returns an AI-generated answer along with source chunks from the repository.
    """
    try:
        logger.info(f"Received query request: {request.question}")
        
        # Validate question
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Query repository
        result = await rag_service.query_repository(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=[
                {
                    "file_name": source["file_name"],
                    "content_type": source["content_type"],
                    "score": source["score"],
                    "content": source["content"]
                }
                for source in result["sources"]
            ]
        )
        
    except ValueError as e:
        logger.error(f"Query validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/analyze-and-query", response_model=AnalyzeAndQueryResponse)
async def analyze_and_query_repository(request: AnalyzeAndQueryRequest):
    """
    Analyze a GitHub repository and immediately query it with a question.
    
    This is a convenience endpoint that combines the /analyze and /query operations
    into a single call. It will analyze the repository and then answer the provided
    question using the analyzed data.
    """
    try:
        logger.info(f"Received analyze-and-query request for repository: {request.repository}, question: {request.question}")
        
        # Validate inputs
        if not request.repository.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Repository path cannot be empty"
            )
            
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # First, analyze the repository
        logger.info(f"Starting repository analysis for: {request.repository}")
        analyze_result = await rag_service.analyze_repository(request.repository)
        
        if analyze_result.get("status") != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to analyze repository: {analyze_result.get('message', 'Unknown error')}"
            )
        
        # Then, query the repository
        logger.info(f"Starting query processing for: {request.question}")
        query_result = await rag_service.query_repository(request.question)
        
        logger.info(f"Successfully completed analyze-and-query for repository: {request.repository}")
        
        return AnalyzeAndQueryResponse(
            status="success",
            repository=request.repository,
            answer=query_result["answer"],
            sources=[
                {
                    "file_name": source["file_name"],
                    "content_type": source["content_type"],
                    "score": source["score"],
                    "content": source["content"]
                }
                for source in query_result["sources"]
            ],
            message=f"Repository '{request.repository}' analyzed and question answered successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        logger.error(f"Validation error in analyze-and-query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in analyze-and-query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze and query repository: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for container health monitoring.
    """
    try:
        # Basic health check - verify service is responsive
        return HealthResponse(status="healthy")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get the current status of the RAG service.
    
    Returns information about whether a repository is currently loaded and ready for queries.
    """
    try:
        status_info = rag_service.get_status()
        
        return StatusResponse(
            repository=status_info["repository"],
            ready=status_info["ready"],
            message=status_info["message"]
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "name": "RAG Repository Analyzer API",
        "version": "1.0.0",
        "description": "A REST API for analyzing GitHub repositories using RAG (Retrieval Augmented Generation)",
        "endpoints": {
            "POST /analyze": "Analyze a GitHub repository",
            "POST /query": "Query the analyzed repository",
            "POST /analyze-and-query": "Analyze a repository and immediately query it with a question",
            "GET /health": "Health check endpoint",
            "GET /status": "Get current service status",
            "GET /docs": "Interactive API documentation"
        }
    }


if __name__ == "__main__":
    # Run the application
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )