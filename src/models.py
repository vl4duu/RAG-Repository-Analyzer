from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AnalyzeRequest(BaseModel):
    repository: str = Field(..., description="Repository path in format 'username/repo_name'")


class AnalyzeResponse(BaseModel):
    status: str
    message: str
    repository: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the repository")


class SourceInfo(BaseModel):
    file_name: str
    content_type: str
    score: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    repository: Optional[str] = None
    ready: bool
    message: Optional[str] = None


class AnalyzeAndQueryRequest(BaseModel):
    repository: str = Field(..., description="Repository path in format 'username/repo_name'")
    question: str = Field(..., description="Question to ask about the repository")


class AnalyzeAndQueryResponse(BaseModel):
    status: str
    repository: str
    answer: str
    sources: List[SourceInfo]
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str