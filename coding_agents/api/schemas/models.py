"""Pydantic models for API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the POST /chat endpoint.

    Attributes:
        message: The user message to send to the agent.
        paradigm: Reasoning paradigm to use. Defaults to "reflection".
        stream: Whether to stream the response via SSE.
    """

    message: str = Field(..., min_length=1, description="User message to the agent")
    paradigm: str = Field(
        default="reflection",
        description="Reasoning paradigm: 'react', 'plan_and_solve', or 'reflection'",
    )
    stream: bool = Field(default=False, description="Enable SSE streaming response")


class ChatResponse(BaseModel):
    """Response body for the POST /chat endpoint.

    Attributes:
        answer: The agent's final answer.
        reasoning_trace: List of reasoning steps taken.
        memory_updates: IDs of memory items created during execution.
    """

    answer: str
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
    memory_updates: list[str] = Field(default_factory=list)


class MemorySearchRequest(BaseModel):
    """Request body for memory search operations.

    Attributes:
        query: The search query string.
        memory_types: Optional list of memory types to search.
        top_k: Maximum number of results to return.
    """

    query: str = Field(..., min_length=1, description="Search query")
    memory_types: list[str] | None = Field(
        default=None, description="Memory types to search (None = all)"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Max results")


class MemoryStoreRequest(BaseModel):
    """Request body for the POST /memory/store endpoint.

    Attributes:
        content: The content to store in memory.
        memory_type: Target memory type.
        metadata: Optional metadata dictionary.
        importance: Importance score in [0.0, 1.0].
    """

    content: str = Field(..., min_length=1, description="Content to store")
    memory_type: str = Field(..., description="Target memory type")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")


class IngestRequest(BaseModel):
    """Request body for the POST /ingest endpoint.

    Attributes:
        file_path: Path to the document file to ingest.
        chunk_size: Size of each chunk in characters.
        overlap_percent: Fraction of overlap between chunks.
    """

    file_path: str = Field(..., min_length=1, description="Path to document file")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Chunk size in chars")
    overlap_percent: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Overlap fraction between chunks"
    )


class HealthResponse(BaseModel):
    """Response body for the GET /health endpoint.

    Attributes:
        status: Overall health status.
        services: Per-service health status mapping.
    """

    status: str = Field(..., description="Overall status: 'healthy', 'degraded', or 'unhealthy'")
    services: dict[str, str] = Field(
        default_factory=dict, description="Service name to status mapping"
    )
