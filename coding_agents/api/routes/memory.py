"""Memory search and store endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request

from coding_agents.api.schemas.models import MemoryStoreRequest
from coding_agents.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory")


@router.get("/search")
async def search_memory(
    request: Request,
    query: str = Query(..., min_length=1, description="Search query"),
    memory_types: str | None = Query(
        default=None, description="Comma-separated memory types to search"
    ),
    top_k: int = Query(default=10, ge=1, le=100, description="Max results"),
) -> list[dict[str, Any]]:
    """Search across configured memory types.

    Args:
        request: The FastAPI request object.
        query: The search query string.
        memory_types: Optional comma-separated list of memory types.
        top_k: Maximum number of results to return.

    Returns:
        A list of memory items with their relevance scores.
    """
    manager: MemoryManager = request.app.state.memory_manager

    types_list: list[str] | None = None
    if memory_types:
        types_list = [t.strip() for t in memory_types.split(",") if t.strip()]

    results = await manager.retrieve(
        query=query, memory_types=types_list, top_k=top_k
    )

    return [
        {
            "id": item.id,
            "content": item.content,
            "metadata": item.metadata,
            "importance": item.importance,
            "memory_type": item.memory_type,
            "score": score,
        }
        for item, score in results
    ]


@router.post("/store")
async def store_memory(
    request: Request, body: MemoryStoreRequest
) -> dict[str, Any]:
    """Store content in a specified memory type.

    Args:
        request: The FastAPI request object.
        body: The memory store request body.

    Returns:
        A dictionary with the stored item's ID and status.
    """
    manager: MemoryManager = request.app.state.memory_manager

    item = await manager.store(
        content=body.content,
        memory_type=body.memory_type,
        metadata=body.metadata,
        importance=body.importance,
    )

    return {
        "id": item.id,
        "memory_type": item.memory_type,
        "status": "stored",
    }
