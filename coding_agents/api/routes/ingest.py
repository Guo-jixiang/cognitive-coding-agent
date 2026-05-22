"""Document ingestion endpoint."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from coding_agents.api.schemas.models import IngestRequest
from coding_agents.memory.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest")
async def ingest_document(
    request: Request, body: IngestRequest
) -> dict[str, Any]:
    """Ingest a document into the memory system via the RAG pipeline.

    Parses, chunks, and stores the document for later retrieval.

    Args:
        request: The FastAPI request object.
        body: The ingest request body.

    Returns:
        A dictionary with the number of chunks stored and status.
    """
    pipeline: RAGPipeline = request.app.state.rag_pipeline

    chunks_stored = await pipeline.ingest_document(
        file_path=body.file_path,
        chunk_size=body.chunk_size,
        overlap=body.overlap_percent,
    )

    return {
        "file_path": body.file_path,
        "chunks_stored": chunks_stored,
        "status": "ingested",
    }
