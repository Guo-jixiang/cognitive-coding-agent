"""Chat endpoint with SSE streaming support."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from coding_agents.api.schemas.models import ChatRequest, ChatResponse
from coding_agents.core.engine import AgentEngine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse | EventSourceResponse:
    """Process a chat message through the Agent Engine.

    Accepts a user message and optional paradigm selection. If ``stream``
    is True, returns an SSE stream of reasoning steps followed by the
    final answer. Otherwise returns the complete response as JSON.

    Args:
        request: The FastAPI request object (provides access to app.state).
        body: The chat request body.

    Returns:
        A ChatResponse or an SSE EventSourceResponse for streaming.
    """
    engine: AgentEngine = request.app.state.engine

    if body.stream:
        return EventSourceResponse(_stream_response(engine, body))

    response = await engine.run(
        user_message=body.message, paradigm=body.paradigm
    )

    return ChatResponse(
        answer=response.answer,
        reasoning_trace=response.reasoning_trace,
        memory_updates=response.memory_updates,
    )


async def _stream_response(
    engine: AgentEngine, body: ChatRequest
) -> Any:
    """Generate SSE events for a streaming chat response.

    Yields reasoning trace steps as individual events, followed by
    the final answer event.

    Args:
        engine: The AgentEngine instance.
        body: The chat request body.

    Yields:
        SSE event data strings (JSON-encoded).
    """
    try:
        response = await engine.run(
            user_message=body.message, paradigm=body.paradigm
        )

        # Emit each reasoning step as an event
        for step in response.reasoning_trace:
            yield json.dumps({"type": "reasoning_step", "data": step})

        # Emit the final answer
        yield json.dumps({
            "type": "answer",
            "data": {
                "answer": response.answer,
                "memory_updates": response.memory_updates,
            },
        })
    except Exception as exc:
        logger.error("Streaming chat error: %s", exc, exc_info=True)
        yield json.dumps({"type": "error", "data": {"message": str(exc)}})
