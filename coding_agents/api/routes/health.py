"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from coding_agents.api.schemas.models import HealthResponse
from coding_agents.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return the health status of all dependent services.

    Checks each memory subsystem and reports overall status as:
    - "healthy": All services operational.
    - "degraded": Some services unavailable.
    - "unhealthy": No services available.

    Args:
        request: The FastAPI request object.

    Returns:
        A HealthResponse with overall status and per-service details.
    """
    manager: MemoryManager = request.app.state.memory_manager

    services: dict[str, str] = {}
    degraded = manager.degraded_subsystems

    for name in manager.subsystems:
        if name in degraded:
            services[name] = "unavailable"
        else:
            services[name] = "healthy"

    # Determine overall status
    total = len(services)
    healthy_count = sum(1 for s in services.values() if s == "healthy")

    if healthy_count == total:
        overall = "healthy"
    elif healthy_count == 0:
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(status=overall, services=services)
