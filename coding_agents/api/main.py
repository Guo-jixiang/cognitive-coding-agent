"""FastAPI application with lifespan management.

This module creates the FastAPI app instance, registers all route modules,
and manages the application lifecycle (startup/shutdown) for the AgentEngine,
MemoryManager, and RAGPipeline.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Load .env before any component imports that read env vars
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass

from coding_agents.api.routes import chat, health, ingest, memory
from coding_agents.context.builder import ContextBuilder
from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.agents.orchestrator import Orchestrator
from coding_agents.core.engine import AgentEngine
from coding_agents.llm.client import LLMClient
from coding_agents.memory.factory import create_memory_manager
from coding_agents.memory.rag.document import DocumentProcessor
from coding_agents.memory.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    On startup:
        - Creates and initializes the MemoryManager.
        - Creates the LLMClient, ContextBuilder, ActionRegistry.
        - Creates and initializes the AgentEngine.
        - Creates the RAGPipeline.
        - Stores all components in app.state for dependency injection.

    On shutdown:
        - Shuts down the AgentEngine (persists memory state).
        - Closes the LLM client.

    Yields:
        Control to the application after startup is complete.
    """
    # Startup
    logger.info("Starting application lifecycle...")

    # Create memory manager via factory
    memory_manager = create_memory_manager()

    # Create LLM client
    llm_client = LLMClient()

    # Create context builder
    context_builder = ContextBuilder(memory_manager)

    # Create action registry (empty — actions registered by caller if needed)
    action_registry = ActionRegistry()

    # Create Orchestrator for SubAgent-based execution
    orchestrator = Orchestrator(
        llm_client=llm_client,
        memory_manager=memory_manager,
        context_builder=context_builder,
    )

    # Create agent engine with Orchestrator
    engine = AgentEngine(
        llm_client=llm_client,
        memory_manager=memory_manager,
        context_builder=context_builder,
        action_registry=action_registry,
        orchestrator=orchestrator,
    )
    await engine.initialize()

    # Create RAG pipeline
    from coding_agents.memory.embedding import EmbeddingService

    embedding_service = EmbeddingService()
    document_processor = DocumentProcessor()

    # Get episodic memory subsystem for RAG storage
    episodic_memory = memory_manager.subsystems.get("episodic")
    if episodic_memory is None:
        # Fallback: use working memory if episodic is unavailable
        episodic_memory = memory_manager.subsystems.get("working")

    rag_pipeline: RAGPipeline | None = None
    if episodic_memory is not None:
        rag_pipeline = RAGPipeline(
            document_processor=document_processor,
            embedding_service=embedding_service,
            memory=episodic_memory,
        )

    # Store in app.state for dependency injection
    app.state.engine = engine
    app.state.memory_manager = memory_manager
    app.state.rag_pipeline = rag_pipeline
    app.state.llm_client = llm_client

    logger.info("Application startup complete.")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await engine.shutdown()
    await llm_client.close()
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI application instance with all routes registered.
    """
    app = FastAPI(
        title="Cognitive Coding Agent",
        description="HTTP API for the Cognitive Coding Agent with Quad-Memory Architecture",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register route modules
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(ingest.router)
    app.include_router(health.router)

    return app


# Module-level app instance for uvicorn
app = create_app()
