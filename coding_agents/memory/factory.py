"""Factory module for assembling the complete memory system.

This module provides the ``create_memory_system`` async factory function and
the ``MemorySystem`` container dataclass. Together they wire all memory
subsystem components — storage backends, embedding service, memory types,
RAG pipeline, and context builder — into a single cohesive unit with proper
lifecycle management.

Public API:
    - ``MemorySystem``: Dataclass holding all initialized components.
    - ``create_memory_system``: Async factory that builds and returns a
      fully-wired ``MemorySystem``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from coding_agents.context.builder import ContextBuilder
from coding_agents.memory.base import BaseMemory, MemoryConfig
from coding_agents.memory.embedding import EmbeddingService
from coding_agents.memory.manager import MemoryManager
from coding_agents.memory.rag.document import DocumentProcessor
from coding_agents.memory.rag.pipeline import RAGPipeline
from coding_agents.memory.storage.document_store import SQLiteDocumentStore
from coding_agents.memory.storage.neo4j_store import Neo4jGraphStore
from coding_agents.memory.storage.qdrant_store import QdrantVectorStore
from coding_agents.memory.types.episodic import EpisodicMemory
from coding_agents.memory.types.perceptual import PerceptualMemory
from coding_agents.memory.types.semantic import SemanticMemory
from coding_agents.memory.types.working import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class MemorySystem:
    """Container holding all initialized memory system components.

    Provides a single ``shutdown`` method that gracefully tears down every
    component in the correct order (application-level first, then storage).

    Attributes:
        config: The configuration used to build this system.
        embedding_service: Unified embedding service with fallback chain.
        memory_manager: Central coordinator for all memory subsystems.
        rag_pipeline: End-to-end RAG ingestion and retrieval pipeline.
        context_builder: GSSC context-building pipeline.
    """

    config: MemoryConfig
    embedding_service: EmbeddingService
    memory_manager: MemoryManager
    rag_pipeline: RAGPipeline
    context_builder: ContextBuilder

    # Internal references for shutdown — not part of the public API.
    _sqlite_store: SQLiteDocumentStore
    _qdrant_store: QdrantVectorStore
    _neo4j_store: Neo4jGraphStore

    async def shutdown(self) -> None:
        """Gracefully shut down all components in reverse dependency order.

        Shutdown sequence:
            1. MemoryManager (persists episodic/semantic state)
            2. SQLiteDocumentStore (closes DB connection)
            3. QdrantVectorStore (closes client connection)
            4. Neo4jGraphStore (closes driver)

        Each step is isolated: a failure in one does not prevent others
        from shutting down.
        """
        # 1. Shut down the memory manager (handles subsystem persistence)
        try:
            await self.memory_manager.shutdown()
        except Exception:
            logger.error("Error shutting down MemoryManager.", exc_info=True)

        # 2. Close SQLite connection
        try:
            await self._sqlite_store.close()
        except Exception:
            logger.error("Error closing SQLiteDocumentStore.", exc_info=True)

        # 3. Close Qdrant client
        try:
            await self._qdrant_store.close()
        except Exception:
            logger.error("Error closing QdrantVectorStore.", exc_info=True)

        # 4. Close Neo4j driver
        try:
            if hasattr(self._neo4j_store, "close"):
                await self._neo4j_store.close()
        except Exception:
            logger.error("Error closing Neo4jGraphStore.", exc_info=True)

        logger.info("MemorySystem shutdown complete.")


async def create_memory_system(config: MemoryConfig | None = None) -> MemorySystem:
    """Create and initialize the complete memory system.

    Instantiates all components in dependency order, wires them together,
    and returns a fully-initialized ``MemorySystem`` ready for use.

    Wiring order:
        1. MemoryConfig (from argument or defaults)
        2. EmbeddingService (default fallback chain)
        3. Storage backends (SQLite, Qdrant, Neo4j)
        4. Memory types (Working, Episodic, Semantic, Perceptual)
        5. MemoryManager (aggregates all memory types)
        6. Initialize MemoryManager
        7. DocumentProcessor
        8. RAGPipeline (document_processor, embedding_service, episodic_memory)
        9. ContextBuilder (memory_manager)
        10. Return MemorySystem

    Args:
        config: Optional configuration. If ``None``, default values from
            ``MemoryConfig`` are used.

    Returns:
        A fully-initialized ``MemorySystem`` instance.

    Raises:
        RuntimeError: If no memory subsystems can be initialized.
    """
    # 1. Configuration
    if config is None:
        config = MemoryConfig()

    # 2. Embedding service (default fallback chain)
    embedding_service = EmbeddingService()

    # 3. Storage backends
    sqlite_store = SQLiteDocumentStore(db_path=config.sqlite_path)
    await sqlite_store._connect()  # noqa: SLF001

    qdrant_store = QdrantVectorStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
    )

    neo4j_store = Neo4jGraphStore(uri=config.neo4j_uri)

    # 4. Memory types
    working_memory = WorkingMemory(ttl=config.working_memory_ttl)

    episodic_memory = EpisodicMemory(
        embedding_service=embedding_service,
        document_store=sqlite_store,
        vector_store=qdrant_store,
    )

    semantic_memory = SemanticMemory(
        embedding_service=embedding_service,
        vector_store=qdrant_store,
        graph_store=neo4j_store,
    )

    perceptual_memory = PerceptualMemory(
        embedding_service=embedding_service,
        vector_store=qdrant_store,
        document_store=sqlite_store,
    )

    # 5. Memory manager with all subsystems
    subsystems = {
        "working": working_memory,
        "episodic": episodic_memory,
        "semantic": semantic_memory,
        "perceptual": perceptual_memory,
    }
    memory_manager = MemoryManager(subsystems=subsystems)

    # 6. Initialize the manager (verifies subsystem health)
    await memory_manager.initialize()

    # 7. Document processor
    document_processor = DocumentProcessor()

    # 8. RAG pipeline
    rag_pipeline = RAGPipeline(
        document_processor=document_processor,
        embedding_service=embedding_service,
        memory=episodic_memory,
    )

    # 9. Context builder
    context_builder = ContextBuilder(memory_manager=memory_manager)

    # 10. Assemble and return
    logger.info("MemorySystem created successfully with config: %s", config)

    return MemorySystem(
        config=config,
        embedding_service=embedding_service,
        memory_manager=memory_manager,
        rag_pipeline=rag_pipeline,
        context_builder=context_builder,
        _sqlite_store=sqlite_store,
        _qdrant_store=qdrant_store,
        _neo4j_store=neo4j_store,
    )


def create_memory_manager(config: MemoryConfig | None = None) -> MemoryManager:
    """Create a lightweight MemoryManager with WorkingMemory and EpisodicMemory.

    This synchronous factory creates a MemoryManager suitable for quick
    startup scenarios (CLI, testing) where external services (Qdrant,
    Neo4j) may not be available. WorkingMemory provides in-session storage
    and EpisodicMemory (SQLite-only, no Qdrant) provides cross-session
    persistence.

    The SQLite connection is deferred — it will be established on the first
    async operation (store/retrieve) via EpisodicMemory's internal calls to
    the document store.

    For a fully-wired system with all memory types and storage backends,
    use :func:`create_memory_system` instead.

    Args:
        config: Optional configuration. If ``None``, defaults are used.

    Returns:
        A MemoryManager with WorkingMemory and EpisodicMemory registered.
    """
    if config is None:
        config = MemoryConfig()

    working_memory: BaseMemory = WorkingMemory(ttl=config.working_memory_ttl)

    # Create SQLite-backed EpisodicMemory (no Qdrant dependency).
    # Connection is deferred to the first async operation.
    sqlite_store = SQLiteDocumentStore(db_path=config.sqlite_path)
    embedding_service = EmbeddingService()
    episodic_memory: BaseMemory = EpisodicMemory(
        embedding_service=embedding_service,
        document_store=sqlite_store,
        vector_store=None,
    )

    subsystems: dict[str, BaseMemory] = {
        "working": working_memory,
        "episodic": episodic_memory,
    }

    return MemoryManager(subsystems=subsystems)


__all__ = ["MemorySystem", "create_memory_manager", "create_memory_system"]
