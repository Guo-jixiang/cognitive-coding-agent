"""Memory Manager for the Cognitive Coding Agent.

This module implements the central ``MemoryManager`` that coordinates all four
memory subsystems (Working, Episodic, Semantic, Perceptual). It provides:

- Unified store/retrieve interface with routing based on memory type
- Cross-memory search with concurrent queries via ``asyncio.gather``
- Result merging with deduplication (retain highest score per unique ID)
- Fault isolation: single subsystem failure does not affect others
- Degraded mode: failed subsystems are marked and skipped
- Lifecycle management: initialize all subsystems, persist on shutdown

Public API:
    - ``MemoryManager``: Core manager class aggregating all memory types.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from coding_agents.memory.base import (
    VALID_MEMORY_TYPES,
    BaseMemory,
    MemoryItem,
    create_memory_item,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """Central coordinator for the Quad-Memory Architecture.

    The manager aggregates all four memory type instances (injected via
    constructor) and provides unified store, retrieve, delete, and
    cross-memory search operations. It implements fault isolation so that
    a failure in one subsystem does not affect others.

    Attributes:
        subsystems: Mapping of memory type name to BaseMemory instance.
    """

    def __init__(self, subsystems: dict[str, BaseMemory]) -> None:
        """Initialize the MemoryManager with injected memory subsystems.

        Args:
            subsystems: A dictionary mapping memory type names (e.g.,
                ``"working"``, ``"episodic"``, ``"semantic"``,
                ``"perceptual"``) to their corresponding ``BaseMemory``
                implementations. The manager does NOT create instances
                itself — they are injected by the factory layer.
        """
        self._subsystems: dict[str, BaseMemory] = dict(subsystems)
        self._degraded: set[str] = set()
        self._initialized: bool = False

    @property
    def subsystems(self) -> dict[str, BaseMemory]:
        """Return the mapping of memory type names to subsystem instances."""
        return dict(self._subsystems)

    @property
    def degraded_subsystems(self) -> set[str]:
        """Return the set of subsystem names currently in degraded mode."""
        return set(self._degraded)

    async def initialize(self) -> None:
        """Start all subsystems and verify connectivity.

        Each subsystem is initialized independently. If a subsystem fails
        to initialize, it is marked as degraded and the manager continues
        with the remaining available subsystems.

        Raises:
            RuntimeError: If no subsystems could be initialized at all.
        """
        self._degraded.clear()
        successful = 0

        for name, subsystem in self._subsystems.items():
            try:
                # Attempt to initialize the subsystem if it has an initialize method
                if subsystem is None:
                    raise ValueError(f"Subsystem '{name}' is None")
                if hasattr(subsystem, "initialize"):
                    init_fn = getattr(subsystem, "initialize")
                    await init_fn()
                successful += 1
                logger.info("Subsystem '%s' initialized successfully.", name)
            except Exception:
                logger.warning(
                    "Subsystem '%s' failed to initialize. Marking as degraded.",
                    name,
                    exc_info=True,
                )
                self._degraded.add(name)

        if successful == 0 and self._subsystems:
            raise RuntimeError(
                f"No memory subsystems could be initialized. All failed: {sorted(self._degraded)}"
            )

        self._initialized = True
        logger.info(
            "MemoryManager initialized. Active: %d, Degraded: %d",
            successful,
            len(self._degraded),
        )

    async def shutdown(self) -> None:
        """Persist state and close connections for all subsystems.

        Working Memory is NOT persisted (requirement 10.3 — session-scoped
        data is intentionally ephemeral). Episodic and Semantic memory state
        is persisted to their respective storage backends.

        Each subsystem shutdown is isolated: a failure in one does not
        prevent others from shutting down.
        """
        for name, subsystem in self._subsystems.items():
            if name in self._degraded:
                logger.debug("Skipping shutdown for degraded subsystem '%s'.", name)
                continue
            # Working memory is ephemeral — skip persistence
            if name == "working":
                logger.debug("Working memory is ephemeral; skipping persistence.")
                continue
            try:
                # If the subsystem has a persist/shutdown method, call it.
                # BaseMemory doesn't mandate this, but concrete types may
                # implement it.
                if hasattr(subsystem, "shutdown"):
                    shutdown_fn = getattr(subsystem, "shutdown")
                    await shutdown_fn()
                elif hasattr(subsystem, "persist"):
                    persist_fn = getattr(subsystem, "persist")
                    await persist_fn()
                logger.info("Subsystem '%s' shut down successfully.", name)
            except Exception:
                logger.error(
                    "Error shutting down subsystem '%s'.",
                    name,
                    exc_info=True,
                )

        self._initialized = False
        logger.info("MemoryManager shutdown complete.")

    async def store(
        self,
        content: str,
        memory_type: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryItem:
        """Store content in the specified memory subsystem.

        Creates a ``MemoryItem`` and routes it to the appropriate subsystem
        based on ``memory_type``.

        Args:
            content: The text content to store.
            memory_type: Target memory type. Must be one of
                ``"working"``, ``"episodic"``, ``"semantic"``,
                ``"perceptual"``.
            metadata: Optional metadata dictionary.
            importance: Importance score in [0.0, 1.0]. Defaults to 0.5.

        Returns:
            The created ``MemoryItem``.

        Raises:
            ValueError: If ``memory_type`` is invalid or the subsystem is
                not registered.
            RuntimeError: If the target subsystem is in degraded mode or
                the store operation fails.
        """
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type '{memory_type}'. Must be one of: {sorted(VALID_MEMORY_TYPES)}"
            )

        if memory_type not in self._subsystems:
            raise ValueError(f"No subsystem registered for memory_type '{memory_type}'.")

        if memory_type in self._degraded:
            raise RuntimeError(
                f"Subsystem '{memory_type}' is in degraded mode and cannot accept store operations."
            )

        # Create the memory item using the factory function
        item = create_memory_item(
            content=content,
            memory_type=memory_type,  # type: ignore[arg-type]
            metadata=metadata,
            importance=importance,
        )

        subsystem = self._subsystems[memory_type]
        try:
            success = await subsystem.store(item)
            if not success:
                raise RuntimeError(
                    f"Subsystem '{memory_type}' returned failure for store operation."
                )
        except Exception as exc:
            logger.error(
                "Store operation failed for subsystem '%s': %s",
                memory_type,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Store operation failed for subsystem '{memory_type}': {exc}"
            ) from exc

        return item

    async def retrieve(
        self,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[MemoryItem, float]]:
        """Retrieve relevant items from specified memory types.

        Queries the specified memory types (or all non-degraded types if
        none specified) and returns merged, deduplicated results ordered
        by descending relevance score.

        Args:
            query: The search query string.
            memory_types: Optional list of memory types to search. If None,
                all available (non-degraded) subsystems are queried.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(MemoryItem, relevance_score)`` tuples ordered by
            descending relevance score. Equal scores are ordered by higher
            importance first.
        """
        if memory_types is None:
            target_types = [name for name in self._subsystems if name not in self._degraded]
        else:
            target_types = [
                t for t in memory_types if t in self._subsystems and t not in self._degraded
            ]

        if not target_types:
            return []

        # Query each subsystem with fault isolation
        all_results: list[tuple[MemoryItem, float]] = []
        for name in target_types:
            subsystem = self._subsystems[name]
            try:
                results = await subsystem.retrieve(query, top_k=top_k)
                # Update last_accessed_at for retrieved items
                now = datetime.now(timezone.utc)
                for item, score in results:
                    item.last_accessed_at = now
                all_results.extend(results)
            except Exception:
                logger.error(
                    "Retrieve failed for subsystem '%s'. Continuing with others.",
                    name,
                    exc_info=True,
                )
                continue

        # Deduplicate by item ID, retaining highest score
        merged = self._deduplicate_results(all_results)

        # Sort by descending score; equal scores by higher importance first
        merged.sort(key=lambda x: (-x[1], -x[0].importance))

        return merged[:top_k]

    async def delete(self, item_id: str, memory_type: str) -> bool:
        """Delete a memory item from the specified subsystem.

        Args:
            item_id: The UUID of the item to delete.
            memory_type: The memory type containing the item.

        Returns:
            True if deletion was successful, False otherwise.

        Raises:
            ValueError: If ``memory_type`` is invalid or not registered.
        """
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type '{memory_type}'. Must be one of: {sorted(VALID_MEMORY_TYPES)}"
            )

        if memory_type not in self._subsystems:
            raise ValueError(f"No subsystem registered for memory_type '{memory_type}'.")

        if memory_type in self._degraded:
            logger.warning("Cannot delete from degraded subsystem '%s'.", memory_type)
            return False

        subsystem = self._subsystems[memory_type]
        try:
            return await subsystem.delete(item_id)
        except Exception:
            logger.error(
                "Delete failed for subsystem '%s', item '%s'.",
                memory_type,
                item_id,
                exc_info=True,
            )
            return False

    async def cross_memory_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[MemoryItem, float]]:
        """Search across all memory types concurrently.

        Uses ``asyncio.gather`` to query all non-degraded subsystems in
        parallel, then merges and deduplicates results.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(MemoryItem, relevance_score)`` tuples ordered by
            descending relevance score. Equal scores are ordered by higher
            importance first.
        """
        available_subsystems = [
            (name, subsystem)
            for name, subsystem in self._subsystems.items()
            if name not in self._degraded
        ]

        if not available_subsystems:
            return []

        # Create coroutines for concurrent execution
        async def _safe_retrieve(
            name: str, subsystem: BaseMemory
        ) -> list[tuple[MemoryItem, float]]:
            """Retrieve from a subsystem with fault isolation."""
            try:
                results = await subsystem.retrieve(query, top_k=top_k)
                # Update last_accessed_at for retrieved items
                now = datetime.now(timezone.utc)
                for item, _score in results:
                    item.last_accessed_at = now
                return results
            except Exception:
                logger.error(
                    "Cross-memory search failed for subsystem '%s'. Continuing with others.",
                    name,
                    exc_info=True,
                )
                return []

        # Execute all queries concurrently
        tasks = [_safe_retrieve(name, subsystem) for name, subsystem in available_subsystems]
        results_per_subsystem = await asyncio.gather(*tasks)

        # Flatten all results
        all_results: list[tuple[MemoryItem, float]] = []
        for results in results_per_subsystem:
            all_results.extend(results)

        # Deduplicate by item ID, retaining highest score
        merged = self._deduplicate_results(all_results)

        # Sort by descending score; equal scores by higher importance first
        merged.sort(key=lambda x: (-x[1], -x[0].importance))

        return merged[:top_k]

    @staticmethod
    def _deduplicate_results(
        results: list[tuple[MemoryItem, float]],
    ) -> list[tuple[MemoryItem, float]]:
        """Deduplicate results by item ID, retaining the highest score.

        Args:
            results: List of (MemoryItem, score) tuples, possibly with
                duplicate item IDs.

        Returns:
            Deduplicated list where each unique ID appears only once,
            with the entry having the highest relevance score retained.
        """
        best: dict[str, tuple[MemoryItem, float]] = {}
        for item, score in results:
            existing = best.get(item.id)
            if existing is None or score > existing[1]:
                best[item.id] = (item, score)
        return list(best.values())


__all__ = ["MemoryManager"]
