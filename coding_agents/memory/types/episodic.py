"""Episodic memory implementation backed by SQLite + Qdrant.

This module implements :class:`EpisodicMemory`, the memory subsystem that
records specific events and experiences. It uses two storage backends:

- **SQLiteDocumentStore** as the canonical, structured store of every
  :class:`~coding_agents.memory.base.MemoryItem` (full payload, JSON
  serialized, with FTS5 full-text search).
- **QdrantVectorStore** as the high-performance similarity-search index
  storing only embedding vectors with a thin payload referencing the
  SQLite item id.

Retrieval prefers Qdrant for vector similarity search; if Qdrant is
unavailable (network error, collection missing, etc.) the implementation
transparently falls back to SQLite FTS so callers always get a result.

Scoring uses the unified Episodic formula::

    (vector_similarity * 0.8 + time_recency * 0.2) * (0.8 + importance * 0.4)

clamped to [0.0, 1.0]. ``time_recency`` is computed as
``exp(-decay_rate * elapsed_seconds_since_creation)`` to favour recent
events, modelling the human "recency effect".
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from coding_agents.memory.base import BaseMemory, MemoryItem, ScoringMixin

if TYPE_CHECKING:
    from coding_agents.memory.embedding import EmbeddingService
    from coding_agents.memory.storage.document_store import SQLiteDocumentStore
    from coding_agents.memory.storage.qdrant_store import QdrantVectorStore


logger = logging.getLogger(__name__)


# Episodic scoring weights — see module docstring.
_VECTOR_SIM_WEIGHT: float = 0.8
_TIME_RECENCY_WEIGHT: float = 0.2

# top_k bounds (see Requirements 3.4).
_MIN_TOP_K: int = 1
_MAX_TOP_K: int = 100

# Default decay rate: 50% decay after 24 hours (Requirements 11.2).
_DEFAULT_DECAY_RATE: float = math.log(2) / 86400


class EpisodicMemory(BaseMemory, ScoringMixin):
    """Event-based memory using SQLite for metadata and Qdrant for vectors.

    Args:
        embedding_service: Service used to vectorise content for storage and
            queries. Its ``get_dimension()`` determines the Qdrant collection
            dimensionality at initialisation time.
        document_store: SQLite document store holding the canonical, fully
            serialised memory items.
        vector_store: Optional Qdrant vector store holding embedding vectors
            keyed by memory item id. If ``None``, only SQLite FTS is used.
        table_name: SQLite table name used for episodic items.
        collection_name: Qdrant collection name used for episodic vectors.
        decay_rate: Exponential decay rate (per second) applied to
            ``elapsed_seconds_since_creation`` when computing
            ``time_recency``. Defaults to ``ln(2) / 86400`` which yields 50%
            decay after 24 hours.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        document_store: SQLiteDocumentStore,
        vector_store: QdrantVectorStore | None = None,
        table_name: str = "episodic_items",
        collection_name: str = "episodic",
        decay_rate: float | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._document_store = document_store
        self._vector_store = vector_store
        self._table_name = table_name
        self._collection_name = collection_name
        self._decay_rate: float = decay_rate if decay_rate is not None else _DEFAULT_DECAY_RATE
        self._collection_ready: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize storage backends.

        Connects the SQLite document store and (optionally) ensures the
        Qdrant collection exists. Failures in Qdrant are non-fatal: the
        memory falls back to SQLite-only operation.
        """
        # Connect SQLite (required)
        await self._document_store._connect()  # noqa: SLF001

        # Initialize Qdrant (optional, best-effort)
        if self._vector_store is None:
            self._collection_ready = False
            return
        dimension = self._embedding_service.get_dimension()
        try:
            await self._vector_store.create_collection(self._collection_name, dimension)
            self._collection_ready = True
        except Exception as exc:  # noqa: BLE001 - intentionally broad fallback
            logger.warning(
                "Failed to initialise Qdrant collection %r (dim=%d): %s",
                self._collection_name,
                dimension,
                exc,
            )
            self._collection_ready = False

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store(self, item: MemoryItem) -> bool:
        """Persist ``item`` to SQLite and (best-effort) Qdrant.

        SQLite is the canonical store and is written first. The Qdrant
        write is best-effort: failures are logged but do not propagate so
        the caller still gets ``True`` and subsequent retrieval falls
        back to FTS.

        Args:
            item: The memory item to persist.

        Returns:
            ``True`` once the SQLite write has completed.
        """
        # 1) Canonical store: full serialised item in SQLite.
        await self._document_store.store(self._table_name, item.id, item.to_dict())

        # 2) Best-effort vector store: embed + upsert to Qdrant.
        if self._vector_store is not None:
            try:
                vector = await self._embedding_service.embed(item.content)
                await self._vector_store.store(
                    self._collection_name,
                    item.id,
                    vector,
                    {"item_id": item.id},
                )
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "Vector store failed for item %s; SQLite-only path will be used: %s",
                    item.id,
                    exc,
                )

        return True

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 10) -> list[tuple[MemoryItem, float]]:
        """Retrieve the top-K most relevant items for ``query``.

        Tries Qdrant vector search first. On any failure (embedding error,
        collection missing, network error) the implementation falls back
        to SQLite FTS so callers always get a best-effort answer.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return. Must be in
                ``[1, 100]``.

        Returns:
            A list of ``(MemoryItem, score)`` tuples ordered by descending
            score. Ties break on ``created_at`` descending so newer events
            appear first.

        Raises:
            ValueError: If ``top_k`` is outside the supported range.
        """
        if not _MIN_TOP_K <= top_k <= _MAX_TOP_K:
            raise ValueError(f"top_k must be in [{_MIN_TOP_K}, {_MAX_TOP_K}], got {top_k}")

        results: list[tuple[MemoryItem, float]] = []
        used_vector_path = False

        # ----- Vector path (preferred) -----
        if self._vector_store is not None:
            try:
                query_vector = await self._embedding_service.embed(query)
                hits = await self._vector_store.search(
                    self._collection_name, query_vector, top_k=top_k * 2
                )
                used_vector_path = True
                for hit_id, similarity, _payload in hits:
                    doc = await self._document_store.get(self._table_name, hit_id)
                    if doc is None:
                        # Vector exists but SQLite row was deleted out of band.
                        continue
                    item = MemoryItem.from_dict(doc)
                    score = self._score_item(similarity, item)
                    results.append((item, score))
            except Exception as exc:  # noqa: BLE001 - fall back to FTS
                logger.warning(
                    "Qdrant retrieval unavailable for query %r; falling back to SQLite FTS: %s",
                    query,
                    exc,
                )

        # ----- FTS fallback -----
        if not used_vector_path:
            fallback_docs: list[dict[str, object]] = []
            try:
                # SQLiteDocumentStore.search returns list[dict[str, Any]];
                # widen to dict[str, object] for stricter local typing.
                fallback_docs = list(
                    await self._document_store.search(self._table_name, query, top_k=top_k * 2)
                )
            except Exception as exc:  # noqa: BLE001 - empty fallback
                logger.warning("SQLite FTS fallback failed for query %r: %s", query, exc)
            for doc in fallback_docs:
                item = MemoryItem.from_dict(dict(doc))
                # FTS rank is not normalised; treat any FTS hit as a perfect
                # textual match (similarity == 1.0) and let the importance /
                # time-recency factors do the differentiation.
                score = self._score_item(1.0, item)
                results.append((item, score))

        # Sort by score desc; tie-break on created_at desc for temporal
        # ordering (Requirement 3.6).
        results.sort(
            key=lambda pair: (pair[1], pair[0].created_at.timestamp()),
            reverse=True,
        )
        results = results[:top_k]

        # Update access timestamps and persist back to SQLite.
        now = datetime.now(timezone.utc)
        for item, _score in results:
            item.last_accessed_at = now
            try:
                await self._document_store.store(self._table_name, item.id, item.to_dict())
            except Exception as exc:  # noqa: BLE001 - non-fatal
                logger.warning(
                    "Failed to persist last_accessed_at update for item %s: %s",
                    item.id,
                    exc,
                )

        return results

    # ------------------------------------------------------------------
    # Delete / Clear
    # ------------------------------------------------------------------

    async def delete(self, item_id: str) -> bool:
        """Delete an item from both stores.

        SQLite is the canonical store: its delete result is the return
        value. Qdrant deletion is best-effort and any failure is logged.

        Args:
            item_id: The id of the memory item to remove.

        Returns:
            ``True`` if a row was removed from SQLite, ``False`` otherwise.
        """
        sqlite_deleted = await self._document_store.delete(self._table_name, item_id)
        if self._vector_store is not None:
            try:
                await self._vector_store.delete(self._collection_name, item_id)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning(
                    "Failed to delete vector for item %s from Qdrant: %s",
                    item_id,
                    exc,
                )
        return sqlite_deleted

    async def clear(self) -> None:
        """Remove every episodic item from both stores."""
        try:
            docs = await self._document_store.list_all(self._table_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list items during clear: %s", exc)
            return
        for doc in docs:
            raw_id = doc.get("id")
            if isinstance(raw_id, str):
                await self.delete(raw_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_item(self, similarity: float, item: MemoryItem) -> float:
        """Compute the episodic relevance score for ``item``.

        Uses the formula::

            (similarity * 0.8 + time_recency * 0.2) * (0.8 + importance * 0.4)

        clamped to ``[0.0, 1.0]`` via
        :meth:`ScoringMixin.compute_relevance_score`.
        """
        now = datetime.now(timezone.utc)
        elapsed = max(0.0, (now - item.created_at).total_seconds())
        time_recency = self.compute_time_decay(elapsed, self._decay_rate)
        return self.compute_relevance_score(
            similarity=similarity,
            time_factor=time_recency,
            importance=item.importance,
            sim_weight=_VECTOR_SIM_WEIGHT,
            time_weight=_TIME_RECENCY_WEIGHT,
        )


__all__ = ["EpisodicMemory"]
