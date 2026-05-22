"""Perceptual Memory: multimodal storage with modal-specific vector collections.

This module implements ``PerceptualMemory``, the multimodal memory subsystem of
the Quad-Memory Architecture. It stores text, image, and audio data in
**separate Qdrant collections per modality** while using a single SQLite table
for shared metadata persistence.

Key responsibilities:

- **Modal routing**: Each ``MemoryItem`` declares its modality via
  ``metadata["modality"]``. Items are routed to the corresponding Qdrant
  collection (``perceptual_text``, ``perceptual_image``, ``perceptual_audio``).
- **Same-modal retrieval**: Search a single modality collection.
- **Cross-modal retrieval**: Search every modality collection concurrently and
  merge results by relevance score.
- **Scoring**: Uses the perceptual scoring formula
  ``(vector_similarity * 0.8 + time_recency * 0.2) * (0.8 + importance * 0.4)``
  where ``time_recency`` is an exponential decay (forgetting curve) on the
  time elapsed since the item's ``last_accessed_at`` timestamp.

Public API:
    - ``MODALITIES``: Tuple of supported modality names.
    - ``Modality``: ``Literal`` alias for the supported modality strings.
    - ``PerceptualMemory``: Concrete ``BaseMemory`` implementation.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Final, Literal

from coding_agents.memory.base import BaseMemory, MemoryItem, ScoringMixin
from coding_agents.memory.embedding import EmbeddingService
from coding_agents.memory.storage.document_store import SQLiteDocumentStore
from coding_agents.memory.storage.qdrant_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Modality declarations
# ---------------------------------------------------------------------------

#: Type alias for the supported perceptual modalities.
Modality = Literal["text", "image", "audio"]

#: Tuple of modalities in canonical order. Used for iteration and validation.
MODALITIES: Final[tuple[Modality, ...]] = ("text", "image", "audio")

# Scoring weights from the design document for perceptual memory.
_SIM_WEIGHT: Final[float] = 0.8
_TIME_WEIGHT: Final[float] = 0.2

# Default decay rate produces 50% time-decay after 24 hours of inactivity.
_DEFAULT_DECAY_RATE: Final[float] = math.log(2) / 86400.0


# ---------------------------------------------------------------------------
# PerceptualMemory implementation
# ---------------------------------------------------------------------------


class PerceptualMemory(BaseMemory, ScoringMixin):
    """Multimodal memory with separate vector collections per modality.

    Each modality (``text``, ``image``, ``audio``) is backed by its own
    Qdrant collection named ``{collection_prefix}_{modality}``. A single
    SQLite table holds the canonical, serializable representation of every
    item so that retrieved hits can be hydrated back into ``MemoryItem``
    instances regardless of which modality collection they came from.

    The class implements the :class:`BaseMemory` interface. Its ``retrieve``
    method performs a **cross-modal** search across every modality
    collection. To restrict a search to a single modality, call
    :meth:`retrieve_same_modal` directly.

    Example::

        memory = PerceptualMemory(embedding_service, qdrant, sqlite)
        await memory.initialize()

        item = create_memory_item("hello", "perceptual", {"modality": "text"})
        await memory.store(item)

        # Cross-modal search across all three modality collections.
        hits = await memory.retrieve("hello")

        # Same-modal search restricted to text only.
        text_hits = await memory.retrieve_same_modal("hello", "text")
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        document_store: SQLiteDocumentStore,
        collection_prefix: str = "perceptual",
        table_name: str = "perceptual_items",
        decay_rate: float | None = None,
    ) -> None:
        """Initialize the perceptual memory subsystem.

        Args:
            embedding_service: Embedding backend used for both store and
                retrieve operations.
            vector_store: Backing :class:`QdrantVectorStore` used for the
                modality-specific collections.
            document_store: SQLite-backed document store used to persist the
                full ``MemoryItem`` payload by ``item.id``.
            collection_prefix: Prefix for modality-specific Qdrant
                collections. Defaults to ``"perceptual"``, producing
                ``"perceptual_text"``, ``"perceptual_image"``, and
                ``"perceptual_audio"``.
            table_name: SQLite table name for shared metadata. Defaults to
                ``"perceptual_items"``.
            decay_rate: Exponential decay rate used by the time-recency
                term in the relevance formula. ``None`` (default) uses
                ``ln(2) / 86400`` — 50% decay after 24 hours.
        """
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._document_store = document_store
        self._collection_prefix = collection_prefix
        self._table_name = table_name
        self._decay_rate = decay_rate if decay_rate is not None else _DEFAULT_DECAY_RATE

    # ---------------------------------------------------------------- helpers

    def _collection_for(self, modality: str) -> str:
        """Return the Qdrant collection name for a given modality."""
        return f"{self._collection_prefix}_{modality}"

    @staticmethod
    def _validate_modality(modality: object) -> Modality:
        """Validate ``modality`` is one of the supported literals.

        Args:
            modality: The candidate modality value (typically read from an
                item's metadata dictionary).

        Returns:
            The validated modality string.

        Raises:
            ValueError: If ``modality`` is missing or not one of the
                supported modality literals.
        """
        if not isinstance(modality, str) or modality not in MODALITIES:
            raise ValueError(
                f"Invalid or missing modality {modality!r}. Must be one of {MODALITIES}."
            )
        return modality

    # ----------------------------------------------------------- lifecycle

    async def initialize(self) -> None:
        """Create one Qdrant collection per supported modality.

        Each collection is created at ``embedding_service.get_dimension()``.
        Existing collections are left untouched (delegated to
        :meth:`QdrantVectorStore.create_collection`).
        """
        dimension = self._embedding_service.get_dimension()
        for modality in MODALITIES:
            await self._vector_store.create_collection(self._collection_for(modality), dimension)

    # --------------------------------------------------------------- store

    async def store(self, item: MemoryItem) -> bool:
        """Store ``item`` in the modality-specific collection and SQLite.

        The modality is read from ``item.metadata["modality"]`` and must be
        one of :data:`MODALITIES`. The content is embedded once; the vector
        is upserted into the corresponding Qdrant collection (using
        ``item.id`` as the point id) and the full ``item.to_dict()`` payload
        is persisted to the shared SQLite table.

        Args:
            item: The memory item to store.

        Returns:
            ``True`` on successful storage in both backends.

        Raises:
            ValueError: If ``item.metadata["modality"]`` is missing or not in
                :data:`MODALITIES`.
        """
        modality = self._validate_modality(item.metadata.get("modality"))

        vector = await self._embedding_service.embed(item.content)
        collection = self._collection_for(modality)
        payload: dict[str, Any] = {"item_id": item.id, "modality": modality}

        await self._vector_store.store(collection, item.id, vector, payload)
        await self._document_store.store(self._table_name, item.id, item.to_dict())
        return True

    # ------------------------------------------------------------- retrieve

    async def retrieve(self, query: str, top_k: int = 10) -> list[tuple[MemoryItem, float]]:
        """Cross-modal retrieve (searches every modality collection).

        This is the :class:`BaseMemory` interface implementation. To restrict
        a search to a single modality, call :meth:`retrieve_same_modal`
        explicitly.

        Args:
            query: The query string.
            top_k: Maximum number of results.

        Returns:
            A list of ``(MemoryItem, relevance_score)`` tuples ordered by
            descending relevance score, with ties broken by descending
            importance.
        """
        return await self.retrieve_cross_modal(query, top_k)

    async def retrieve_same_modal(
        self, query: str, modality: str, top_k: int = 10
    ) -> list[tuple[MemoryItem, float]]:
        """Search a single modality's collection.

        Args:
            query: The query string.
            modality: One of :data:`MODALITIES`.
            top_k: Maximum number of results.

        Returns:
            A list of ``(MemoryItem, relevance_score)`` tuples ordered by
            descending relevance score.

        Raises:
            ValueError: If ``modality`` is not in :data:`MODALITIES`.
        """
        validated = self._validate_modality(modality)
        query_vector = await self._embedding_service.embed(query)
        hits = await self._vector_store.search(
            self._collection_for(validated), query_vector, top_k=top_k
        )
        return await self._materialize_hits(hits, top_k)

    async def retrieve_cross_modal(
        self, query: str, top_k: int = 10
    ) -> list[tuple[MemoryItem, float]]:
        """Search every modality collection concurrently and merge results.

        Each per-modality search is launched concurrently with
        :func:`asyncio.gather`. The combined hit list is deduplicated by
        ``item_id`` (retaining the highest similarity), scored, sorted, and
        truncated to ``top_k``.

        Args:
            query: The query string.
            top_k: Maximum number of results in the merged output.

        Returns:
            A list of ``(MemoryItem, relevance_score)`` tuples ordered by
            descending relevance score.
        """
        query_vector = await self._embedding_service.embed(query)
        searches = [
            self._vector_store.search(self._collection_for(modality), query_vector, top_k=top_k)
            for modality in MODALITIES
        ]
        per_collection: list[list[tuple[str, float, dict[str, Any]]]] = await asyncio.gather(
            *searches
        )
        merged: list[tuple[str, float, dict[str, Any]]] = []
        for hits in per_collection:
            merged.extend(hits)
        return await self._materialize_hits(merged, top_k)

    async def _materialize_hits(
        self,
        hits: list[tuple[str, float, dict[str, Any]]],
        top_k: int,
    ) -> list[tuple[MemoryItem, float]]:
        """Hydrate raw Qdrant hits into scored ``MemoryItem`` results.

        Steps:
            1. Deduplicate hits by ``item_id``, keeping the highest raw
               similarity (mostly defensive — a given id should appear in
               only one modality collection).
            2. Load the canonical document from SQLite for each id.
            3. Compute the perceptual relevance score using
               :class:`ScoringMixin`.
            4. Sort by ``(score desc, importance desc)`` and truncate to
               ``top_k``.

        Args:
            hits: Raw search hits as ``(item_id, similarity, payload)``.
            top_k: Maximum number of results to return.

        Returns:
            A sorted list of ``(MemoryItem, score)`` tuples of length at
            most ``top_k``.
        """
        best_per_id: dict[str, float] = {}
        for item_id, similarity, _payload in hits:
            existing = best_per_id.get(item_id)
            if existing is None or similarity > existing:
                best_per_id[item_id] = similarity

        scored: list[tuple[MemoryItem, float]] = []
        now = datetime.now(timezone.utc)
        for item_id, raw_similarity in best_per_id.items():
            doc = await self._document_store.get(self._table_name, item_id)
            if doc is None:
                continue
            item = MemoryItem.from_dict(doc)
            elapsed = max(0.0, (now - item.last_accessed_at).total_seconds())
            time_recency = self.compute_time_decay(elapsed, self._decay_rate)
            similarity = max(0.0, min(1.0, raw_similarity))
            score = self.compute_relevance_score(
                similarity=similarity,
                time_factor=time_recency,
                importance=item.importance,
                sim_weight=_SIM_WEIGHT,
                time_weight=_TIME_WEIGHT,
            )
            scored.append((item, score))

        scored.sort(key=lambda pair: (pair[1], pair[0].importance), reverse=True)
        return scored[:top_k]

    # ---------------------------------------------------------------- delete

    async def delete(self, item_id: str) -> bool:
        """Remove an item from its modality collection and from SQLite.

        Looks up the modality from the SQLite document so that the correct
        Qdrant collection can be targeted. If the document is absent the
        method returns ``False`` without contacting Qdrant.

        Args:
            item_id: The id of the item to remove.

        Returns:
            ``True`` if the SQLite record existed and was removed, ``False``
            otherwise.
        """
        doc = await self._document_store.get(self._table_name, item_id)
        if doc is None:
            return False

        metadata = doc.get("metadata") if isinstance(doc, dict) else None
        modality_value: object = None
        if isinstance(metadata, dict):
            modality_value = metadata.get("modality")
        if isinstance(modality_value, str) and modality_value in MODALITIES:
            await self._vector_store.delete(self._collection_for(modality_value), item_id)

        await self._document_store.delete(self._table_name, item_id)
        return True

    # ----------------------------------------------------------------- clear

    async def clear(self) -> None:
        """Best-effort removal of every item from every modality collection.

        Iterates the SQLite table and removes each item from its modality
        collection and from the SQLite table. Per-item failures (e.g.,
        Qdrant errors for orphaned documents) are swallowed so that a
        single failure does not abort the rest of the cleanup.
        """
        try:
            docs = await self._document_store.list_all(self._table_name)
        except Exception:
            docs = []

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            item_id = doc.get("id")
            if not isinstance(item_id, str):
                continue

            metadata = doc.get("metadata")
            modality_value: object = None
            if isinstance(metadata, dict):
                modality_value = metadata.get("modality")
            if isinstance(modality_value, str) and modality_value in MODALITIES:
                try:
                    await self._vector_store.delete(self._collection_for(modality_value), item_id)
                except Exception:
                    pass

            try:
                await self._document_store.delete(self._table_name, item_id)
            except Exception:
                pass


__all__ = ["MODALITIES", "Modality", "PerceptualMemory"]
