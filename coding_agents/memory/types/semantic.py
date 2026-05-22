"""Semantic Memory implementation.

Semantic Memory stores abstract concepts, coding rules, and knowledge
relationships in a hybrid form: vector embeddings live in Qdrant for
similarity-based retrieval and typed relationships live in Neo4j for
graph-based reasoning.

The retrieval score combines vector similarity with a graph-derived
similarity term, then re-weights by item importance:

    relevance = (vector_similarity * 0.7 + graph_similarity * 0.3)
                * (0.8 + importance * 0.4)

If Neo4j is unavailable the module falls back to vector-only mode: items
are still searchable in Qdrant, the graph term collapses to zero, and
relationship-management calls surface the underlying error.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from coding_agents.memory.base import BaseMemory, MemoryItem, ScoringMixin
from coding_agents.memory.embedding import EmbeddingService
from coding_agents.memory.storage.neo4j_store import Neo4jGraphStore
from coding_agents.memory.storage.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# Number of neighbours at which graph similarity saturates to 1.0.
_GRAPH_SATURATION: float = 5.0

# Component weights for the semantic relevance formula.
_VECTOR_WEIGHT: float = 0.7
_GRAPH_WEIGHT: float = 0.3


class SemanticMemory(BaseMemory, ScoringMixin):
    """Knowledge-graph-aware memory backed by Qdrant and Neo4j.

    The memory uses two coordinated stores:

    * ``vector_store`` (:class:`QdrantVectorStore`) holds the embedding
      and a self-describing payload that allows reconstructing the
      original :class:`MemoryItem` on retrieval.
    * ``graph_store`` (:class:`Neo4jGraphStore`) holds the same item as
      a node and any typed relationships connecting it to other items.

    Retrieval is hybrid: a vector similarity search produces an initial
    candidate set; for each candidate the number of one-hop neighbours
    in the graph is converted into a normalized graph similarity term;
    the two are blended into a single relevance score.

    Args:
        embedding_service: The embedding service used to vectorize content.
        vector_store: The Qdrant-backed vector store.
        graph_store: The Neo4j-backed graph store.
        collection_name: Name of the Qdrant collection used by this memory.
            Defaults to ``"semantic"``.
        decay_rate: Optional decay rate retained for compatibility with the
            other memory types and the unified configuration model. The
            current scoring formula does not use a time component so the
            value is stored but not consulted. ``None`` keeps the default.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        graph_store: Neo4jGraphStore,
        collection_name: str = "semantic",
        decay_rate: float | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._collection_name = collection_name
        self._decay_rate = decay_rate

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the underlying Qdrant collection if it does not exist.

        The Neo4j store is schema-less and requires no per-memory
        initialization. The vector dimension is taken from the configured
        embedding service so that the collection always matches the
        backend that will populate it.
        """
        dimension = self._embedding_service.get_dimension()
        await self._vector_store.create_collection(self._collection_name, dimension)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def store(self, item: MemoryItem) -> bool:
        """Persist ``item`` in both the vector and graph stores.

        The content is embedded once and written to Qdrant alongside a
        self-describing payload. A matching node is then merged into
        Neo4j. If the Neo4j write fails the call still reports success
        because the item remains discoverable through vector search; the
        failure is logged at warning level.

        Args:
            item: The semantic memory item to store.

        Returns:
            ``True`` if the vector write succeeded. If the vector write
            raises, the exception is propagated unchanged.
        """
        vector = await self._embedding_service.embed(item.content)
        payload: dict[str, Any] = {
            "item_id": item.id,
            "content": item.content,
            "metadata": item.metadata,
            "importance": item.importance,
            "created_at": item.created_at.isoformat(),
            "last_accessed_at": item.last_accessed_at.isoformat(),
        }
        await self._vector_store.store(self._collection_name, item.id, vector, payload)

        node_properties: dict[str, Any] = {
            "item_id": item.id,
            "content": item.content,
            "importance": item.importance,
            "created_at": item.created_at.isoformat(),
        }
        try:
            await self._graph_store.create_node(item.id, node_properties)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning(
                "Neo4j unavailable for item %s; continuing in vector-only mode: %s",
                item.id,
                exc,
            )

        return True

    async def add_relationship(
        self,
        source_item_id: str,
        target_item_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a typed relationship between two semantic items.

        Args:
            source_item_id: The ID of the source item.
            target_item_id: The ID of the target item.
            rel_type: Relationship type. Must be one of the values
                accepted by :class:`Neo4jGraphStore` (``depends_on``,
                ``implements``, ``extends``, ``uses``, ``related_to``).
            properties: Optional relationship properties.

        Returns:
            ``True`` if the relationship was created or merged.

        Raises:
            ValueError: If ``rel_type`` is not a recognized relationship
                type. The validation is delegated to the graph store.
        """
        return await self._graph_store.create_relationship(
            source_item_id,
            target_item_id,
            rel_type,
            properties,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 10) -> list[tuple[MemoryItem, float]]:
        """Return the ``top_k`` most relevant items for ``query``.

        The query is embedded and used to fetch a wider candidate window
        from Qdrant (``top_k * 2``). For each candidate the one-hop graph
        neighbourhood is sized and converted into a normalized graph
        similarity term. The blended relevance score is then computed and
        the candidates are re-ranked.

        Args:
            query: The free-text query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(MemoryItem, relevance)`` tuples ordered by
            descending relevance, length at most ``top_k``.
        """
        query_vector = await self._embedding_service.embed(query)
        candidate_window = max(top_k * 2, top_k)
        raw_hits = await self._vector_store.search(
            self._collection_name,
            query_vector,
            top_k=candidate_window,
        )

        scored: list[tuple[MemoryItem, float]] = []
        for _point_id, vector_sim, payload in raw_hits:
            item = self._reconstruct_item(payload)
            if item is None:
                continue
            graph_sim = await self._graph_similarity(item.id)
            relevance = self.compute_relevance_score(
                similarity=vector_sim,
                time_factor=graph_sim,
                importance=item.importance,
                sim_weight=_VECTOR_WEIGHT,
                time_weight=_GRAPH_WEIGHT,
            )
            scored.append((item, relevance))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    async def _graph_similarity(self, item_id: str) -> float:
        """Compute the graph similarity term for ``item_id``.

        Currently defined as ``min(1.0, n_neighbours / _GRAPH_SATURATION)``
        where ``n_neighbours`` is the number of one-hop neighbours in the
        knowledge graph. Returns ``0.0`` when the graph store is
        unreachable so retrieval still works in vector-only mode.
        """
        try:
            neighbors = await self._graph_store.get_neighbors(item_id, depth=1)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning(
                "Graph similarity unavailable for %s; defaulting to 0.0: %s",
                item_id,
                exc,
            )
            return 0.0
        return min(1.0, len(neighbors) / _GRAPH_SATURATION)

    @staticmethod
    def _reconstruct_item(payload: dict[str, Any]) -> MemoryItem | None:
        """Rebuild a :class:`MemoryItem` from a Qdrant payload.

        Returns ``None`` if the payload is missing required fields or
        cannot be parsed. The reconstructed item is always tagged with
        ``memory_type="semantic"``.
        """
        try:
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            return MemoryItem(
                id=str(payload["item_id"]),
                content=str(payload["content"]),
                metadata=dict(metadata),
                importance=float(payload["importance"]),
                created_at=datetime.fromisoformat(str(payload["created_at"])),
                last_accessed_at=datetime.fromisoformat(str(payload["last_accessed_at"])),
                memory_type="semantic",
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed semantic payload: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete(self, item_id: str) -> bool:
        """Cascade-delete an item from both stores.

        The Neo4j node is removed first via ``DETACH DELETE`` so all
        relationships touching the node are dropped atomically. The
        Qdrant point is then removed. The call returns ``True`` if at
        least one of the two stores reports a successful deletion so
        the caller learns that some cleanup happened even when the
        other store had nothing to remove.

        Args:
            item_id: The ID of the item to delete.

        Returns:
            ``True`` if either the graph node or the vector point was
            deleted; ``False`` if both stores reported nothing to delete.
        """
        deleted_any = False

        try:
            graph_deleted = await self._graph_store.delete_node(item_id)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning("Neo4j delete failed for %s: %s", item_id, exc)
            graph_deleted = False
        if graph_deleted:
            deleted_any = True

        try:
            vector_deleted = await self._vector_store.delete(self._collection_name, item_id)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning("Qdrant delete failed for %s: %s", item_id, exc)
            vector_deleted = False
        if vector_deleted:
            deleted_any = True

        return deleted_any

    async def clear(self) -> None:
        """Bulk clear is not supported for Semantic Memory.

        Both Qdrant collections and Neo4j graphs require explicit
        management for full clears. This method is intentionally a
        no-op: callers that need to wipe semantic state should drop
        and recreate the collection and the graph database directly.
        """
        return None


__all__ = ["SemanticMemory"]
