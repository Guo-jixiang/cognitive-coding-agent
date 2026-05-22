"""Unit tests for :class:`SemanticMemory`.

The tests stub the embedding service with a deterministic implementation
and mock the Qdrant and Neo4j stores via ``AsyncMock(spec=...)`` so the
behaviour of :class:`SemanticMemory` itself is exercised in isolation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from coding_agents.memory.base import MemoryItem
from coding_agents.memory.embedding import EmbeddingService
from coding_agents.memory.storage.neo4j_store import Neo4jGraphStore
from coding_agents.memory.storage.qdrant_store import QdrantVectorStore
from coding_agents.memory.types.semantic import SemanticMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEmbeddingService(EmbeddingService):
    """Deterministic embedding service for tests.

    Produces the constant unit vector ``(1/sqrt(d), ..., 1/sqrt(d))`` for
    every input so that the tests do not depend on any real backend.
    """

    def __init__(self, dimension: int = 8) -> None:
        super().__init__(backends=[])
        self._dimension = dimension

    def get_dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        if text == "":
            raise ValueError("Embedding input must be a non-empty string.")
        value = 1.0 / math.sqrt(self._dimension)
        return np.full(self._dimension, value, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


def _make_item(item_id: str = "11111111-1111-4111-8111-111111111111") -> MemoryItem:
    """Build a deterministic semantic MemoryItem."""
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return MemoryItem(
        id=item_id,
        content="def hello(): return 1",
        metadata={"language": "python"},
        importance=0.5,
        created_at=now,
        last_accessed_at=now,
        memory_type="semantic",
    )


def _payload_for(item: MemoryItem) -> dict[str, Any]:
    """Reproduce the payload SemanticMemory.store would write."""
    return {
        "item_id": item.id,
        "content": item.content,
        "metadata": item.metadata,
        "importance": item.importance,
        "created_at": item.created_at.isoformat(),
        "last_accessed_at": item.last_accessed_at.isoformat(),
    }


@pytest.fixture
def embedding() -> _FakeEmbeddingService:
    return _FakeEmbeddingService(dimension=8)


@pytest.fixture
def vector_store() -> AsyncMock:
    mock = AsyncMock(spec=QdrantVectorStore)
    mock.create_collection.return_value = None
    mock.store.return_value = True
    mock.delete.return_value = True
    mock.search.return_value = []
    return mock


@pytest.fixture
def graph_store() -> AsyncMock:
    mock = AsyncMock(spec=Neo4jGraphStore)
    mock.create_node.return_value = True
    mock.create_relationship.return_value = True
    mock.delete_node.return_value = True
    mock.get_neighbors.return_value = []
    return mock


@pytest.fixture
def memory(
    embedding: _FakeEmbeddingService,
    vector_store: AsyncMock,
    graph_store: AsyncMock,
) -> SemanticMemory:
    return SemanticMemory(
        embedding_service=embedding,
        vector_store=vector_store,
        graph_store=graph_store,
        collection_name="semantic-test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialize:
    """Tests for :meth:`SemanticMemory.initialize`."""

    async def test_initialize_creates_collection(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        embedding: _FakeEmbeddingService,
    ) -> None:
        await memory.initialize()

        vector_store.create_collection.assert_awaited_once_with(
            "semantic-test",
            embedding.get_dimension(),
        )


class TestStore:
    """Tests for :meth:`SemanticMemory.store`."""

    async def test_store_writes_to_both_stores(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        item = _make_item()
        result = await memory.store(item)

        assert result is True

        vector_store.store.assert_awaited_once()
        vec_args = vector_store.store.await_args
        assert vec_args is not None
        assert vec_args.args[0] == "semantic-test"
        assert vec_args.args[1] == item.id
        vector = vec_args.args[2]
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (8,)
        payload = vec_args.args[3]
        assert payload["item_id"] == item.id
        assert payload["content"] == item.content
        assert payload["metadata"] == item.metadata
        assert payload["importance"] == item.importance
        assert payload["created_at"] == item.created_at.isoformat()
        assert payload["last_accessed_at"] == item.last_accessed_at.isoformat()

        graph_store.create_node.assert_awaited_once()
        node_args = graph_store.create_node.await_args
        assert node_args is not None
        assert node_args.args[0] == item.id
        node_props = node_args.args[1]
        assert node_props["item_id"] == item.id
        assert node_props["content"] == item.content
        assert node_props["importance"] == item.importance
        assert node_props["created_at"] == item.created_at.isoformat()

    async def test_store_succeeds_when_neo4j_fails(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        graph_store.create_node.side_effect = RuntimeError("neo4j down")
        item = _make_item()

        result = await memory.store(item)

        assert result is True
        vector_store.store.assert_awaited_once()
        graph_store.create_node.assert_awaited_once()


class TestRelationships:
    """Tests for :meth:`SemanticMemory.add_relationship`."""

    async def test_add_relationship_delegates_to_graph_store(
        self,
        memory: SemanticMemory,
        graph_store: AsyncMock,
    ) -> None:
        properties = {"weight": 0.9}

        result = await memory.add_relationship(
            "source-id",
            "target-id",
            "depends_on",
            properties,
        )

        assert result is True
        graph_store.create_relationship.assert_awaited_once_with(
            "source-id",
            "target-id",
            "depends_on",
            properties,
        )

    async def test_relationship_types_validated(
        self,
        memory: SemanticMemory,
        graph_store: AsyncMock,
    ) -> None:
        graph_store.create_relationship.side_effect = ValueError(
            "Invalid relationship type: 'invalid_rel'"
        )

        with pytest.raises(ValueError, match="Invalid relationship type"):
            await memory.add_relationship("a", "b", "invalid_rel")


class TestRetrieve:
    """Tests for :meth:`SemanticMemory.retrieve`."""

    async def test_retrieve_combines_vector_and_graph(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        item = _make_item()
        vector_store.search.return_value = [(item.id, 0.9, _payload_for(item))]
        # Two neighbours -> graph_sim = 2 / 5 = 0.4
        graph_store.get_neighbors.return_value = [
            {"node_id": "n1", "properties": {}, "relationship_type": "depends_on", "depth": 1},
            {"node_id": "n2", "properties": {}, "relationship_type": "uses", "depth": 1},
        ]

        results = await memory.retrieve("query", top_k=5)

        assert len(results) == 1
        retrieved_item, score = results[0]
        assert retrieved_item.id == item.id
        assert retrieved_item.memory_type == "semantic"

        # (0.9 * 0.7 + 0.4 * 0.3) * (0.8 + 0.5 * 0.4) = 0.75 * 1.0 = 0.75
        assert score == pytest.approx(0.75, abs=1e-6)

        vector_store.search.assert_awaited_once()
        search_args = vector_store.search.await_args
        assert search_args is not None
        assert search_args.args[0] == "semantic-test"
        # Wider candidate window: top_k * 2
        assert search_args.kwargs["top_k"] == 10
        graph_store.get_neighbors.assert_awaited_once_with(item.id, depth=1)

    async def test_retrieve_falls_back_to_vector_only_on_neo4j_failure(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        item = _make_item()
        vector_store.search.return_value = [(item.id, 1.0, _payload_for(item))]
        graph_store.get_neighbors.side_effect = RuntimeError("neo4j down")

        results = await memory.retrieve("query", top_k=3)

        assert len(results) == 1
        retrieved_item, score = results[0]
        assert retrieved_item.id == item.id
        # graph_sim = 0 -> (1.0 * 0.7 + 0.0 * 0.3) * (0.8 + 0.5 * 0.4) = 0.7 * 1.0 = 0.7
        assert score == pytest.approx(0.7, abs=1e-6)

    async def test_score_formula_correctness(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        # Build an item with importance=0.8 so the importance multiplier is
        # 0.8 + 0.8 * 0.4 = 1.12.
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        item = MemoryItem(
            id="22222222-2222-4222-8222-222222222222",
            content="content",
            metadata={},
            importance=0.8,
            created_at=now,
            last_accessed_at=now,
            memory_type="semantic",
        )
        vector_store.search.return_value = [(item.id, 0.6, _payload_for(item))]
        # Five or more neighbours saturates graph_sim to 1.0.
        graph_store.get_neighbors.return_value = [
            {"node_id": f"n{i}", "properties": {}, "relationship_type": "uses", "depth": 1}
            for i in range(5)
        ]

        results = await memory.retrieve("query", top_k=1)

        assert len(results) == 1
        _, score = results[0]
        # (0.6 * 0.7 + 1.0 * 0.3) * (0.8 + 0.8 * 0.4)
        # = (0.42 + 0.30) * 1.12 = 0.72 * 1.12 = 0.8064
        assert score == pytest.approx(0.8064, abs=1e-6)


class TestDelete:
    """Tests for :meth:`SemanticMemory.delete`."""

    async def test_cascade_delete_removes_node_and_vector(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        graph_store.delete_node.return_value = True
        vector_store.delete.return_value = True

        result = await memory.delete("item-1")

        assert result is True
        graph_store.delete_node.assert_awaited_once_with("item-1")
        vector_store.delete.assert_awaited_once_with("semantic-test", "item-1")

    async def test_delete_returns_true_when_only_vector_deleted(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        graph_store.delete_node.return_value = False
        vector_store.delete.return_value = True

        result = await memory.delete("item-1")

        assert result is True

    async def test_delete_returns_false_when_nothing_deleted(
        self,
        memory: SemanticMemory,
        vector_store: AsyncMock,
        graph_store: AsyncMock,
    ) -> None:
        graph_store.delete_node.return_value = False
        vector_store.delete.return_value = False

        result = await memory.delete("missing-id")

        assert result is False
