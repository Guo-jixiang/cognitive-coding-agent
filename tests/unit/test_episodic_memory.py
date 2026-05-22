"""Unit tests for :mod:`coding_agents.memory.types.episodic`.

These tests use a real in-memory :class:`SQLiteDocumentStore` and a mock
:class:`QdrantVectorStore` so we can deterministically test both the
preferred Qdrant retrieval path and the SQLite FTS fallback path
without standing up an actual Qdrant server.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from coding_agents.memory.base import MemoryItem, create_memory_item
from coding_agents.memory.embedding import EmbeddingService, TFIDFEmbedding
from coding_agents.memory.storage.document_store import SQLiteDocumentStore
from coding_agents.memory.storage.qdrant_store import (
    QdrantVectorStore,
    StorageConnectionError,
)
from coding_agents.memory.types.episodic import EpisodicMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def document_store() -> AsyncIterator[SQLiteDocumentStore]:
    """Provide a real, in-memory :class:`SQLiteDocumentStore`.

    Using ``:memory:`` keeps the test hermetic and fast while still
    exercising the real FTS5 code path (so the fallback behaviour we
    assert is genuine).
    """
    async with SQLiteDocumentStore(":memory:") as store:
        yield store


@pytest.fixture
def vector_store() -> AsyncMock:
    """Provide a mock :class:`QdrantVectorStore` with async methods."""
    mock = AsyncMock(spec=QdrantVectorStore)
    mock.create_collection.return_value = None
    mock.store.return_value = True
    mock.search.return_value = []
    mock.delete.return_value = True
    return mock


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Provide a deterministic embedding service backed by TF-IDF.

    A small dimension keeps tests fast and predictable.
    """
    return EmbeddingService(backends=[TFIDFEmbedding(dimension=32)])


@pytest.fixture
async def episodic(
    embedding_service: EmbeddingService,
    document_store: SQLiteDocumentStore,
    vector_store: AsyncMock,
) -> EpisodicMemory:
    """Provide an initialised :class:`EpisodicMemory` instance."""
    memory = EpisodicMemory(
        embedding_service=embedding_service,
        document_store=document_store,
        vector_store=vector_store,
    )
    await memory.initialize()
    return memory


def _make_item(
    content: str = "An interesting event occurred",
    importance: float = 0.5,
    created_at: datetime | None = None,
) -> MemoryItem:
    """Construct an episodic :class:`MemoryItem` for tests."""
    item = create_memory_item(
        content=content,
        memory_type="episodic",
        importance=importance,
    )
    if created_at is not None:
        item.created_at = created_at
        item.last_accessed_at = created_at
    return item


# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------


class TestInitialize:
    """Tests for :meth:`EpisodicMemory.initialize`."""

    async def test_initialize_creates_collection(
        self,
        embedding_service: EmbeddingService,
        document_store: SQLiteDocumentStore,
        vector_store: AsyncMock,
    ) -> None:
        memory = EpisodicMemory(
            embedding_service=embedding_service,
            document_store=document_store,
            vector_store=vector_store,
            collection_name="my_collection",
        )

        await memory.initialize()

        vector_store.create_collection.assert_awaited_once_with(
            "my_collection", embedding_service.get_dimension()
        )

    async def test_initialize_tolerates_qdrant_failure(
        self,
        embedding_service: EmbeddingService,
        document_store: SQLiteDocumentStore,
        vector_store: AsyncMock,
    ) -> None:
        vector_store.create_collection.side_effect = StorageConnectionError("localhost", 6333)
        memory = EpisodicMemory(
            embedding_service=embedding_service,
            document_store=document_store,
            vector_store=vector_store,
        )
        # Should not raise; falls back to SQLite-only mode.
        await memory.initialize()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    """Tests for :meth:`EpisodicMemory.store`."""

    async def test_store_persists_to_sqlite_and_qdrant(
        self,
        episodic: EpisodicMemory,
        document_store: SQLiteDocumentStore,
        vector_store: AsyncMock,
    ) -> None:
        item = _make_item(content="A meaningful coding session")

        result = await episodic.store(item)

        assert result is True

        # SQLite: full payload round-trips.
        stored_doc = await document_store.get("episodic_items", item.id)
        assert stored_doc is not None
        assert stored_doc["id"] == item.id
        assert stored_doc["content"] == item.content

        # Qdrant: vector store called with the item id and matching payload.
        vector_store.store.assert_awaited_once()
        call_kwargs = vector_store.store.call_args
        args = call_kwargs.args
        kwargs = call_kwargs.kwargs
        # Support both positional and kwargs invocation styles.
        collection = kwargs.get("collection", args[0] if args else None)
        item_id = kwargs.get("id", args[1] if len(args) > 1 else None)
        payload = kwargs.get("payload", args[3] if len(args) > 3 else None)
        assert collection == "episodic"
        assert item_id == item.id
        assert payload == {"item_id": item.id}

    async def test_store_succeeds_when_qdrant_fails(
        self,
        episodic: EpisodicMemory,
        document_store: SQLiteDocumentStore,
        vector_store: AsyncMock,
    ) -> None:
        vector_store.store.side_effect = StorageConnectionError("localhost", 6333)
        item = _make_item(content="Stored even when Qdrant is down")

        result = await episodic.store(item)

        assert result is True
        # SQLite still has the canonical row.
        stored_doc = await document_store.get("episodic_items", item.id)
        assert stored_doc is not None
        assert stored_doc["id"] == item.id


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Tests for :meth:`EpisodicMemory.retrieve`."""

    async def test_top_k_validation(self, episodic: EpisodicMemory) -> None:
        with pytest.raises(ValueError):
            await episodic.retrieve("anything", top_k=0)
        with pytest.raises(ValueError):
            await episodic.retrieve("anything", top_k=101)

    async def test_retrieve_uses_qdrant_when_available(
        self,
        episodic: EpisodicMemory,
        vector_store: AsyncMock,
    ) -> None:
        # Two stored items: known importance values let us verify the formula.
        item_high = _make_item(content="alpha event", importance=1.0)
        item_low = _make_item(content="beta event", importance=0.0)
        await episodic.store(item_high)
        await episodic.store(item_low)

        # Mock Qdrant search to return both items with known similarities.
        vector_store.search.return_value = [
            (item_high.id, 1.0, {"item_id": item_high.id}),
            (item_low.id, 0.5, {"item_id": item_low.id}),
        ]

        results = await episodic.retrieve("alpha", top_k=2)

        assert len(results) == 2
        assert vector_store.search.await_count >= 1

        # Score formula: (sim * 0.8 + recency * 0.2) * (0.8 + importance * 0.4)
        # For freshly created items recency ~= 1.0.
        # item_high: (1.0 * 0.8 + 1.0 * 0.2) * (0.8 + 1.0 * 0.4) = 1.2 -> clamp to 1.0
        # item_low : (0.5 * 0.8 + 1.0 * 0.2) * (0.8 + 0.0 * 0.4) = 0.6 * 0.8 = 0.48
        scores = {item.id: score for item, score in results}
        assert scores[item_high.id] == pytest.approx(1.0, abs=1e-3)
        assert scores[item_low.id] == pytest.approx(0.48, abs=1e-3)

        # Highest score first.
        assert results[0][0].id == item_high.id

    async def test_retrieve_falls_back_to_sqlite_when_qdrant_fails(
        self,
        episodic: EpisodicMemory,
        vector_store: AsyncMock,
    ) -> None:
        item = _make_item(content="python programming session", importance=0.5)
        await episodic.store(item)

        # Make Qdrant search blow up so the FTS fallback triggers.
        vector_store.search.side_effect = StorageConnectionError("localhost", 6333)

        results = await episodic.retrieve("python", top_k=5)

        assert len(results) == 1
        retrieved_item, score = results[0]
        assert retrieved_item.id == item.id
        # Fallback uses similarity = 1.0; recency ~= 1.0; importance = 0.5.
        # score = (1.0 * 0.8 + 1.0 * 0.2) * (0.8 + 0.5 * 0.4) = 1.0 * 1.0 = 1.0
        assert score == pytest.approx(1.0, abs=1e-3)

    async def test_retrieve_updates_last_accessed_at(
        self,
        episodic: EpisodicMemory,
        document_store: SQLiteDocumentStore,
        vector_store: AsyncMock,
    ) -> None:
        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        item = _make_item(content="historic event", created_at=old_time)
        await episodic.store(item)

        vector_store.search.return_value = [
            (item.id, 0.9, {"item_id": item.id}),
        ]

        results = await episodic.retrieve("historic", top_k=1)
        assert len(results) == 1

        # The persisted last_accessed_at should now be much more recent.
        stored = await document_store.get("episodic_items", item.id)
        assert stored is not None
        last_accessed = datetime.fromisoformat(stored["last_accessed_at"])
        assert last_accessed > old_time + timedelta(days=1)

    async def test_temporal_ordering(
        self,
        episodic: EpisodicMemory,
        vector_store: AsyncMock,
    ) -> None:
        # Two items with identical importance and similarity, but different
        # creation times. The newer one must come first.
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        new_time = datetime.now(timezone.utc)
        item_old = _make_item(content="older event", created_at=old_time)
        item_new = _make_item(content="newer event", created_at=new_time)
        await episodic.store(item_old)
        await episodic.store(item_new)

        vector_store.search.return_value = [
            (item_old.id, 1.0, {"item_id": item_old.id}),
            (item_new.id, 1.0, {"item_id": item_new.id}),
        ]

        results = await episodic.retrieve("event", top_k=2)
        ordered_ids = [item.id for item, _ in results]
        assert ordered_ids == [item_new.id, item_old.id]


# ---------------------------------------------------------------------------
# Delete / Clear
# ---------------------------------------------------------------------------


class TestDelete:
    """Tests for :meth:`EpisodicMemory.delete` and :meth:`clear`."""

    async def test_delete_removes_from_both_stores(
        self,
        episodic: EpisodicMemory,
        document_store: SQLiteDocumentStore,
        vector_store: AsyncMock,
    ) -> None:
        item = _make_item(content="delete me")
        await episodic.store(item)

        result = await episodic.delete(item.id)

        assert result is True
        assert await document_store.get("episodic_items", item.id) is None
        vector_store.delete.assert_awaited_with("episodic", item.id)

    async def test_delete_returns_false_when_missing(
        self,
        episodic: EpisodicMemory,
    ) -> None:
        result = await episodic.delete("550e8400-e29b-41d4-a716-446655440000")
        assert result is False

    async def test_delete_tolerates_qdrant_failure(
        self,
        episodic: EpisodicMemory,
        vector_store: AsyncMock,
    ) -> None:
        item = _make_item(content="delete me too")
        await episodic.store(item)
        vector_store.delete.side_effect = StorageConnectionError("localhost", 6333)

        result = await episodic.delete(item.id)
        assert result is True

    async def test_clear_removes_all_items(
        self,
        episodic: EpisodicMemory,
        document_store: SQLiteDocumentStore,
    ) -> None:
        for n in range(3):
            await episodic.store(_make_item(content=f"event {n}"))

        await episodic.clear()

        remaining = await document_store.list_all("episodic_items")
        assert remaining == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """Round-trip test: a new EpisodicMemory recovers items from SQLite."""

    async def test_persistence_round_trip(
        self,
        embedding_service: EmbeddingService,
        document_store: SQLiteDocumentStore,
    ) -> None:
        # First memory writes through both stores.
        first_vector_store = AsyncMock(spec=QdrantVectorStore)
        first_vector_store.create_collection.return_value = None
        first_vector_store.store.return_value = True

        first = EpisodicMemory(
            embedding_service=embedding_service,
            document_store=document_store,
            vector_store=first_vector_store,
        )
        await first.initialize()

        items = [_make_item(content=f"event number {n}") for n in range(3)]
        for item in items:
            await first.store(item)

        # A fresh EpisodicMemory shares the same SQLite store and a fresh
        # Qdrant mock that always fails -> retrieval must use FTS only.
        second_vector_store = AsyncMock(spec=QdrantVectorStore)
        second_vector_store.create_collection.return_value = None
        second_vector_store.search.side_effect = StorageConnectionError("localhost", 6333)

        second = EpisodicMemory(
            embedding_service=embedding_service,
            document_store=document_store,
            vector_store=second_vector_store,
        )
        await second.initialize()

        # We can recover every original item from FTS.
        for item in items:
            results = await second.retrieve(item.content, top_k=5)
            recovered_ids = {recovered.id for recovered, _ in results}
            assert item.id in recovered_ids


# ---------------------------------------------------------------------------
# Cancellation safety: ensure retrieve does not leave dangling tasks. This
# is a smoke test that we can call retrieve concurrently without raising.
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Smoke tests for concurrent retrieve calls."""

    async def test_concurrent_retrieve(
        self,
        episodic: EpisodicMemory,
        vector_store: AsyncMock,
    ) -> None:
        item = _make_item(content="concurrent event")
        await episodic.store(item)

        async def _hits(*_a: object, **_kw: object) -> list[tuple[str, float, dict[str, Any]]]:
            return [(item.id, 0.9, {"item_id": item.id})]

        # Make search return a fresh list each time.
        vector_store.search.side_effect = _hits

        # Numpy import is here for the type checker; the embedding service
        # ultimately produces ndarrays so referencing the symbol guards
        # against accidental removal.
        assert isinstance(np.zeros(1), np.ndarray)

        results = await asyncio.gather(
            *(episodic.retrieve("concurrent", top_k=1) for _ in range(4))
        )
        for batch in results:
            assert len(batch) == 1
