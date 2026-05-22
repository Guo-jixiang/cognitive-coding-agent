"""Unit tests for :mod:`coding_agents.memory.types.perceptual`.

These tests cover modal routing, single-modal retrieval, cross-modal
retrieval, scoring, deletion, and the time-recency exponential decay
contract. The Qdrant store is replaced with an :class:`AsyncMock` (spec'd
on :class:`QdrantVectorStore`) and SQLite is exercised against a real
in-memory database for fidelity. Embedding is supplied by a deterministic
in-memory stub so vector outputs are predictable and side-effect free.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from coding_agents.memory.base import MemoryItem, create_memory_item
from coding_agents.memory.embedding import EmbeddingService
from coding_agents.memory.storage.document_store import SQLiteDocumentStore
from coding_agents.memory.storage.qdrant_store import QdrantVectorStore
from coding_agents.memory.types.perceptual import (
    MODALITIES,
    PerceptualMemory,
)

# ---------------------------------------------------------------------------
# Stubs and fixtures
# ---------------------------------------------------------------------------


class _StubEmbedding:
    """Deterministic embedding stub returning a fixed-dimension unit vector.

    The first character of each text controls which axis of the unit basis
    is set — this makes vector outputs predictable for assertions.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def get_dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not text:
            vec[0] = 1.0
            return vec
        index = ord(text[0]) % self._dimension
        vec[index] = 1.0
        return vec

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
def stub_embedding() -> EmbeddingService:
    """Return a :class:`_StubEmbedding` typed as :class:`EmbeddingService`."""
    return _StubEmbedding()  # type: ignore[return-value]


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Return an :class:`AsyncMock` spec'd on :class:`QdrantVectorStore`."""
    mock = AsyncMock(spec=QdrantVectorStore)
    mock.create_collection = AsyncMock(return_value=None)
    mock.store = AsyncMock(return_value=True)
    mock.search = AsyncMock(return_value=[])
    mock.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture
async def doc_store() -> AsyncIterator[SQLiteDocumentStore]:
    """Yield a fresh, file-backed :class:`SQLiteDocumentStore` per test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        async with SQLiteDocumentStore(path) as store:
            yield store
    finally:
        os.unlink(path)


@pytest.fixture
async def memory(
    stub_embedding: EmbeddingService,
    mock_vector_store: AsyncMock,
    doc_store: SQLiteDocumentStore,
) -> PerceptualMemory:
    """A fully wired :class:`PerceptualMemory` with mocked Qdrant."""
    return PerceptualMemory(
        embedding_service=stub_embedding,
        vector_store=mock_vector_store,
        document_store=doc_store,
    )


def _make_item(
    content: str,
    modality: str,
    *,
    importance: float = 0.5,
    last_accessed_at: datetime | None = None,
) -> MemoryItem:
    """Create a perceptual MemoryItem with the requested modality."""
    item = create_memory_item(
        content=content,
        memory_type="perceptual",
        metadata={"modality": modality},
        importance=importance,
    )
    if last_accessed_at is not None:
        item.last_accessed_at = last_accessed_at
    return item


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    """``initialize()`` must create one collection per modality."""

    async def test_initialize_creates_three_collections(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        await memory.initialize()
        assert mock_vector_store.create_collection.await_count == len(MODALITIES)

        called_names = {
            call.args[0] if call.args else call.kwargs.get("name")
            for call in mock_vector_store.create_collection.await_args_list
        }
        assert called_names == {
            "perceptual_text",
            "perceptual_image",
            "perceptual_audio",
        }

        # Every call must use the embedding-service dimension.
        for call in mock_vector_store.create_collection.await_args_list:
            dim = call.args[1] if len(call.args) > 1 else call.kwargs.get("dimension")
            assert dim == 4


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TestStore:
    """``store()`` validates modality and routes to the correct collection."""

    async def test_store_routes_to_correct_modality_text(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        item = _make_item("text content", "text")
        ok = await memory.store(item)
        assert ok is True

        mock_vector_store.store.assert_awaited_once()
        collection = mock_vector_store.store.await_args.args[0]
        point_id = mock_vector_store.store.await_args.args[1]
        assert collection == "perceptual_text"
        assert point_id == item.id

    async def test_store_routes_image_modality(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        item = _make_item("image bytes ref", "image")
        await memory.store(item)
        collection = mock_vector_store.store.await_args.args[0]
        assert collection == "perceptual_image"

    async def test_store_routes_audio_modality(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        item = _make_item("audio waveform ref", "audio")
        await memory.store(item)
        collection = mock_vector_store.store.await_args.args[0]
        assert collection == "perceptual_audio"

    async def test_store_persists_to_sqlite(
        self,
        memory: PerceptualMemory,
        doc_store: SQLiteDocumentStore,
    ) -> None:
        item = _make_item("hello", "text")
        await memory.store(item)
        loaded = await doc_store.get("perceptual_items", item.id)
        assert loaded is not None
        assert loaded["id"] == item.id
        assert loaded["metadata"]["modality"] == "text"

    async def test_store_invalid_modality_raises(
        self,
        memory: PerceptualMemory,
    ) -> None:
        item = _make_item("oops", "text")
        item.metadata["modality"] = "video"  # unsupported modality
        with pytest.raises(ValueError):
            await memory.store(item)

    async def test_store_missing_modality_raises(
        self,
        memory: PerceptualMemory,
    ) -> None:
        item = _make_item("oops", "text")
        del item.metadata["modality"]
        with pytest.raises(ValueError):
            await memory.store(item)


# ---------------------------------------------------------------------------
# retrieve_same_modal
# ---------------------------------------------------------------------------


class TestSameModalRetrieve:
    """Same-modal retrieve hits exactly one Qdrant collection."""

    async def test_same_modal_retrieve_only_searches_one_collection(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
        doc_store: SQLiteDocumentStore,
    ) -> None:
        item = _make_item("hello text", "text")
        await memory.store(item)

        mock_vector_store.search.return_value = [
            (item.id, 0.9, {"item_id": item.id, "modality": "text"}),
        ]

        results = await memory.retrieve_same_modal("hello text", "text", top_k=5)

        # Only one search call, against perceptual_text
        assert mock_vector_store.search.await_count == 1
        called_collection = mock_vector_store.search.await_args.args[0]
        assert called_collection == "perceptual_text"
        assert len(results) == 1
        assert results[0][0].id == item.id

    async def test_same_modal_invalid_modality_raises(
        self,
        memory: PerceptualMemory,
    ) -> None:
        with pytest.raises(ValueError):
            await memory.retrieve_same_modal("q", "video", top_k=5)


# ---------------------------------------------------------------------------
# retrieve_cross_modal
# ---------------------------------------------------------------------------


class TestCrossModalRetrieve:
    """Cross-modal retrieve searches all collections and merges results."""

    async def test_cross_modal_retrieve_searches_all_collections_and_merges(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
        doc_store: SQLiteDocumentStore,
    ) -> None:
        # Use a fixed timestamp so time_recency is essentially 1.0 for all
        now = datetime.now(timezone.utc)
        text_item = _make_item("t", "text", importance=0.5, last_accessed_at=now)
        image_item = _make_item("i", "image", importance=0.5, last_accessed_at=now)
        audio_item = _make_item("a", "audio", importance=0.5, last_accessed_at=now)

        for it in (text_item, image_item, audio_item):
            await memory.store(it)

        # Different similarity per modality so we can verify ordering.
        per_modality = {
            "perceptual_text": [(text_item.id, 0.5, {"modality": "text"})],
            "perceptual_image": [(image_item.id, 0.9, {"modality": "image"})],
            "perceptual_audio": [(audio_item.id, 0.7, {"modality": "audio"})],
        }

        async def fake_search(
            collection: str, _vector: np.ndarray, top_k: int = 10
        ) -> list[tuple[str, float, dict[str, Any]]]:
            return per_modality.get(collection, [])

        mock_vector_store.search.side_effect = fake_search

        # Reset state from store() calls so we're only counting retrieve work.
        retrieve_call_count_before = mock_vector_store.search.await_count

        results = await memory.retrieve_cross_modal("query", top_k=5)

        # All three collections searched.
        assert mock_vector_store.search.await_count - retrieve_call_count_before == len(MODALITIES)
        searched_collections = {
            c.args[0] for c in mock_vector_store.search.await_args_list[retrieve_call_count_before:]
        }
        assert searched_collections == set(per_modality.keys())

        # Merged and ordered by descending score (image > audio > text).
        ids = [item.id for item, _ in results]
        assert ids == [image_item.id, audio_item.id, text_item.id]


# ---------------------------------------------------------------------------
# Score formula correctness
# ---------------------------------------------------------------------------


class TestScoreFormula:
    """The retrieve scoring follows the perceptual formula and is clamped."""

    async def test_score_formula_correctness(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Pin last_accessed_at to "now" so time_recency ≈ 1.0
        now = datetime.now(timezone.utc)
        importance = 0.5
        item = _make_item("payload", "text", importance=importance, last_accessed_at=now)
        await memory.store(item)

        similarity = 0.8
        mock_vector_store.search.return_value = [(item.id, similarity, {"modality": "text"})]

        results = await memory.retrieve_same_modal("payload", "text", top_k=1)
        assert len(results) == 1
        _, score = results[0]

        # Expected = (sim*0.8 + recency*0.2) * (0.8 + importance*0.4)
        # With recency very close to 1.0 (we hydrated immediately).
        expected_lower = (similarity * 0.8 + 0.99 * 0.2) * (0.8 + importance * 0.4)
        expected_upper = (similarity * 0.8 + 1.0 * 0.2) * (0.8 + importance * 0.4)
        assert expected_lower <= score <= expected_upper
        assert 0.0 <= score <= 1.0

    async def test_score_clamped_when_similarity_above_one(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        now = datetime.now(timezone.utc)
        item = _make_item("p", "text", importance=1.0, last_accessed_at=now)
        await memory.store(item)
        # Misbehaving backend: similarity > 1.
        mock_vector_store.search.return_value = [(item.id, 5.0, {"modality": "text"})]
        results = await memory.retrieve_same_modal("p", "text", top_k=1)
        assert results[0][1] <= 1.0

    async def test_score_ordering_breaks_ties_by_importance(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        now = datetime.now(timezone.utc)
        low = _make_item("a", "text", importance=0.1, last_accessed_at=now)
        high = _make_item("b", "text", importance=0.9, last_accessed_at=now)
        await memory.store(low)
        await memory.store(high)

        # Same similarity for both → importance must break the tie.
        mock_vector_store.search.return_value = [
            (low.id, 0.5, {"modality": "text"}),
            (high.id, 0.5, {"modality": "text"}),
        ]
        results = await memory.retrieve_same_modal("q", "text", top_k=2)
        assert [it.id for it, _ in results] == [high.id, low.id]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    """``delete()`` removes from the correct modality collection and SQLite."""

    async def test_delete_removes_from_correct_modality_and_sqlite(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
        doc_store: SQLiteDocumentStore,
    ) -> None:
        item = _make_item("hi", "image")
        await memory.store(item)

        mock_vector_store.delete.reset_mock()

        ok = await memory.delete(item.id)
        assert ok is True

        mock_vector_store.delete.assert_awaited_once()
        collection = mock_vector_store.delete.await_args.args[0]
        deleted_id = mock_vector_store.delete.await_args.args[1]
        assert collection == "perceptual_image"
        assert deleted_id == item.id

        assert await doc_store.get("perceptual_items", item.id) is None

    async def test_delete_nonexistent_returns_false(
        self,
        memory: PerceptualMemory,
        mock_vector_store: AsyncMock,
    ) -> None:
        ok = await memory.delete("00000000-0000-4000-8000-000000000000")
        assert ok is False
        mock_vector_store.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Time recency exponential decay
# ---------------------------------------------------------------------------


class TestTimeRecency:
    """The recency factor must follow exp(-decay_rate * elapsed)."""

    async def test_time_recency_uses_exponential_decay(
        self,
        stub_embedding: EmbeddingService,
        mock_vector_store: AsyncMock,
        doc_store: SQLiteDocumentStore,
    ) -> None:
        # Use an aggressive, easy-to-reason-about decay rate.
        decay_rate = 0.001
        memory = PerceptualMemory(
            embedding_service=stub_embedding,
            vector_store=mock_vector_store,
            document_store=doc_store,
            decay_rate=decay_rate,
        )

        # Two items: one recent, one stale by exactly 1000 seconds.
        now = datetime.now(timezone.utc)
        recent = _make_item("r", "text", importance=0.5, last_accessed_at=now)
        stale = _make_item(
            "s", "text", importance=0.5, last_accessed_at=now - timedelta(seconds=1000)
        )
        await memory.store(recent)
        await memory.store(stale)

        similarity = 0.6
        mock_vector_store.search.return_value = [
            (recent.id, similarity, {"modality": "text"}),
            (stale.id, similarity, {"modality": "text"}),
        ]

        results = await memory.retrieve_same_modal("q", "text", top_k=2)
        scored = {item.id: score for item, score in results}

        # Recent recency ≈ 1.0; stale recency ≈ exp(-1.0).
        importance_factor = 0.8 + 0.5 * 0.4
        recent_expected_upper = (similarity * 0.8 + 1.0 * 0.2) * importance_factor
        stale_expected = (similarity * 0.8 + math.exp(-1.0) * 0.2) * importance_factor

        assert scored[recent.id] <= recent_expected_upper
        # The stale score must be strictly less than the recent score.
        assert scored[stale.id] < scored[recent.id]
        # And it must match exp(-decay_rate * elapsed) within a small tolerance
        # (we allow some slack because elapsed is measured at retrieve time and
        # therefore slightly larger than 1000 seconds).
        assert scored[stale.id] == pytest.approx(stale_expected, rel=0.05)
