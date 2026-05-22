"""Unit tests for WorkingMemory."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from coding_agents.memory.base import MemoryItem, create_memory_item
from coding_agents.memory.types.working import (
    MAX_TTL_SECONDS,
    MIN_TTL_SECONDS,
    WorkingMemory,
)


def _make_item(
    content: str,
    *,
    importance: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> MemoryItem:
    """Build a fresh working-memory item for tests."""
    return create_memory_item(
        content=content,
        memory_type="working",
        metadata=metadata,
        importance=importance,
    )


class TestWorkingMemoryConstruction:
    """Tests for the WorkingMemory constructor."""

    def test_default_ttl_accepted(self) -> None:
        memory = WorkingMemory()
        assert memory.ttl == 3600

    def test_invalid_ttl_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(ttl=0)

    def test_invalid_ttl_too_large_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(ttl=MAX_TTL_SECONDS + 1)

    def test_minimum_ttl_accepted(self) -> None:
        memory = WorkingMemory(ttl=MIN_TTL_SECONDS)
        assert memory.ttl == MIN_TTL_SECONDS

    def test_maximum_ttl_accepted(self) -> None:
        memory = WorkingMemory(ttl=MAX_TTL_SECONDS)
        assert memory.ttl == MAX_TTL_SECONDS


class TestWorkingMemoryStoreAndRetrieve:
    """Tests for storing and retrieving items."""

    async def test_store_and_retrieve(self) -> None:
        memory = WorkingMemory()
        item = _make_item("python programming language tutorial")
        stored = await memory.store(item)
        assert stored is True

        results = await memory.retrieve("python tutorial")
        assert len(results) == 1
        retrieved_item, score = results[0]
        assert retrieved_item.id == item.id
        assert 0.0 < score <= 1.0

    async def test_retrieve_empty_when_nothing_stored(self) -> None:
        memory = WorkingMemory()
        results = await memory.retrieve("anything")
        assert results == []

    async def test_keyword_fallback_when_tfidf_low(self) -> None:
        # Single-word stored content with a query that shares the same word
        # but doesn't trigger meaningful TF-IDF (degenerate single-doc IDF).
        memory = WorkingMemory()
        item = _make_item("alpha")
        await memory.store(item)

        results = await memory.retrieve("alpha beta gamma")
        assert len(results) == 1
        retrieved_item, score = results[0]
        assert retrieved_item.id == item.id
        assert score > 0.0

    async def test_keyword_fallback_returns_empty_for_no_overlap(self) -> None:
        memory = WorkingMemory()
        await memory.store(_make_item("alpha"))

        results = await memory.retrieve("totally unrelated")
        assert results == []


class TestWorkingMemoryTTL:
    """Tests for TTL-based expiration."""

    async def test_ttl_expiration(self) -> None:
        memory = WorkingMemory(ttl=1)
        await memory.store(_make_item("ephemeral content about pythons"))

        # First retrieval before expiry should succeed.
        before = await memory.retrieve("pythons")
        assert len(before) == 1

        await asyncio.sleep(1.2)

        after = await memory.retrieve("pythons")
        assert after == []


class TestWorkingMemoryDelete:
    """Tests for delete and clear operations."""

    async def test_delete_existing(self) -> None:
        memory = WorkingMemory()
        item = _make_item("content to delete")
        await memory.store(item)

        assert await memory.delete(item.id) is True
        results = await memory.retrieve("content")
        assert results == []

    async def test_delete_nonexistent(self) -> None:
        memory = WorkingMemory()
        unknown_id = str(uuid.uuid4())
        assert await memory.delete(unknown_id) is False

    async def test_clear_removes_all(self) -> None:
        memory = WorkingMemory()
        await memory.store(_make_item("alpha content one"))
        await memory.store(_make_item("alpha content two"))
        await memory.store(_make_item("alpha content three"))

        await memory.clear()

        results = await memory.retrieve("alpha")
        assert results == []


class TestWorkingMemoryOrdering:
    """Tests for retrieval ordering and top_k limits."""

    async def test_results_ordered_by_score(self) -> None:
        memory = WorkingMemory()
        # All items contain "alpha" but with varying additional context and
        # importance values that influence the final relevance score.
        item_high = _make_item("alpha alpha alpha keyword keyword", importance=0.9)
        item_mid = _make_item("alpha keyword something", importance=0.5)
        item_low = _make_item("alpha brief", importance=0.1)
        await memory.store(item_high)
        await memory.store(item_mid)
        await memory.store(item_low)

        results = await memory.retrieve("alpha keyword")
        assert len(results) >= 2
        scores = [score for _item, score in results]
        assert scores == sorted(scores, reverse=True)

    async def test_top_k_limits_results(self) -> None:
        memory = WorkingMemory()
        for index in range(5):
            await memory.store(_make_item(f"shared keyword item number {index}"))

        results = await memory.retrieve("shared keyword", top_k=2)
        assert len(results) == 2


class TestWorkingMemoryAccessTimestamp:
    """Tests for last-accessed timestamp updates."""

    async def test_last_accessed_updated_on_retrieval(self) -> None:
        memory = WorkingMemory()
        item = _make_item("traceable keyword content")
        # Backdate the timestamp to make the change observable.
        backdated = datetime(2000, 1, 1, tzinfo=timezone.utc)
        item.last_accessed_at = backdated
        await memory.store(item)

        results = await memory.retrieve("traceable keyword")
        assert len(results) == 1
        retrieved_item, _score = results[0]
        assert retrieved_item.last_accessed_at > backdated
