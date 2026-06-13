"""Unit tests for MemoryManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from coding_agents.memory.base import MemoryItem, create_memory_item
from coding_agents.memory.manager import MemoryManager


@pytest.fixture
def mock_working():
    m = AsyncMock()
    m.store = AsyncMock(return_value=True)
    m.retrieve = AsyncMock(return_value=[])
    m.delete = AsyncMock(return_value=True)
    return m


@pytest.fixture
def mock_episodic():
    m = AsyncMock()
    m.store = AsyncMock(return_value=True)
    m.retrieve = AsyncMock(return_value=[])
    m.delete = AsyncMock(return_value=True)
    return m


@pytest.fixture
def manager(mock_working, mock_episodic):
    return MemoryManager({"working": mock_working, "episodic": mock_episodic})


# ---------------------------------------------------------------------------
# Tests for initialization
# ---------------------------------------------------------------------------


class TestMemoryManagerInitialize:
    """Tests for the initialize method."""

    async def test_initialize_sets_initialized(self, manager: MemoryManager) -> None:
        await manager.initialize()
        assert "working" not in manager.degraded_subsystems

    async def test_initialize_calls_subsystem_initialize(self, manager: MemoryManager, mock_working) -> None:
        mock_working.initialize = AsyncMock()
        await manager.initialize()
        mock_working.initialize.assert_awaited_once()

    async def test_initialize_marks_degraded_on_failure(self, mock_episodic) -> None:
        mock_working = AsyncMock()
        mock_working.initialize = AsyncMock(side_effect=Exception("fail"))
        mgr = MemoryManager({"working": mock_working, "episodic": mock_episodic})
        await mgr.initialize()
        assert "working" in mgr.degraded_subsystems

    async def test_initialize_all_fail_raises(self) -> None:
        sub1 = AsyncMock()
        sub1.initialize = AsyncMock(side_effect=Exception("fail1"))
        sub2 = AsyncMock()
        sub2.initialize = AsyncMock(side_effect=Exception("fail2"))
        mgr = MemoryManager({"a": sub1, "b": sub2})
        with pytest.raises(RuntimeError, match="No memory subsystems"):
            await mgr.initialize()


# ---------------------------------------------------------------------------
# Tests for shutdown
# ---------------------------------------------------------------------------


class TestMemoryManagerShutdown:
    """Tests for the shutdown method."""

    async def test_shutdown_calls_subsystem_shutdown(self, manager: MemoryManager, mock_working, mock_episodic) -> None:
        mock_working.shutdown = AsyncMock()
        mock_episodic.shutdown = AsyncMock()
        await manager.initialize()
        await manager.shutdown()
        # Working memory is skipped (ephemeral)
        mock_episodic.shutdown.assert_awaited_once()

    async def test_shutdown_skips_working_memory(self, manager: MemoryManager, mock_working) -> None:
        mock_working.shutdown = AsyncMock()
        await manager.initialize()
        await manager.shutdown()
        mock_working.shutdown.assert_not_awaited()

    async def test_shutdown_skips_degraded(self, mock_episodic) -> None:
        mock_working = AsyncMock()
        mock_working.initialize = AsyncMock(side_effect=Exception("fail"))
        mock_working.shutdown = AsyncMock()
        mgr = MemoryManager({"working": mock_working, "episodic": mock_episodic})
        await mgr.initialize()
        await manager_shutdown_safe(mgr)
        mock_working.shutdown.assert_not_awaited()


async def manager_shutdown_safe(mgr):
    await mgr.shutdown()


# ---------------------------------------------------------------------------
# Tests for store
# ---------------------------------------------------------------------------


class TestMemoryManagerStore:
    """Tests for the store method."""

    async def test_store_routes_to_correct_subsystem(self, manager: MemoryManager, mock_working) -> None:
        await manager.initialize()
        item = await manager.store("content", "working")
        mock_working.store.assert_awaited_once()

    async def test_store_creates_memory_item(self, manager: MemoryManager) -> None:
        await manager.initialize()
        item = await manager.store("test content", "working", importance=0.8)
        assert isinstance(item, MemoryItem)
        assert item.content == "test content"
        assert item.importance == 0.8

    async def test_store_invalid_type_raises(self, manager: MemoryManager) -> None:
        await manager.initialize()
        with pytest.raises(ValueError, match="Invalid memory_type"):
            await manager.store("content", "invalid_type")

    async def test_store_unregistered_type_raises(self) -> None:
        mgr = MemoryManager({"working": AsyncMock()})
        await mgr.initialize()
        with pytest.raises(ValueError, match="No subsystem"):
            await mgr.store("content", "episodic")

    async def test_store_degraded_raises(self, mock_episodic) -> None:
        mock_working = AsyncMock()
        mock_working.initialize = AsyncMock(side_effect=Exception("fail"))
        mgr = MemoryManager({"working": mock_working, "episodic": mock_episodic})
        await mgr.initialize()
        with pytest.raises(RuntimeError, match="degraded"):
            await mgr.store("content", "working")

    async def test_store_failure_raises(self, manager: MemoryManager, mock_working) -> None:
        mock_working.store = AsyncMock(return_value=False)
        await manager.initialize()
        with pytest.raises(RuntimeError, match="failure"):
            await manager.store("content", "working")


# ---------------------------------------------------------------------------
# Tests for retrieve
# ---------------------------------------------------------------------------


class TestMemoryManagerRetrieve:
    """Tests for the retrieve method."""

    async def test_retrieve_queries_subsystems(self, manager: MemoryManager, mock_working) -> None:
        item = create_memory_item(content="test", memory_type="working")
        mock_working.retrieve = AsyncMock(return_value=[(item, 0.8)])
        await manager.initialize()

        results = await manager.retrieve("query")
        assert len(results) == 1
        mock_working.retrieve.assert_awaited_once()

    async def test_retrieve_specific_type(self, manager: MemoryManager, mock_working, mock_episodic) -> None:
        await manager.initialize()
        await manager.retrieve("query", memory_types=["episodic"])
        mock_episodic.retrieve.assert_awaited_once()
        mock_working.retrieve.assert_not_awaited()

    async def test_retrieve_skips_degraded(self, mock_episodic) -> None:
        mock_working = AsyncMock()
        mock_working.initialize = AsyncMock(side_effect=Exception("fail"))
        mock_working.retrieve = AsyncMock(return_value=[])
        mgr = MemoryManager({"working": mock_working, "episodic": mock_episodic})
        await mgr.initialize()
        results = await mgr.retrieve("query")
        mock_working.retrieve.assert_not_awaited()

    async def test_retrieve_deduplicates(self, manager: MemoryManager, mock_working) -> None:
        item = create_memory_item(content="dup", memory_type="working")
        mock_working.retrieve = AsyncMock(return_value=[(item, 0.5), (item, 0.9)])
        await manager.initialize()
        results = await manager.retrieve("query")
        assert len(results) == 1
        assert results[0][1] == 0.9  # higher score retained

    async def test_retrieve_sorted_by_score(self, manager: MemoryManager, mock_working) -> None:
        item1 = create_memory_item(content="a", memory_type="working")
        item2 = create_memory_item(content="b", memory_type="working")
        mock_working.retrieve = AsyncMock(return_value=[(item1, 0.3), (item2, 0.9)])
        await manager.initialize()
        results = await manager.retrieve("query")
        assert results[0][1] >= results[1][1]

    async def test_retrieve_handles_subsystem_error(self, manager: MemoryManager, mock_working) -> None:
        mock_working.retrieve = AsyncMock(side_effect=Exception("fail"))
        await manager.initialize()
        results = await manager.retrieve("query")
        assert results == []


# ---------------------------------------------------------------------------
# Tests for cross_memory_search
# ---------------------------------------------------------------------------


class TestCrossMemorySearch:
    """Tests for cross_memory_search."""

    async def test_cross_search_queries_all(self, manager: MemoryManager, mock_working, mock_episodic) -> None:
        await manager.initialize()
        await manager.cross_memory_search("query")
        mock_working.retrieve.assert_awaited_once()
        mock_episodic.retrieve.assert_awaited_once()

    async def test_cross_search_deduplicates(self, manager: MemoryManager, mock_working, mock_episodic) -> None:
        item = create_memory_item(content="shared", memory_type="working")
        mock_working.retrieve = AsyncMock(return_value=[(item, 0.7)])
        mock_episodic.retrieve = AsyncMock(return_value=[(item, 0.9)])
        await manager.initialize()
        results = await manager.cross_memory_search("query")
        assert len(results) == 1
        assert results[0][1] == 0.9


# ---------------------------------------------------------------------------
# Tests for delete
# ---------------------------------------------------------------------------


class TestMemoryManagerDelete:
    """Tests for the delete method."""

    async def test_delete_delegates(self, manager: MemoryManager, mock_working) -> None:
        await manager.initialize()
        result = await manager.delete("item-id", "working")
        assert result is True
        mock_working.delete.assert_awaited_once_with("item-id")

    async def test_delete_invalid_type_raises(self, manager: MemoryManager) -> None:
        await manager.initialize()
        with pytest.raises(ValueError, match="Invalid"):
            await manager.delete("id", "bad_type")

    async def test_delete_degraded_returns_false(self, mock_episodic) -> None:
        mock_working = AsyncMock()
        mock_working.initialize = AsyncMock(side_effect=Exception("fail"))
        mgr = MemoryManager({"working": mock_working, "episodic": mock_episodic})
        await mgr.initialize()
        result = await mgr.delete("id", "working")
        assert result is False


# ---------------------------------------------------------------------------
# Tests for _deduplicate_results
# ---------------------------------------------------------------------------


class TestDeduplicateResults:
    """Tests for the deduplication logic."""

    def test_no_duplicates(self) -> None:
        item1 = create_memory_item(content="a", memory_type="working")
        item2 = create_memory_item(content="b", memory_type="working")
        results = [(item1, 0.5), (item2, 0.8)]
        deduped = MemoryManager._deduplicate_results(results)
        assert len(deduped) == 2

    def test_keeps_highest_score(self) -> None:
        item = create_memory_item(content="dup", memory_type="working")
        results = [(item, 0.3), (item, 0.9), (item, 0.5)]
        deduped = MemoryManager._deduplicate_results(results)
        assert len(deduped) == 1
        assert deduped[0][1] == 0.9

    def test_empty_input(self) -> None:
        assert MemoryManager._deduplicate_results([]) == []


# ---------------------------------------------------------------------------
# Tests for subsystems property
# ---------------------------------------------------------------------------


class TestMemoryManagerProperties:
    """Tests for properties."""

    def test_subsystems_returns_copy(self, manager: MemoryManager) -> None:
        subs = manager.subsystems
        subs["new"] = AsyncMock()
        assert "new" not in manager.subsystems

    def test_degraded_returns_copy(self, manager: MemoryManager) -> None:
        deg = manager.degraded_subsystems
        deg.add("test")
        assert "test" not in manager.degraded_subsystems
