"""Unit tests for the SQLite-backed document store.

These tests use a real in-memory SQLite database (`:memory:`) since
``aiosqlite`` is a local dependency and not an external service. Mocking
the SQLite driver would obscure FTS5 behavior we genuinely want to verify.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator

import pytest

from coding_agents.memory.storage.document_store import (
    SQLiteDocumentStore,
    StorageConnectionError,
)


@pytest.fixture
async def store() -> AsyncIterator[SQLiteDocumentStore]:
    """Provide an open in-memory document store for each test."""
    async with SQLiteDocumentStore(":memory:") as s:
        yield s


class TestStorageConnectionError:
    """Tests for the typed ``StorageConnectionError`` exception."""

    def test_is_connection_error_subclass(self) -> None:
        err = StorageConnectionError(path=":memory:")
        assert isinstance(err, ConnectionError)

    def test_contains_path_and_retry_guidance(self) -> None:
        err = StorageConnectionError(path="/tmp/test.db", retry_after=2.0)
        message = str(err)
        assert "/tmp/test.db" in message
        assert "2.0" in message
        assert err.path == "/tmp/test.db"
        assert err.retry_after == 2.0


class TestSQLiteDocumentStoreCRUD:
    """CRUD operations against an in-memory SQLite store."""

    async def test_store_and_get(self, store: SQLiteDocumentStore) -> None:
        data = {"title": "Hello", "body": "World"}
        result = await store.store("docs", "doc-1", data)
        assert result is True

        retrieved = await store.get("docs", "doc-1")
        assert retrieved == data

    async def test_get_nonexistent_returns_none(
        self, store: SQLiteDocumentStore
    ) -> None:
        retrieved = await store.get("docs", "missing")
        assert retrieved is None

    async def test_delete_existing_returns_true(
        self, store: SQLiteDocumentStore
    ) -> None:
        await store.store("docs", "doc-1", {"v": 1})
        deleted = await store.delete("docs", "doc-1")
        assert deleted is True
        assert await store.get("docs", "doc-1") is None

    async def test_delete_nonexistent_returns_false(
        self, store: SQLiteDocumentStore
    ) -> None:
        deleted = await store.delete("docs", "missing")
        assert deleted is False

    async def test_list_all(self, store: SQLiteDocumentStore) -> None:
        await store.store("docs", "doc-1", {"title": "First"})
        await store.store("docs", "doc-2", {"title": "Second"})
        await store.store("docs", "doc-3", {"title": "Third"})

        all_docs = await store.list_all("docs")
        assert len(all_docs) == 3
        titles = {d["title"] for d in all_docs}
        assert titles == {"First", "Second", "Third"}

    async def test_replace_on_duplicate_id(
        self, store: SQLiteDocumentStore
    ) -> None:
        await store.store("docs", "doc-1", {"version": 1})
        await store.store("docs", "doc-1", {"version": 2})

        retrieved = await store.get("docs", "doc-1")
        assert retrieved == {"version": 2}

        all_docs = await store.list_all("docs")
        assert len(all_docs) == 1


class TestSQLiteDocumentStoreSearch:
    """Full-text search via FTS5."""

    async def test_search_fts5(self, store: SQLiteDocumentStore) -> None:
        await store.store("docs", "doc-1", {"title": "Python programming"})
        await store.store("docs", "doc-2", {"title": "Java programming"})
        await store.store("docs", "doc-3", {"title": "Cooking recipes"})

        results = await store.search("docs", "programming")

        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"Python programming", "Java programming"}


class TestSQLiteDocumentStoreLifecycle:
    """Connection lifecycle and async context manager behavior."""

    async def test_async_context_manager(self) -> None:
        store = SQLiteDocumentStore(":memory:")
        # Connection not yet established
        assert store._connection is None  # noqa: SLF001

        async with store as opened:
            assert opened is store
            assert store._connection is not None  # noqa: SLF001
            await store.store("t", "1", {"x": 1})
            assert await store.get("t", "1") == {"x": 1}

        # Connection released after exit
        assert store._connection is None  # noqa: SLF001

    async def test_connection_error_on_invalid_path(self) -> None:
        # Parent directory does not exist; SQLite cannot open the file.
        invalid_path = os.path.join(
            tempfile.gettempdir(),
            "__cca_no_such_dir_abc_xyz__",
            "nested",
            "db.sqlite",
        )
        # Ensure the parent definitely does not exist.
        assert not os.path.exists(os.path.dirname(invalid_path))

        store = SQLiteDocumentStore(invalid_path)
        with pytest.raises(StorageConnectionError) as exc_info:
            async with store:
                pass

        assert exc_info.value.path == invalid_path
