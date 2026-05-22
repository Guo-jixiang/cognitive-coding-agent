"""Unit tests for the Neo4j graph storage backend.

Neo4j is an external service, so the entire driver is mocked. The mocks
emulate three patterns required by the production code:

* ``driver.session(database=...)`` returns an **async context manager**.
* ``session.run(...)`` returns an **awaitable** ``Result``.
* ``Result`` supports both ``await result.single()`` / ``await result.consume()``
  and the asynchronous iteration form ``[record async for record in result]``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agents.memory.storage.neo4j_store import (
    Neo4jGraphStore,
    StorageConnectionError,
)


class _FakeAsyncResult:
    """Mock Neo4j ``AsyncResult`` supporting ``single``, ``consume``, and ``__aiter__``."""

    def __init__(
        self,
        records: list[dict[str, Any]] | None = None,
        single_value: dict[str, Any] | None = None,
    ) -> None:
        self._records = records or []
        self._single_value = single_value
        self.consume = AsyncMock(return_value=None)

    async def single(self) -> dict[str, Any] | None:
        return self._single_value

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            for record in self._records:
                yield record

        return gen()


class _FakeAsyncSession:
    """Mock Neo4j ``AsyncSession`` doubling as an async context manager."""

    def __init__(self, result: _FakeAsyncResult) -> None:
        self._result = result
        self.run = AsyncMock(return_value=result)
        # Capture the most recent run() arguments for assertion convenience.
        self.last_query: str | None = None
        self.last_params: dict[str, Any] = {}

        async def _run(query: str, **params: Any) -> _FakeAsyncResult:
            self.last_query = query
            self.last_params = params
            return result

        self.run.side_effect = _run

    async def __aenter__(self) -> _FakeAsyncSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeAsyncDriver:
    """Mock Neo4j ``AsyncDriver`` that returns a ``_FakeAsyncSession``."""

    def __init__(self, session: _FakeAsyncSession) -> None:
        self._session = session
        self.close = AsyncMock(return_value=None)

    def session(self, database: str | None = None) -> _FakeAsyncSession:  # noqa: ARG002
        return self._session


@pytest.fixture
def fake_result() -> _FakeAsyncResult:
    """Default empty result; individual tests replace via attribute assignment."""
    return _FakeAsyncResult()


@pytest.fixture
def fake_session(fake_result: _FakeAsyncResult) -> _FakeAsyncSession:
    return _FakeAsyncSession(fake_result)


@pytest.fixture
def fake_driver(fake_session: _FakeAsyncSession) -> _FakeAsyncDriver:
    return _FakeAsyncDriver(fake_session)


@pytest.fixture
def patched_graph_db(fake_driver: _FakeAsyncDriver) -> Iterator[MagicMock]:
    """Patch ``AsyncGraphDatabase`` at the import location."""
    with patch(
        "coding_agents.memory.storage.neo4j_store.AsyncGraphDatabase"
    ) as graph_db:
        graph_db.driver = MagicMock(return_value=fake_driver)
        yield graph_db


@pytest.fixture
def store(patched_graph_db: MagicMock) -> Neo4jGraphStore:  # noqa: ARG001
    return Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
    )


class TestCreateNode:
    """Tests for ``Neo4jGraphStore.create_node``."""

    async def test_create_node_runs_query(
        self, store: Neo4jGraphStore, fake_session: _FakeAsyncSession
    ) -> None:
        ok = await store.create_node("node-1", {"label": "module"})
        assert ok is True

        assert fake_session.last_query is not None
        assert "MERGE" in fake_session.last_query
        assert "KnowledgeNode" in fake_session.last_query
        assert fake_session.last_params == {
            "node_id": "node-1",
            "properties": {"label": "module"},
        }


class TestCreateRelationship:
    """Tests for ``Neo4jGraphStore.create_relationship``."""

    async def test_create_relationship_valid_type(
        self, store: Neo4jGraphStore, fake_session: _FakeAsyncSession
    ) -> None:
        ok = await store.create_relationship(
            source_id="a",
            target_id="b",
            rel_type="depends_on",
        )
        assert ok is True

        assert fake_session.last_query is not None
        assert "DEPENDS_ON" in fake_session.last_query
        assert fake_session.last_params["source_id"] == "a"
        assert fake_session.last_params["target_id"] == "b"

    async def test_create_relationship_invalid_type_raises(
        self, store: Neo4jGraphStore
    ) -> None:
        with pytest.raises(ValueError, match="Invalid relationship type"):
            await store.create_relationship(
                source_id="a",
                target_id="b",
                rel_type="invalid_type",
            )


class TestGetNeighbors:
    """Tests for ``Neo4jGraphStore.get_neighbors``."""

    async def test_get_neighbors_returns_list(
        self,
        patched_graph_db: MagicMock,  # noqa: ARG002
    ) -> None:
        records: list[dict[str, Any]] = [
            {
                "neighbor_id": "b",
                "props": {"node_id": "b", "label": "Service"},
                "rel_type": "DEPENDS_ON",
                "depth": 1,
            },
            {
                "neighbor_id": "c",
                "props": {"node_id": "c", "label": "Repo"},
                "rel_type": "USES",
                "depth": 1,
            },
        ]
        result = _FakeAsyncResult(records=records)
        session = _FakeAsyncSession(result)
        driver = _FakeAsyncDriver(session)
        patched_graph_db.driver = MagicMock(return_value=driver)

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="u",
            password="p",
            database="neo4j",
        )

        neighbors = await store.get_neighbors("a", depth=1)

        assert len(neighbors) == 2
        assert neighbors[0] == {
            "node_id": "b",
            "properties": {"label": "Service"},
            "relationship_type": "DEPENDS_ON",
            "depth": 1,
        }
        assert neighbors[1]["node_id"] == "c"
        assert neighbors[1]["relationship_type"] == "USES"


class TestDeleteNode:
    """Tests for ``Neo4jGraphStore.delete_node``."""

    async def test_delete_node_returns_true_when_deleted(
        self,
        patched_graph_db: MagicMock,
    ) -> None:
        result = _FakeAsyncResult(single_value={"deleted_count": 1})
        session = _FakeAsyncSession(result)
        driver = _FakeAsyncDriver(session)
        patched_graph_db.driver = MagicMock(return_value=driver)

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="u",
            password="p",
            database="neo4j",
        )
        assert await store.delete_node("node-1") is True

    async def test_delete_node_returns_false_when_not_found(
        self,
        patched_graph_db: MagicMock,
    ) -> None:
        result = _FakeAsyncResult(single_value={"deleted_count": 0})
        session = _FakeAsyncSession(result)
        driver = _FakeAsyncDriver(session)
        patched_graph_db.driver = MagicMock(return_value=driver)

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="u",
            password="p",
            database="neo4j",
        )
        assert await store.delete_node("missing") is False


class TestHealthCheck:
    """Tests for ``Neo4jGraphStore.health_check``."""

    async def test_health_check_success(
        self,
        patched_graph_db: MagicMock,
    ) -> None:
        result = _FakeAsyncResult(single_value={"ping": 1})
        session = _FakeAsyncSession(result)
        driver = _FakeAsyncDriver(session)
        patched_graph_db.driver = MagicMock(return_value=driver)

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="u",
            password="p",
            database="neo4j",
        )
        assert await store.health_check() is True

    async def test_health_check_failure(
        self,
        patched_graph_db: MagicMock,
    ) -> None:
        # session.run raises -> health check should return False, not raise.
        session = _FakeAsyncSession(_FakeAsyncResult())
        session.run.side_effect = RuntimeError("connection lost")
        driver = _FakeAsyncDriver(session)
        patched_graph_db.driver = MagicMock(return_value=driver)

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="u",
            password="p",
            database="neo4j",
        )
        assert await store.health_check() is False


class TestClose:
    """Tests for ``Neo4jGraphStore.close``."""

    async def test_close_releases_driver(
        self,
        store: Neo4jGraphStore,
        fake_driver: _FakeAsyncDriver,
    ) -> None:
        # Trigger driver creation by performing a healthy operation.
        await store.create_node("n", {})

        await store.close()
        fake_driver.close.assert_awaited_once()
        # A subsequent close() is a no-op since driver was reset to None.
        await store.close()
        assert fake_driver.close.await_count == 1


class TestStorageConnectionError:
    """Tests for ``StorageConnectionError`` raised on operation failures."""

    async def test_create_node_wraps_failure(
        self,
        patched_graph_db: MagicMock,
    ) -> None:
        session = _FakeAsyncSession(_FakeAsyncResult())
        session.run.side_effect = RuntimeError("db down")
        driver = _FakeAsyncDriver(session)
        patched_graph_db.driver = MagicMock(return_value=driver)

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="u",
            password="p",
            database="neo4j",
        )
        with pytest.raises(StorageConnectionError) as exc_info:
            await store.create_node("n", {})

        assert exc_info.value.uri == "bolt://localhost:7687"
        assert "db down" in str(exc_info.value)
