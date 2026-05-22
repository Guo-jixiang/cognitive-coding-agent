"""Unit tests for the Qdrant vector storage backend.

Qdrant is an external service, so the entire ``AsyncQdrantClient`` is mocked
via ``unittest.mock.AsyncMock``. These tests verify the wrapper's contract:
that the right client methods are invoked with the expected arguments and
that responses are translated into the documented return shapes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from coding_agents.memory.storage.qdrant_store import (
    QdrantVectorStore,
    StorageConnectionError,
)


@pytest.fixture
def mock_client() -> Iterator[AsyncMock]:
    """Patch ``AsyncQdrantClient`` at the import location and yield the mock."""
    with patch(
        "coding_agents.memory.storage.qdrant_store.AsyncQdrantClient"
    ) as cls_mock:
        client = AsyncMock()
        cls_mock.return_value = client
        yield client


@pytest.fixture
def store(mock_client: AsyncMock) -> QdrantVectorStore:
    """Build a ``QdrantVectorStore`` wired to the mocked client."""
    return QdrantVectorStore(host="localhost", port=6333)


def _collections_response(names: list[str]) -> MagicMock:
    """Build a fake ``CollectionsResponse`` containing the given names."""
    response = MagicMock()
    collections: list[MagicMock] = []
    for real_name in names:
        col = MagicMock()
        col.name = real_name
        collections.append(col)
    response.collections = collections
    return response


class TestCreateCollection:
    """Tests for ``QdrantVectorStore.create_collection``."""

    async def test_create_collection_skips_existing(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        mock_client.get_collections.return_value = _collections_response(
            ["episodic", "semantic"]
        )

        await store.create_collection("episodic", dimension=1024)

        mock_client.get_collections.assert_awaited_once()
        mock_client.create_collection.assert_not_called()

    async def test_create_collection_creates_new(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        mock_client.get_collections.return_value = _collections_response([])

        await store.create_collection("episodic", dimension=1024)

        mock_client.create_collection.assert_awaited_once()
        kwargs = mock_client.create_collection.await_args.kwargs
        assert kwargs["collection_name"] == "episodic"
        vectors_config = kwargs["vectors_config"]
        assert isinstance(vectors_config, VectorParams)
        assert vectors_config.size == 1024
        assert vectors_config.distance == Distance.COSINE

    async def test_storage_connection_error_on_create_failure(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        mock_client.get_collections.side_effect = ConnectionRefusedError("nope")

        with pytest.raises(StorageConnectionError) as exc_info:
            await store.create_collection("episodic", dimension=1024)

        assert exc_info.value.host == "localhost"
        assert exc_info.value.port == 6333


class TestStore:
    """Tests for ``QdrantVectorStore.store``."""

    async def test_store_calls_upsert(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        payload: dict[str, Any] = {"text": "hello", "tag": "unit"}

        ok = await store.store("episodic", "item-1", vector, payload)
        assert ok is True

        mock_client.upsert.assert_awaited_once()
        kwargs = mock_client.upsert.await_args.kwargs
        assert kwargs["collection_name"] == "episodic"
        points = kwargs["points"]
        assert len(points) == 1
        point = points[0]
        assert isinstance(point, PointStruct)
        assert point.id == "item-1"
        assert point.vector == pytest.approx(vector.tolist())
        assert point.payload == payload


class TestSearch:
    """Tests for ``QdrantVectorStore.search``."""

    async def test_search_returns_results(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        hit_a = MagicMock()
        hit_a.id = "item-a"
        hit_a.score = 0.95
        hit_a.payload = {"text": "alpha"}

        hit_b = MagicMock()
        hit_b.id = "item-b"
        hit_b.score = 0.80
        hit_b.payload = {"text": "beta"}

        response = MagicMock()
        response.points = [hit_a, hit_b]
        mock_client.query_points.return_value = response

        query_vec = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        results = await store.search("episodic", query_vec, top_k=2)

        mock_client.query_points.assert_awaited_once()
        kwargs = mock_client.query_points.await_args.kwargs
        assert kwargs["collection_name"] == "episodic"
        assert kwargs["query"] == pytest.approx(query_vec.tolist())
        assert kwargs["limit"] == 2
        assert kwargs["with_payload"] is True

        assert results == [
            ("item-a", 0.95, {"text": "alpha"}),
            ("item-b", 0.80, {"text": "beta"}),
        ]


class TestDelete:
    """Tests for ``QdrantVectorStore.delete``."""

    async def test_delete_calls_client(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        ok = await store.delete("episodic", "item-1")
        assert ok is True

        mock_client.delete.assert_awaited_once()
        kwargs = mock_client.delete.await_args.kwargs
        assert kwargs["collection_name"] == "episodic"
        assert kwargs["points_selector"] == ["item-1"]


class TestHealthCheck:
    """Tests for ``QdrantVectorStore.health_check``."""

    async def test_health_check_returns_true_on_success(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        mock_client.get_collections.return_value = _collections_response([])
        assert await store.health_check() is True

    async def test_health_check_returns_false_on_failure(
        self, store: QdrantVectorStore, mock_client: AsyncMock
    ) -> None:
        mock_client.get_collections.side_effect = RuntimeError("boom")
        assert await store.health_check() is False
