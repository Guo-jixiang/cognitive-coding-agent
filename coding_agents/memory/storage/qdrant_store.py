"""Qdrant vector database storage backend.

This module provides the QdrantVectorStore class that wraps the qdrant-client
library to offer an async interface for vector storage, retrieval, and search
operations. Collections are namespace-isolated (one per memory type/modality).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


class StorageConnectionError(ConnectionError):
    """Raised when a connection to the Qdrant storage backend fails.

    Attributes:
        host: The Qdrant server hostname that was unreachable.
        port: The Qdrant server port that was unreachable.
        message: Human-readable error description with retry guidance.
    """

    def __init__(self, host: str, port: int, cause: Exception | None = None) -> None:
        self.host = host
        self.port = port
        guidance = (
            f"Failed to connect to Qdrant at {host}:{port}. "
            f"Please verify the server is running and accessible. "
            f"Retry after checking network connectivity and server status."
        )
        if cause is not None:
            guidance += f" Original error: {cause}"
        self.message = guidance
        super().__init__(guidance)


class QdrantVectorStore:
    """Async vector storage backend using Qdrant.

    Provides namespace-isolated collection management where each memory type
    or modality gets its own collection. Supports vector CRUD operations and
    similarity search with configurable top-K results.

    Configuration is read from environment variables:
        - QDRANT_HOST: Server hostname (default: "localhost")
        - QDRANT_PORT: Server port (default: 6333)

    Example:
        >>> store = QdrantVectorStore()
        >>> await store.create_collection("episodic", dimension=1024)
        >>> await store.store("episodic", "item-1", vector, {"key": "value"})
        >>> results = await store.search("episodic", query_vec, top_k=5)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            host: Qdrant server hostname. Falls back to QDRANT_HOST env var,
                then to "localhost".
            port: Qdrant server port. Falls back to QDRANT_PORT env var,
                then to 6333.
        """
        self._host = host or os.environ.get("QDRANT_HOST", "localhost")
        self._port = port or int(os.environ.get("QDRANT_PORT", "6333"))
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            host=self._host,
            port=self._port,
        )

    @property
    def host(self) -> str:
        """The configured Qdrant server hostname."""
        return self._host

    @property
    def port(self) -> int:
        """The configured Qdrant server port."""
        return self._port

    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a vector collection if it does not already exist.

        Each collection acts as an isolated namespace for a specific memory
        type or modality.

        Args:
            name: The collection name (e.g., "episodic", "semantic",
                "perceptual_text").
            dimension: The dimensionality of vectors stored in this collection.

        Raises:
            StorageConnectionError: If the Qdrant server is unreachable.
        """
        try:
            collections = await self._client.get_collections()
            existing_names = [c.name for c in collections.collections]
            if name not in existing_names:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as exc:
            if isinstance(exc, StorageConnectionError):
                raise
            raise StorageConnectionError(self._host, self._port, exc) from exc

    async def store(
        self,
        collection: str,
        id: str,
        vector: np.ndarray,
        payload: dict[str, Any],
    ) -> bool:
        """Store a vector with associated payload in a collection.

        Args:
            collection: Target collection name.
            id: Unique identifier for the point.
            vector: The embedding vector as a numpy array.
            payload: Metadata dictionary to store alongside the vector.

        Returns:
            True if the operation succeeded.

        Raises:
            StorageConnectionError: If the Qdrant server is unreachable.
        """
        try:
            await self._client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as exc:
            if isinstance(exc, StorageConnectionError):
                raise
            raise StorageConnectionError(self._host, self._port, exc) from exc

    async def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for the most similar vectors in a collection.

        Args:
            collection: The collection to search in.
            query_vector: The query embedding vector as a numpy array.
            top_k: Maximum number of results to return. Defaults to 10.

        Returns:
            A list of (id, score, payload) tuples ordered by descending
            similarity score.

        Raises:
            StorageConnectionError: If the Qdrant server is unreachable.
        """
        try:
            response = await self._client.query_points(
                collection_name=collection,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
            )
            return [
                (
                    str(hit.id),
                    float(hit.score),
                    dict(hit.payload) if hit.payload else {},
                )
                for hit in response.points
            ]
        except Exception as exc:
            if isinstance(exc, StorageConnectionError):
                raise
            raise StorageConnectionError(self._host, self._port, exc) from exc

    async def delete(self, collection: str, id: str) -> bool:
        """Delete a point from a collection by its ID.

        Args:
            collection: The collection containing the point.
            id: The unique identifier of the point to delete.

        Returns:
            True if the deletion was successful.

        Raises:
            StorageConnectionError: If the Qdrant server is unreachable.
        """
        try:
            await self._client.delete(
                collection_name=collection,
                points_selector=[id],
            )
            return True
        except Exception as exc:
            if isinstance(exc, StorageConnectionError):
                raise
            raise StorageConnectionError(self._host, self._port, exc) from exc

    async def health_check(self) -> bool:
        """Check connectivity to the Qdrant server.

        Returns:
            True if the server is reachable and responding, False otherwise.
        """
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying client connection."""
        await self._client.close()
