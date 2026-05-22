"""Neo4j graph storage backend for the Cognitive Coding Agent.

This module provides the Neo4jGraphStore class that wraps the Neo4j async driver
to manage knowledge graph nodes and relationships for Semantic Memory.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase


class StorageConnectionError(Exception):
    """Raised when a connection to the Neo4j storage backend fails.

    Attributes:
        uri: The Neo4j URI that was attempted.
        message: Human-readable error description.
        retry_guidance: Suggested steps to resolve the connection issue.
    """

    def __init__(self, uri: str, message: str, retry_guidance: str | None = None) -> None:
        self.uri = uri
        self.retry_guidance = retry_guidance or (
            "Verify that Neo4j is running and accessible at the configured URI. "
            "Check network connectivity and authentication credentials."
        )
        full_message = f"Neo4j connection failed (uri={uri}): {message}. {self.retry_guidance}"
        super().__init__(full_message)


class RelationshipType(str, Enum):
    """Valid relationship types for the knowledge graph."""

    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    EXTENDS = "EXTENDS"
    USES = "USES"
    RELATED_TO = "RELATED_TO"


# Mapping from user-facing lowercase names to enum values
_RELATIONSHIP_MAP: dict[str, RelationshipType] = {
    "depends_on": RelationshipType.DEPENDS_ON,
    "implements": RelationshipType.IMPLEMENTS,
    "extends": RelationshipType.EXTENDS,
    "uses": RelationshipType.USES,
    "related_to": RelationshipType.RELATED_TO,
}

VALID_RELATIONSHIP_TYPES: frozenset[str] = frozenset(_RELATIONSHIP_MAP.keys())


def _resolve_relationship_type(rel_type: str) -> str:
    """Resolve a user-facing relationship type string to the Neo4j label.

    Args:
        rel_type: Relationship type string (case-insensitive).

    Returns:
        The uppercase Neo4j relationship type label.

    Raises:
        ValueError: If the relationship type is not recognized.
    """
    normalized = rel_type.strip().lower()
    if normalized not in _RELATIONSHIP_MAP:
        raise ValueError(
            f"Invalid relationship type: {rel_type!r}. "
            f"Must be one of: {sorted(VALID_RELATIONSHIP_TYPES)}"
        )
    return _RELATIONSHIP_MAP[normalized].value


class Neo4jGraphStore:
    """Async graph storage implementation backed by Neo4j.

    Provides node and relationship management for the knowledge graph used
    by Semantic Memory. Uses the neo4j async driver for non-blocking I/O.

    Configuration is read from environment variables:
        - NEO4J_URI: Connection URI (default: bolt://localhost:7687)
        - NEO4J_USERNAME: Authentication username (default: neo4j)
        - NEO4J_PASSWORD: Authentication password (default: coding-agent-password)
        - NEO4J_DATABASE: Database name (default: neo4j)
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        """Initialize the Neo4j graph store.

        Args:
            uri: Neo4j connection URI. Falls back to NEO4J_URI env var.
            username: Authentication username. Falls back to NEO4J_USERNAME env var.
            password: Authentication password. Falls back to NEO4J_PASSWORD env var.
            database: Target database name. Falls back to NEO4J_DATABASE env var.
        """
        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "coding-agent-password")
        self._database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        self._driver: AsyncDriver | None = None

    async def _get_driver(self) -> AsyncDriver:
        """Get or create the async Neo4j driver.

        Returns:
            The active AsyncDriver instance.

        Raises:
            StorageConnectionError: If the driver cannot be created.
        """
        if self._driver is None:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self._uri,
                    auth=(self._username, self._password),
                )
            except Exception as exc:
                raise StorageConnectionError(
                    uri=self._uri,
                    message=str(exc),
                ) from exc
        return self._driver

    async def close(self) -> None:
        """Close the Neo4j driver and release resources."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def create_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Create or merge a node in the knowledge graph.

        Uses MERGE to ensure idempotent node creation. If a node with the
        given ID already exists, its properties are updated.

        Args:
            node_id: Unique identifier for the node.
            properties: Dictionary of node properties to set.

        Returns:
            True if the node was created or updated successfully.

        Raises:
            StorageConnectionError: If the database is unreachable.
        """
        driver = await self._get_driver()
        query = (
            "MERGE (n:KnowledgeNode {node_id: $node_id}) "
            "SET n += $properties "
            "RETURN n"
        )
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, node_id=node_id, properties=properties)
                await result.consume()
                return True
        except Exception as exc:
            raise StorageConnectionError(
                uri=self._uri,
                message=f"Failed to create node {node_id!r}: {exc}",
            ) from exc

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a typed relationship between two nodes.

        Both source and target nodes are created (via MERGE) if they do not
        already exist. The relationship is also merged to avoid duplicates.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            rel_type: Relationship type (e.g., "depends_on", "implements").
            properties: Optional dictionary of relationship properties.

        Returns:
            True if the relationship was created or updated successfully.

        Raises:
            ValueError: If rel_type is not a valid relationship type.
            StorageConnectionError: If the database is unreachable.
        """
        neo4j_rel_type = _resolve_relationship_type(rel_type)
        props = properties or {}

        driver = await self._get_driver()
        # Use dynamic relationship type via APOC-free approach with string formatting
        # for the relationship type (safe because it's validated via enum).
        query = (
            "MERGE (a:KnowledgeNode {node_id: $source_id}) "
            "MERGE (b:KnowledgeNode {node_id: $target_id}) "
            f"MERGE (a)-[r:{neo4j_rel_type}]->(b) "
            "SET r += $properties "
            "RETURN r"
        )
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    properties=props,
                )
                await result.consume()
                return True
        except Exception as exc:
            raise StorageConnectionError(
                uri=self._uri,
                message=(
                    f"Failed to create relationship {source_id!r} "
                    f"-[{rel_type}]-> {target_id!r}: {exc}"
                ),
            ) from exc

    async def get_neighbors(
        self,
        node_id: str,
        rel_type: str | None = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes connected to the given node.

        Traverses outgoing relationships from the specified node up to the
        given depth. Optionally filters by relationship type.

        Args:
            node_id: ID of the starting node.
            rel_type: Optional relationship type filter. If None, all types
                are traversed.
            depth: Maximum traversal depth (default 1).

        Returns:
            A list of dictionaries, each containing:
                - node_id: The neighbor's ID.
                - properties: The neighbor's properties.
                - relationship_type: The type of relationship connecting them.
                - depth: The traversal depth at which this neighbor was found.

        Raises:
            ValueError: If rel_type is provided but not valid.
            StorageConnectionError: If the database is unreachable.
        """
        if rel_type is not None:
            neo4j_rel_type = _resolve_relationship_type(rel_type)
            rel_pattern = f":{neo4j_rel_type}"
        else:
            rel_pattern = ""

        driver = await self._get_driver()
        query = (
            f"MATCH path = (start:KnowledgeNode {{node_id: $node_id}})"
            f"-[r{rel_pattern}*1..{depth}]->(neighbor:KnowledgeNode) "
            "WITH neighbor, relationships(path) AS rels, length(path) AS d "
            "RETURN DISTINCT neighbor.node_id AS neighbor_id, "
            "properties(neighbor) AS props, "
            "type(last(rels)) AS rel_type, "
            "d AS depth"
        )
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, node_id=node_id)
                records = [record async for record in result]
                neighbors: list[dict[str, Any]] = []
                for record in records:
                    props = dict(record["props"]) if record["props"] else {}
                    # Remove internal node_id from properties to avoid duplication
                    props.pop("node_id", None)
                    neighbors.append(
                        {
                            "node_id": record["neighbor_id"],
                            "properties": props,
                            "relationship_type": record["rel_type"],
                            "depth": record["depth"],
                        }
                    )
                return neighbors
        except Exception as exc:
            raise StorageConnectionError(
                uri=self._uri,
                message=f"Failed to get neighbors for node {node_id!r}: {exc}",
            ) from exc

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships from the graph.

        Performs a cascade delete: removes the node and any relationships
        connected to it (both incoming and outgoing).

        Args:
            node_id: ID of the node to delete.

        Returns:
            True if the node was found and deleted, False if not found.

        Raises:
            StorageConnectionError: If the database is unreachable.
        """
        driver = await self._get_driver()
        query = (
            "MATCH (n:KnowledgeNode {node_id: $node_id}) "
            "DETACH DELETE n "
            "RETURN count(n) AS deleted_count"
        )
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                if record is None:
                    return False
                return bool(record["deleted_count"] > 0)
        except Exception as exc:
            raise StorageConnectionError(
                uri=self._uri,
                message=f"Failed to delete node {node_id!r}: {exc}",
            ) from exc

    async def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> bool:
        """Delete a specific relationship between two nodes.

        Only removes the relationship; the nodes themselves are preserved.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            rel_type: The relationship type to delete.

        Returns:
            True if the relationship was found and deleted, False if not found.

        Raises:
            ValueError: If rel_type is not a valid relationship type.
            StorageConnectionError: If the database is unreachable.
        """
        neo4j_rel_type = _resolve_relationship_type(rel_type)

        driver = await self._get_driver()
        query = (
            "MATCH (a:KnowledgeNode {node_id: $source_id})"
            f"-[r:{neo4j_rel_type}]->"
            "(b:KnowledgeNode {node_id: $target_id}) "
            "DELETE r "
            "RETURN count(r) AS deleted_count"
        )
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                )
                record = await result.single()
                if record is None:
                    return False
                return bool(record["deleted_count"] > 0)
        except Exception as exc:
            raise StorageConnectionError(
                uri=self._uri,
                message=(
                    f"Failed to delete relationship {source_id!r} "
                    f"-[{rel_type}]-> {target_id!r}: {exc}"
                ),
            ) from exc

    async def health_check(self) -> bool:
        """Check connectivity to the Neo4j database.

        Executes a lightweight query to verify the database is reachable
        and responsive.

        Returns:
            True if the database is healthy and reachable, False otherwise.
        """
        try:
            driver = await self._get_driver()
            async with driver.session(database=self._database) as session:
                result = await session.run("RETURN 1 AS ping")
                record = await result.single()
                return record is not None and record["ping"] == 1
        except Exception:
            return False
