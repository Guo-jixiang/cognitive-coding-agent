"""SQLite-based document store with full-text search support.

This module provides a persistent document storage backend using SQLite with
FTS5 full-text search capabilities. Documents are stored as JSON-serialized
strings alongside an FTS5 index for efficient text search.
"""

from __future__ import annotations

import json
from typing import Any

import aiosqlite


class StorageConnectionError(ConnectionError):
    """Raised when a storage backend connection fails.

    Provides connection details and retry guidance to help diagnose
    and recover from connection failures.

    Attributes:
        path: The database file path that failed to connect.
        message: Human-readable error description.
        retry_after: Suggested seconds to wait before retrying.
    """

    def __init__(
        self,
        path: str,
        message: str = "Failed to connect to SQLite database",
        retry_after: float = 1.0,
    ) -> None:
        self.path = path
        self.retry_after = retry_after
        full_message = (
            f"{message} (path={path!r}). "
            f"Retry after {retry_after}s. "
            f"Ensure the file path is accessible and not locked by another process."
        )
        super().__init__(full_message)


class SQLiteDocumentStore:
    """Async SQLite document store with FTS5 full-text search.

    Stores documents as JSON-serialized strings in SQLite tables with
    automatic FTS5 index creation for full-text search capabilities.
    Supports being used as an async context manager for proper resource cleanup.

    Example::

        async with SQLiteDocumentStore("./data.db") as store:
            await store.store("notes", "note-1", {"title": "Hello", "body": "World"})
            results = await store.search("notes", "Hello")
    """

    def __init__(self, db_path: str = "./memory.db") -> None:
        """Initialize the document store.

        Args:
            db_path: Path to the SQLite database file. Created if it does not exist.
        """
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._initialized_tables: set[str] = set()

    async def __aenter__(self) -> SQLiteDocumentStore:
        """Open the database connection."""
        await self._connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Close the database connection."""
        await self.close()

    async def _connect(self) -> None:
        """Establish a connection to the SQLite database.

        Raises:
            StorageConnectionError: If the connection cannot be established.
        """
        if self._connection is not None:
            return
        try:
            self._connection = await aiosqlite.connect(self._db_path)
            await self._connection.execute("PRAGMA journal_mode=WAL")
        except Exception as exc:
            raise StorageConnectionError(
                path=self._db_path,
                message=f"Failed to connect to SQLite database: {exc}",
            ) from exc

    async def close(self) -> None:
        """Close the database connection and release resources."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._initialized_tables.clear()

    async def _ensure_table(self, table: str) -> None:
        """Create the document table and FTS5 index if they do not exist.

        Args:
            table: The table name to ensure exists.

        Raises:
            StorageConnectionError: If no active connection exists.
        """
        if table in self._initialized_tables:
            return
        conn = self._get_connection()
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS [{table}] ("
            f"  doc_id TEXT PRIMARY KEY,"
            f"  data TEXT NOT NULL"
            f")"
        )
        # Standalone FTS5 table (not a content table) for reliable full-text search
        await conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS [{table}_fts] "
            f"USING fts5(doc_id UNINDEXED, data)"
        )
        await conn.commit()
        self._initialized_tables.add(table)

    def _get_connection(self) -> aiosqlite.Connection:
        """Return the active connection or raise if not connected.

        Returns:
            The active aiosqlite connection.

        Raises:
            StorageConnectionError: If no connection is active.
        """
        if self._connection is None:
            raise StorageConnectionError(
                path=self._db_path,
                message="No active database connection. Use 'async with' or call _connect()",
            )
        return self._connection

    async def store(self, table: str, doc_id: str, data: dict[str, Any]) -> bool:
        """Store a document in the specified table.

        If a document with the same ID already exists, it will be replaced.

        Args:
            table: The table name to store the document in.
            doc_id: Unique document identifier.
            data: Document data as a dictionary (will be JSON-serialized).

        Returns:
            True if the document was stored successfully.
        """
        await self._ensure_table(table)
        conn = self._get_connection()
        json_data = json.dumps(data, ensure_ascii=False)
        # Remove old FTS entry if replacing
        await conn.execute(
            f"DELETE FROM [{table}_fts] WHERE doc_id = ?",
            (doc_id,),
        )
        await conn.execute(
            f"INSERT OR REPLACE INTO [{table}] (doc_id, data) VALUES (?, ?)",
            (doc_id, json_data),
        )
        await conn.execute(
            f"INSERT INTO [{table}_fts] (doc_id, data) VALUES (?, ?)",
            (doc_id, json_data),
        )
        await conn.commit()
        return True

    async def get(self, table: str, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document by its ID.

        Args:
            table: The table name to search in.
            doc_id: The document identifier to look up.

        Returns:
            The document data as a dictionary, or None if not found.
        """
        await self._ensure_table(table)
        conn = self._get_connection()
        cursor = await conn.execute(
            f"SELECT data FROM [{table}] WHERE doc_id = ?",
            (doc_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])  # type: ignore[no-any-return]

    async def search(self, table: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search documents using FTS5 full-text search.

        Args:
            table: The table name to search in.
            query: The search query string (FTS5 query syntax supported).
            top_k: Maximum number of results to return.

        Returns:
            A list of matching documents as dictionaries, ordered by relevance.
        """
        await self._ensure_table(table)
        conn = self._get_connection()
        # Join FTS results back to main table to get the canonical data
        cursor = await conn.execute(
            f"SELECT [{table}].data FROM [{table}_fts] "
            f"JOIN [{table}] ON [{table}_fts].doc_id = [{table}].doc_id "
            f"WHERE [{table}_fts] MATCH ? "
            f"ORDER BY rank "
            f"LIMIT ?",
            (query, top_k),
        )
        rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    async def delete(self, table: str, doc_id: str) -> bool:
        """Delete a document by its ID.

        Args:
            table: The table name to delete from.
            doc_id: The document identifier to delete.

        Returns:
            True if the document was deleted, False if it was not found.
        """
        await self._ensure_table(table)
        conn = self._get_connection()
        cursor = await conn.execute(
            f"DELETE FROM [{table}] WHERE doc_id = ?",
            (doc_id,),
        )
        if cursor.rowcount > 0:
            await conn.execute(
                f"DELETE FROM [{table}_fts] WHERE doc_id = ?",
                (doc_id,),
            )
            await conn.commit()
            return True
        await conn.commit()
        return False

    async def list_all(self, table: str) -> list[dict[str, Any]]:
        """List all documents in a table.

        Args:
            table: The table name to list documents from.

        Returns:
            A list of all documents as dictionaries.
        """
        await self._ensure_table(table)
        conn = self._get_connection()
        cursor = await conn.execute(f"SELECT data FROM [{table}]")
        rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]
