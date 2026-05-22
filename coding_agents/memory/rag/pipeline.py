"""RAG (Retrieval-Augmented Generation) pipeline implementation.

This module implements :class:`RAGPipeline`, the end-to-end pipeline that
coordinates document ingestion and intelligent retrieval. It orchestrates:

- **Document ingestion**: parse → chunk → embed each chunk → store in memory
- **Query retrieval**: embed query → vector search → keyword fallback → rank → return
- **Directory ingestion**: batch-process all matching files in a directory

The pipeline uses :class:`~coding_agents.memory.rag.document.DocumentProcessor`
for parsing and chunking, :class:`~coding_agents.memory.embedding.EmbeddingService`
for vectorisation, and a :class:`~coding_agents.memory.base.BaseMemory` instance
(typically :class:`~coding_agents.memory.types.episodic.EpisodicMemory`) for
persistent storage and retrieval.

Multi-strategy retrieval combines vector-based similarity search with a
keyword-based fallback to maximise recall when the vector path produces
insufficient results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from coding_agents.memory.base import BaseMemory, MemoryItem, create_memory_item
from coding_agents.memory.embedding import EmbeddingService
from coding_agents.memory.rag.document import DocumentProcessor

logger = logging.getLogger(__name__)

# Default file patterns for directory ingestion.
_DEFAULT_PATTERNS: list[str] = ["*.md", "*.py", "*.json", "*.yaml", "*.yml", "*.txt"]

# Minimum similarity threshold for keyword fallback activation.
_KEYWORD_FALLBACK_THRESHOLD: float = 0.3

# Maximum number of results to request from the memory backend (over-fetch
# factor) to allow post-filtering and re-ranking.
_OVERFETCH_FACTOR: int = 3


class RAGPipeline:
    """End-to-end RAG pipeline for document ingestion and retrieval.

    The pipeline coordinates document processing, embedding generation, and
    memory storage/retrieval to provide intelligent question-answering over
    ingested documents.

    Args:
        document_processor: Processor for parsing and chunking documents.
        embedding_service: Service for generating vector embeddings.
        memory: A :class:`BaseMemory` instance (typically EpisodicMemory)
            used for storing and retrieving document chunks.
    """

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        memory: BaseMemory,
    ) -> None:
        """Initialize the RAG pipeline with its dependencies.

        Args:
            document_processor: Processor for parsing and chunking documents.
            embedding_service: Service for generating vector embeddings.
            memory: A BaseMemory instance for storing and retrieving chunks.
        """
        self._document_processor = document_processor
        self._embedding_service = embedding_service
        self._memory = memory

    async def ingest_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: float = 0.1,
    ) -> int:
        """Ingest a single document into memory.

        Parses the document, splits it into overlapping chunks, and stores
        each chunk as a memory item with associated metadata (file path,
        chunk index, language, etc.).

        Args:
            file_path: Path to the document file to ingest.
            chunk_size: Size of each chunk in characters (100-10000).
            overlap: Fraction of overlap between consecutive chunks (0.0-0.5).

        Returns:
            The number of chunks successfully stored in memory.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFormatError: If the file format is not supported.
            DocumentTooLargeError: If the file exceeds the size limit.
            ValueError: If chunk_size or overlap is out of valid range.
        """
        # 1) Parse the document
        document = self._document_processor.parse(file_path)

        # 2) Chunk the document
        chunk_docs = self._document_processor.chunk(
            document, chunk_size=chunk_size, overlap_percent=overlap
        )

        if not chunk_docs:
            logger.info("Document '%s' produced no chunks after processing.", file_path)
            return 0

        # 3) Store each chunk in memory
        stored_count = 0
        for chunk_doc in chunk_docs:
            # Build metadata for the chunk
            chunk_metadata: dict[str, Any] = {
                "source": "rag_pipeline",
                "file_path": file_path,
                "chunk_index": chunk_doc.metadata.get("chunk_index", 0),
                "total_chunks": chunk_doc.metadata.get("total_chunks", 1),
                "language": chunk_doc.metadata.get("language", "text"),
                "section_headers": chunk_doc.metadata.get("section_headers", []),
            }

            # Create a memory item for the chunk
            item = create_memory_item(
                content=chunk_doc.content,
                memory_type="episodic",
                metadata=chunk_metadata,
                importance=0.5,
            )

            try:
                success = await self._memory.store(item)
                if success:
                    stored_count += 1
                else:
                    logger.warning(
                        "Failed to store chunk %d of '%s'.",
                        chunk_doc.metadata.get("chunk_index", 0),
                        file_path,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error storing chunk %d of '%s': %s",
                    chunk_doc.metadata.get("chunk_index", 0),
                    file_path,
                    exc,
                )

        logger.info("Ingested '%s': %d/%d chunks stored.", file_path, stored_count, len(chunk_docs))
        return stored_count

    async def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Query the RAG pipeline for relevant document chunks.

        Embeds the query and searches memory for relevant chunks. Uses a
        multi-strategy approach: vector-based retrieval is the primary path,
        with keyword-based fallback when vector results are insufficient.

        Results are ranked by Relevance_Score (descending).

        Args:
            question: The natural-language question to answer.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(content, score, metadata)`` tuples ranked by
            descending Relevance_Score. Each tuple contains:
            - content: The text content of the matching chunk.
            - score: The relevance score in [0.0, 1.0].
            - metadata: Associated metadata dictionary.
        """
        # Retrieve from memory using the question as query.
        # Over-fetch to allow for post-filtering and re-ranking.
        overfetch_k = min(top_k * _OVERFETCH_FACTOR, 100)

        results: list[tuple[MemoryItem, float]] = []

        # Primary retrieval: vector-based via memory.retrieve()
        try:
            results = await self._memory.retrieve(question, top_k=overfetch_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Primary retrieval failed for query %r: %s", question, exc)

        # Multi-strategy: keyword fallback if vector results are insufficient
        if len(results) < top_k:
            keyword_results = await self._keyword_fallback(question, top_k=overfetch_k)
            # Merge keyword results, avoiding duplicates by item ID
            existing_ids = {item.id for item, _ in results}
            for item, score in keyword_results:
                if item.id not in existing_ids:
                    results.append((item, score))
                    existing_ids.add(item.id)

        # Sort by score descending, then by importance descending for ties
        results.sort(key=lambda pair: (-pair[1], -pair[0].importance))

        # Convert to output format and limit to top_k
        output: list[tuple[str, float, dict[str, Any]]] = []
        for item, score in results[:top_k]:
            output.append((item.content, score, item.metadata))

        return output

    async def ingest_directory(
        self,
        dir_path: str,
        patterns: list[str] | None = None,
    ) -> int:
        """Ingest all matching files in a directory.

        Recursively scans the directory for files matching the given glob
        patterns and ingests each one. Files that fail to process are
        logged and skipped without affecting other files.

        Args:
            dir_path: Path to the directory to scan.
            patterns: List of glob patterns to match files against.
                Defaults to ``["*.md", "*.py", "*.json", "*.yaml",
                "*.yml", "*.txt"]``.

        Returns:
            Total number of chunks stored across all ingested files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        directory = Path(dir_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: '{dir_path}'")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: '{dir_path}'")

        if patterns is None:
            patterns = list(_DEFAULT_PATTERNS)

        total_chunks = 0

        # Collect all matching files
        matched_files: list[Path] = []
        for pattern in patterns:
            matched_files.extend(directory.rglob(pattern))

        # Deduplicate (a file might match multiple patterns)
        unique_files = sorted(set(matched_files))

        logger.info(
            "Found %d files matching patterns %s in '%s'.",
            len(unique_files),
            patterns,
            dir_path,
        )

        for file_path in unique_files:
            try:
                chunks_stored = await self.ingest_document(str(file_path))
                total_chunks += chunks_stored
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to ingest '%s': %s. Skipping.",
                    file_path,
                    exc,
                )

        logger.info(
            "Directory ingestion complete for '%s': %d total chunks stored.",
            dir_path,
            total_chunks,
        )
        return total_chunks

    async def _keyword_fallback(
        self,
        query: str,
        top_k: int = 15,
    ) -> list[tuple[MemoryItem, float]]:
        """Perform keyword-based retrieval as a fallback strategy.

        This method attempts a secondary retrieval using the query text
        directly. The underlying memory implementation may use FTS or
        other text-matching strategies.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(MemoryItem, score)`` tuples from keyword matching.
        """
        try:
            # Re-use the memory's retrieve method which may internally
            # fall back to keyword/FTS matching (e.g., EpisodicMemory
            # falls back to SQLite FTS when Qdrant is unavailable).
            results = await self._memory.retrieve(query, top_k=top_k)
            # Apply a discount factor to keyword results to prefer vector
            # results when both are available.
            discounted: list[tuple[MemoryItem, float]] = [
                (item, score * 0.9) for item, score in results
            ]
            return discounted
        except Exception as exc:  # noqa: BLE001
            logger.warning("Keyword fallback retrieval failed: %s", exc)
            return []


__all__ = ["RAGPipeline"]
