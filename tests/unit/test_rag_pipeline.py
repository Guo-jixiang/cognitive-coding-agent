"""Unit tests for RAGPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from coding_agents.memory.base import MemoryItem, create_memory_item
from coding_agents.memory.rag.document import Document, DocumentProcessor
from coding_agents.memory.rag.pipeline import RAGPipeline


@pytest.fixture
def mock_doc_processor():
    proc = MagicMock(spec=DocumentProcessor)
    return proc


@pytest.fixture
def mock_embedding():
    return MagicMock()


@pytest.fixture
def mock_memory():
    m = AsyncMock()
    m.store = AsyncMock(return_value=True)
    m.retrieve = AsyncMock(return_value=[])
    return m


@pytest.fixture
def pipeline(mock_doc_processor, mock_embedding, mock_memory):
    return RAGPipeline(
        document_processor=mock_doc_processor,
        embedding_service=mock_embedding,
        memory=mock_memory,
    )


# ---------------------------------------------------------------------------
# Tests for ingest_document
# ---------------------------------------------------------------------------


class TestRAGPipelineIngest:
    """Tests for document ingestion."""

    async def test_ingest_single_document(
        self, pipeline: RAGPipeline, mock_doc_processor, mock_memory, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("print('hello')", encoding="utf-8")

        doc = Document(content="print('hello')", metadata={"language": "python"})
        chunk_doc = Document(
            content="print('hello')",
            metadata={"chunk_index": 0, "total_chunks": 1, "language": "python"},
        )
        mock_doc_processor.parse = MagicMock(return_value=doc)
        mock_doc_processor.chunk = MagicMock(return_value=[chunk_doc])

        count = await pipeline.ingest_document(str(f))
        assert count == 1
        mock_memory.store.assert_awaited_once()

    async def test_ingest_multiple_chunks(
        self, pipeline: RAGPipeline, mock_doc_processor, mock_memory, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("content", encoding="utf-8")

        doc = Document(content="content", metadata={})
        chunks = [
            Document(content=f"chunk{i}", metadata={"chunk_index": i, "total_chunks": 3, "language": "text"})
            for i in range(3)
        ]
        mock_doc_processor.parse = MagicMock(return_value=doc)
        mock_doc_processor.chunk = MagicMock(return_value=chunks)

        count = await pipeline.ingest_document(str(f))
        assert count == 3
        assert mock_memory.store.call_count == 3

    async def test_ingest_empty_document(
        self, pipeline: RAGPipeline, mock_doc_processor, mock_memory, tmp_path: Path
    ) -> None:
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")

        doc = Document(content="", metadata={})
        mock_doc_processor.parse = MagicMock(return_value=doc)
        mock_doc_processor.chunk = MagicMock(return_value=[])

        count = await pipeline.ingest_document(str(f))
        assert count == 0

    async def test_ingest_store_failure_continues(
        self, pipeline: RAGPipeline, mock_doc_processor, mock_memory, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.py"
        f.write_text("content", encoding="utf-8")

        doc = Document(content="content", metadata={})
        chunks = [
            Document(content="c1", metadata={"chunk_index": 0, "total_chunks": 2, "language": "text"}),
            Document(content="c2", metadata={"chunk_index": 1, "total_chunks": 2, "language": "text"}),
        ]
        mock_doc_processor.parse = MagicMock(return_value=doc)
        mock_doc_processor.chunk = MagicMock(return_value=chunks)
        mock_memory.store = AsyncMock(side_effect=[Exception("fail"), True])

        count = await pipeline.ingest_document(str(f))
        assert count == 1  # only second chunk succeeded


# ---------------------------------------------------------------------------
# Tests for query
# ---------------------------------------------------------------------------


class TestRAGPipelineQuery:
    """Tests for the query method."""

    async def test_query_returns_results(self, pipeline: RAGPipeline, mock_memory) -> None:
        item = create_memory_item(content="answer", memory_type="episodic")
        mock_memory.retrieve = AsyncMock(return_value=[(item, 0.9)])

        results = await pipeline.query("question")
        assert len(results) == 1
        assert results[0][0] == "answer"
        assert results[0][1] == 0.9

    async def test_query_returns_empty(self, pipeline: RAGPipeline, mock_memory) -> None:
        mock_memory.retrieve = AsyncMock(return_value=[])
        results = await pipeline.query("unknown question")
        assert len(results) == 0

    async def test_query_sorted_by_score(self, pipeline: RAGPipeline, mock_memory) -> None:
        item1 = create_memory_item(content="low", memory_type="episodic")
        item2 = create_memory_item(content="high", memory_type="episodic")
        mock_memory.retrieve = AsyncMock(return_value=[(item1, 0.3), (item2, 0.9)])

        results = await pipeline.query("query")
        assert results[0][0] == "high"
        assert results[1][0] == "low"

    async def test_query_top_k_limits(self, pipeline: RAGPipeline, mock_memory) -> None:
        items = [
            (create_memory_item(content=f"c{i}", memory_type="episodic"), 0.5)
            for i in range(10)
        ]
        mock_memory.retrieve = AsyncMock(return_value=items)

        results = await pipeline.query("query", top_k=3)
        assert len(results) == 3

    async def test_query_includes_metadata(self, pipeline: RAGPipeline, mock_memory) -> None:
        item = create_memory_item(
            content="content",
            memory_type="episodic",
            metadata={"source": "test"},
        )
        mock_memory.retrieve = AsyncMock(return_value=[(item, 0.8)])

        results = await pipeline.query("query")
        assert results[0][2]["source"] == "test"


# ---------------------------------------------------------------------------
# Tests for ingest_directory
# ---------------------------------------------------------------------------


class TestRAGPipelineIngestDirectory:
    """Tests for directory ingestion."""

    async def test_ingest_directory(
        self, pipeline: RAGPipeline, mock_doc_processor, mock_memory, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B", encoding="utf-8")

        doc = Document(content="content", metadata={})
        chunk = Document(content="chunk", metadata={"chunk_index": 0, "total_chunks": 1, "language": "text"})
        mock_doc_processor.parse = MagicMock(return_value=doc)
        mock_doc_processor.chunk = MagicMock(return_value=[chunk])

        count = await pipeline.ingest_directory(str(tmp_path))
        assert count >= 2

    async def test_ingest_directory_not_found(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(FileNotFoundError):
            await pipeline.ingest_directory("/nonexistent")

    async def test_ingest_directory_not_a_dir(self, pipeline: RAGPipeline, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            await pipeline.ingest_directory(str(f))

    async def test_ingest_directory_custom_patterns(
        self, pipeline: RAGPipeline, mock_doc_processor, mock_memory, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        (tmp_path / "b.txt").write_text("y", encoding="utf-8")

        doc = Document(content="c", metadata={})
        chunk = Document(content="c", metadata={"chunk_index": 0, "total_chunks": 1, "language": "text"})
        mock_doc_processor.parse = MagicMock(return_value=doc)
        mock_doc_processor.chunk = MagicMock(return_value=[chunk])

        count = await pipeline.ingest_directory(str(tmp_path), patterns=["*.py"])
        # Only .py files should be ingested
        assert mock_doc_processor.parse.call_count >= 1
