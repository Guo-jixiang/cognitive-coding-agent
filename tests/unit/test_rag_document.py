"""Unit tests for DocumentProcessor (RAG document processing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agents.memory.rag.document import (
    Document,
    DocumentProcessor,
    DocumentTooLargeError,
    UnsupportedFormatError,
)


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor()


@pytest.fixture
def sample_py(tmp_path: Path) -> Path:
    p = tmp_path / "sample.py"
    p.write_text(
        '"""Sample module."""\n\n'
        "def hello():\n    print('hello')\n\n"
        "class Foo:\n    pass\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "sample.md"
    p.write_text(
        "# Title\n\nSome intro text.\n\n## Section 1\n\nDetails here.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    p = tmp_path / "sample.json"
    p.write_text('{"name": "test", "version": "1.0"}', encoding="utf-8")
    return p


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "sample.yaml"
    p.write_text("name: test\nversion: 1.0\n", encoding="utf-8")
    return p


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("Plain text content.", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests for parse
# ---------------------------------------------------------------------------


class TestDocumentProcessorParse:
    """Tests for the parse method."""

    def test_parse_python(self, processor: DocumentProcessor, sample_py: Path) -> None:
        doc = processor.parse(str(sample_py))
        assert doc.metadata["language"] == "python"
        assert "hello" in doc.metadata.get("definitions", [])
        assert "Foo" in doc.metadata.get("definitions", [])

    def test_parse_markdown(self, processor: DocumentProcessor, sample_md: Path) -> None:
        doc = processor.parse(str(sample_md))
        assert doc.metadata["language"] == "markdown"
        assert "Title" in doc.metadata["section_headers"]
        assert "Section 1" in doc.metadata["section_headers"]

    def test_parse_json(self, processor: DocumentProcessor, sample_json: Path) -> None:
        doc = processor.parse(str(sample_json))
        assert doc.metadata["language"] == "json"
        assert "name" in doc.metadata.get("json_keys", [])
        assert "version" in doc.metadata.get("json_keys", [])

    def test_parse_yaml(self, processor: DocumentProcessor, sample_yaml: Path) -> None:
        doc = processor.parse(str(sample_yaml))
        assert doc.metadata["language"] == "yaml"
        assert "name" in doc.metadata.get("yaml_keys", [])

    def test_parse_text(self, processor: DocumentProcessor, sample_txt: Path) -> None:
        doc = processor.parse(str(sample_txt))
        assert doc.metadata["language"] == "text"
        assert doc.content == "Plain text content."

    def test_parse_nonexistent_raises(self, processor: DocumentProcessor) -> None:
        with pytest.raises(FileNotFoundError):
            processor.parse("/nonexistent/file.py")

    def test_parse_unsupported_format(self, processor: DocumentProcessor, tmp_path: Path) -> None:
        p = tmp_path / "file.xyz"
        p.write_text("content", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            processor.parse(str(p))

    def test_parse_file_path_in_metadata(self, processor: DocumentProcessor, sample_txt: Path) -> None:
        doc = processor.parse(str(sample_txt))
        assert "file_path" in doc.metadata

    def test_supported_formats(self, processor: DocumentProcessor) -> None:
        formats = processor.supported_formats()
        assert ".py" in formats
        assert ".md" in formats
        assert ".json" in formats
        assert ".yaml" in formats
        assert ".txt" in formats


# ---------------------------------------------------------------------------
# Tests for chunk
# ---------------------------------------------------------------------------


class TestDocumentProcessorChunk:
    """Tests for the chunk method."""

    def test_chunk_basic(self, processor: DocumentProcessor) -> None:
        doc = Document(content="a" * 2500, metadata={"language": "text"})
        chunks = processor.chunk(doc, chunk_size=1000, overlap_percent=0.1)
        assert len(chunks) >= 2

    def test_chunk_empty_content(self, processor: DocumentProcessor) -> None:
        doc = Document(content="", metadata={})
        assert processor.chunk(doc) == []

    def test_chunk_single_chunk(self, processor: DocumentProcessor) -> None:
        doc = Document(content="short", metadata={})
        chunks = processor.chunk(doc, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].content == "short"

    def test_chunk_metadata_populated(self, processor: DocumentProcessor) -> None:
        doc = Document(content="x" * 2500, metadata={"language": "text"})
        chunks = processor.chunk(doc, chunk_size=1000, overlap_percent=0.1)
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == len(chunks)

    def test_chunk_invalid_size_raises(self, processor: DocumentProcessor) -> None:
        doc = Document(content="x", metadata={})
        with pytest.raises(ValueError, match="chunk_size"):
            processor.chunk(doc, chunk_size=50)

    def test_chunk_invalid_overlap_raises(self, processor: DocumentProcessor) -> None:
        doc = Document(content="x", metadata={})
        with pytest.raises(ValueError, match="overlap_percent"):
            processor.chunk(doc, chunk_size=1000, overlap_percent=0.9)


# ---------------------------------------------------------------------------
# Tests for reassemble
# ---------------------------------------------------------------------------


class TestDocumentProcessorReassemble:
    """Tests for the reassemble method."""

    def test_reassemble_round_trip(self, processor: DocumentProcessor) -> None:
        content = "The quick brown fox jumps over the lazy dog. " * 50
        doc = Document(content=content, metadata={})
        chunks = processor.chunk(doc, chunk_size=500, overlap_percent=0.1)
        reassembled = processor.reassemble(chunks)
        assert reassembled == content

    def test_reassemble_single_chunk(self, processor: DocumentProcessor) -> None:
        doc = Document(content="single chunk", metadata={})
        chunks = processor.chunk(doc, chunk_size=1000)
        assert processor.reassemble(chunks) == "single chunk"

    def test_reassemble_empty(self, processor: DocumentProcessor) -> None:
        assert processor.reassemble([]) == ""

    def test_reassemble_with_overlap(self, processor: DocumentProcessor) -> None:
        content = "abcdefghij" * 100
        doc = Document(content=content, metadata={})
        chunks = processor.chunk(doc, chunk_size=200, overlap_percent=0.2)
        reassembled = processor.reassemble(chunks)
        assert reassembled == content


# ---------------------------------------------------------------------------
# Tests for format-specific parsing
# ---------------------------------------------------------------------------


class TestFormatParsing:
    """Tests for format-specific metadata extraction."""

    def test_python_definitions(self, processor: DocumentProcessor) -> None:
        result = processor._parse_python("def foo():\n    pass\nclass Bar:\n    pass\n")
        assert "foo" in result["definitions"]
        assert "Bar" in result["definitions"]

    def test_python_no_definitions(self, processor: DocumentProcessor) -> None:
        result = processor._parse_python("x = 1\ny = 2\n")
        assert result == {}

    def test_json_valid(self, processor: DocumentProcessor) -> None:
        result = processor._parse_json('{"a": 1, "b": 2}', "test.json")
        assert "a" in result["json_keys"]
        assert "b" in result["json_keys"]

    def test_json_invalid(self, processor: DocumentProcessor) -> None:
        result = processor._parse_json("not json", "test.json")
        assert result == {}

    def test_json_array(self, processor: DocumentProcessor) -> None:
        result = processor._parse_json("[1, 2, 3]", "test.json")
        assert result == {}  # arrays don't have keys

    def test_yaml_valid(self, processor: DocumentProcessor) -> None:
        result = processor._parse_yaml("a: 1\nb: 2\n", "test.yaml")
        assert "a" in result["yaml_keys"]
        assert "b" in result["yaml_keys"]

    def test_yaml_invalid(self, processor: DocumentProcessor) -> None:
        result = processor._parse_yaml("{{invalid", "test.yaml")
        assert result == {}

    def test_markdown_headers(self, processor: DocumentProcessor) -> None:
        result = processor._extract_section_headers("# H1\n## H2\n### H3\n", "markdown")
        assert "H1" in result
        assert "H2" in result
        assert "H3" in result

    def test_non_markdown_no_headers(self, processor: DocumentProcessor) -> None:
        assert processor._extract_section_headers("# not a header", "python") == []


# ---------------------------------------------------------------------------
# Tests for error classes
# ---------------------------------------------------------------------------


class TestDocumentErrors:
    """Tests for custom error classes."""

    def test_unsupported_format_error(self) -> None:
        err = UnsupportedFormatError(".xyz", [".py", ".md"])
        assert ".xyz" in str(err)
        assert err.format_name == ".xyz"

    def test_document_too_large_error(self) -> None:
        err = DocumentTooLargeError("big.txt", 20_000_000, 10_000_000)
        assert "big.txt" in str(err)
        assert err.file_size == 20_000_000
