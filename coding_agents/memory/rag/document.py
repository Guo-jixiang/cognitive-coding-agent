"""Document processing module for the RAG pipeline.

This module provides multi-format document parsing, intelligent chunking with
configurable overlap, and lossless reassembly. It supports Markdown, Python,
JSON, YAML, and plain text file formats.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Maximum document size in bytes (10 MB)
MAX_DOCUMENT_SIZE: int = 10 * 1024 * 1024

# Valid chunk size range
MIN_CHUNK_SIZE: int = 100
MAX_CHUNK_SIZE: int = 10_000

# Valid overlap range
MIN_OVERLAP: float = 0.0
MAX_OVERLAP: float = 0.5

# Supported file extensions mapped to language names
_EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".md": "markdown",
    ".py": "python",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".txt": "text",
}


class UnsupportedFormatError(ValueError):
    """Raised when a document format is not supported.

    Attributes:
        format_name: The unsupported format that was encountered.
        supported_formats: List of formats that are supported.
    """

    def __init__(self, format_name: str, supported_formats: list[str]) -> None:
        """Initialize with the unsupported format and list of supported formats.

        Args:
            format_name: The file extension or format name that is not supported.
            supported_formats: List of supported file extensions.
        """
        self.format_name = format_name
        self.supported_formats = supported_formats
        super().__init__(
            f"Unsupported document format: '{format_name}'. "
            f"Supported formats: {supported_formats}"
        )


class DocumentTooLargeError(ValueError):
    """Raised when a document exceeds the maximum allowed size.

    Attributes:
        file_path: Path to the oversized document.
        file_size: Actual size of the document in bytes.
        max_size: Maximum allowed size in bytes.
    """

    def __init__(self, file_path: str, file_size: int, max_size: int) -> None:
        """Initialize with file details and size limits.

        Args:
            file_path: Path to the document that is too large.
            file_size: Actual file size in bytes.
            max_size: Maximum allowed file size in bytes.
        """
        self.file_path = file_path
        self.file_size = file_size
        self.max_size = max_size
        super().__init__(
            f"Document too large: '{file_path}' is {file_size} bytes, "
            f"maximum allowed is {max_size} bytes (10 MB)"
        )


@dataclass
class Document:
    """Represents a parsed document with content, metadata, and optional chunks.

    Attributes:
        content: The full text content of the document.
        metadata: Dictionary containing file_path, language, and section_headers.
        chunks: List of chunked content segments (populated after chunking).
    """

    content: str
    metadata: dict[str, Any]
    chunks: list[str] = field(default_factory=list)


class DocumentProcessor:
    """Processes multi-format documents for the RAG pipeline.

    Supports parsing, intelligent chunking with configurable overlap,
    and lossless reassembly of document content.
    """

    def parse(self, file_path: str) -> Document:
        """Parse a file and extract its content and metadata.

        Reads the file, validates its format and size, extracts metadata
        including file path, language type (inferred from extension), and
        section headers (for Markdown files).

        Args:
            file_path: Path to the file to parse.

        Returns:
            A Document instance with content and metadata populated.

        Raises:
            UnsupportedFormatError: If the file format is not supported.
            DocumentTooLargeError: If the file exceeds 10 MB.
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: '{file_path}'")

        # Check file extension
        extension = path.suffix.lower()
        if extension not in _EXTENSION_LANGUAGE_MAP:
            raise UnsupportedFormatError(extension, self.supported_formats())

        # Check file size
        file_size = path.stat().st_size
        if file_size > MAX_DOCUMENT_SIZE:
            raise DocumentTooLargeError(file_path, file_size, MAX_DOCUMENT_SIZE)

        # Read file content
        content = path.read_text(encoding="utf-8")

        # Extract metadata
        language = _EXTENSION_LANGUAGE_MAP[extension]
        section_headers = self._extract_section_headers(content, language)

        # Run format-specific parsing/validation
        extra_metadata = self._parse_format(content, language, file_path)

        metadata: dict[str, Any] = {
            "file_path": str(path.resolve()),
            "language": language,
            "section_headers": section_headers,
            **extra_metadata,
        }

        return Document(content=content, metadata=metadata)

    def chunk(
        self,
        document: Document,
        chunk_size: int = 1000,
        overlap_percent: float = 0.1,
    ) -> list[Document]:
        """Split a document into overlapping chunks for processing.

        Uses a sliding window approach where each chunk has a configurable
        overlap with the previous chunk. The algorithm guarantees lossless
        round-trip: reassemble(chunk(doc)) == doc.content.

        Args:
            document: The Document to chunk.
            chunk_size: Size of each chunk in characters (100-10000).
            overlap_percent: Fraction of overlap between consecutive chunks (0.0-0.5).

        Returns:
            A list of Document instances, each containing a chunk of the
            original content. The chunks field of each Document contains
            a single-element list with that chunk's text.

        Raises:
            ValueError: If chunk_size or overlap_percent is out of valid range.
        """
        self._validate_chunk_params(chunk_size, overlap_percent)

        content = document.content

        # Handle empty content
        if not content:
            return []

        # Calculate step size (how much new content each chunk advances)
        overlap_chars = int(chunk_size * overlap_percent)
        step = chunk_size - overlap_chars

        # Ensure step is at least 1 to avoid infinite loops
        if step < 1:
            step = 1

        # Generate chunks using sliding window
        raw_chunks: list[tuple[str, int]] = []  # (chunk_text, overlap_with_previous)
        pos = 0
        chunk_index = 0
        while pos < len(content):
            end = min(pos + chunk_size, len(content))
            chunk_text = content[pos:end]

            # Calculate actual overlap with previous chunk
            if chunk_index == 0:
                actual_overlap = 0
            else:
                actual_overlap = min(overlap_chars, len(chunk_text))

            # Filter out empty chunks with warning
            if chunk_text:
                raw_chunks.append((chunk_text, actual_overlap))
            else:
                logger.warning(
                    "Filtered out empty chunk at position %d in document '%s'",
                    pos,
                    document.metadata.get("file_path", "unknown"),
                )

            # If we've reached the end, stop
            if end >= len(content):
                break

            pos += step
            chunk_index += 1

        # Create Document instances for each chunk
        chunk_documents: list[Document] = []
        for i, (chunk_text, overlap) in enumerate(raw_chunks):
            chunk_metadata: dict[str, Any] = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "overlap_chars": overlap,
            }
            chunk_doc = Document(
                content=chunk_text,
                metadata=chunk_metadata,
                chunks=[chunk_text],
            )
            chunk_documents.append(chunk_doc)

        return chunk_documents

    def reassemble(self, chunks: list[Document]) -> str:
        """Reconstruct original content from chunked documents.

        Takes the first chunk fully, then for each subsequent chunk, skips
        the overlap portion (stored in metadata) and appends only the new
        content. This ensures the round-trip property:
        reassemble(chunk(doc)) == doc.content.

        Args:
            chunks: List of Document instances produced by the chunk() method.

        Returns:
            The reassembled original content string.
        """
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0].content

        # Start with the first chunk's full content
        result = chunks[0].content

        # For subsequent chunks, skip the overlap portion and append new content
        for i in range(1, len(chunks)):
            current_chunk = chunks[i].content
            overlap_chars = chunks[i].metadata.get("overlap_chars", 0)
            # Append only the non-overlapping portion
            result += current_chunk[overlap_chars:]

        return result

    def supported_formats(self) -> list[str]:
        """Return the list of supported file extensions.

        Returns:
            A sorted list of supported file extensions (e.g., ['.json', '.md', ...]).
        """
        return sorted(_EXTENSION_LANGUAGE_MAP.keys())

    def _extract_section_headers(self, content: str, language: str) -> list[str]:
        """Extract section headers from document content.

        For Markdown files, extracts lines starting with '#'.
        For other formats, returns an empty list.

        Args:
            content: The document text content.
            language: The detected language type.

        Returns:
            A list of section header strings.
        """
        if language != "markdown":
            return []

        headers: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                # Extract the header text (remove leading # and whitespace)
                header_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
                if header_match:
                    headers.append(header_match.group(2).strip())
        return headers

    def _parse_format(self, content: str, language: str, file_path: str) -> dict[str, Any]:
        """Run format-specific parsing to extract additional metadata.

        Args:
            content: The raw file content.
            language: The detected language type.
            file_path: Path to the file (for error messages).

        Returns:
            A dictionary of additional metadata extracted from the format.
        """
        if language == "python":
            return self._parse_python(content)
        if language == "json":
            return self._parse_json(content, file_path)
        if language == "yaml":
            return self._parse_yaml(content, file_path)
        return {}

    @staticmethod
    def _parse_python(content: str) -> dict[str, Any]:
        """Extract metadata from Python source files.

        Extracts top-level function and class names.

        Args:
            content: Python source code.

        Returns:
            Dictionary with 'definitions' key listing top-level names.
        """
        definitions: list[str] = []
        for line in content.splitlines():
            # Match top-level def and class statements
            match = re.match(r"^(def|class)\s+(\w+)", line)
            if match:
                definitions.append(match.group(2))
        return {"definitions": definitions} if definitions else {}

    @staticmethod
    def _parse_json(content: str, file_path: str) -> dict[str, Any]:
        """Validate JSON content and extract top-level keys.

        Args:
            content: JSON file content.
            file_path: Path to the file (for error context).

        Returns:
            Dictionary with 'json_keys' if content is a JSON object.
        """
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return {"json_keys": list(data.keys())}
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in file: %s", file_path)
        return {}

    @staticmethod
    def _parse_yaml(content: str, file_path: str) -> dict[str, Any]:
        """Validate YAML content and extract top-level keys.

        Args:
            content: YAML file content.
            file_path: Path to the file (for error context).

        Returns:
            Dictionary with 'yaml_keys' if content is a YAML mapping.
        """
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                return {"yaml_keys": list(data.keys())}
        except yaml.YAMLError:
            logger.warning("Invalid YAML in file: %s", file_path)
        return {}

    def _validate_chunk_params(self, chunk_size: int, overlap_percent: float) -> None:
        """Validate chunking parameters are within allowed ranges.

        Args:
            chunk_size: Chunk size to validate (must be 100-10000).
            overlap_percent: Overlap percentage to validate (must be 0.0-0.5).

        Raises:
            ValueError: If parameters are out of range.
        """
        if not (MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE):
            raise ValueError(
                f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}, "
                f"got {chunk_size}"
            )
        if not (MIN_OVERLAP <= overlap_percent <= MAX_OVERLAP):
            raise ValueError(
                f"overlap_percent must be between {MIN_OVERLAP} and {MAX_OVERLAP}, "
                f"got {overlap_percent}"
            )
