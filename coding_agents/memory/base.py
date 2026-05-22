"""Base infrastructure for the Cognitive Coding Agent memory system.

This module defines the core data structures, abstract base classes, and scoring
utilities shared across all memory types in the Quad-Memory Architecture.
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# Valid memory type literals
VALID_MEMORY_TYPES: frozenset[str] = frozenset({"working", "episodic", "semantic", "perceptual"})

# Maximum content length in characters
MAX_CONTENT_LENGTH: int = 100_000


class MemoryValidationError(ValueError):
    """Raised when a MemoryItem fails validation."""


@dataclass
class MemoryItem:
    """Unified memory data structure shared across all memory types.

    Attributes:
        id: UUID v4 unique identifier.
        content: Memory content (max 100,000 characters).
        metadata: Extensible metadata dictionary.
        importance: Importance score in [0.0, 1.0], default 0.5.
        created_at: Creation timestamp in UTC.
        last_accessed_at: Last access timestamp in UTC.
        memory_type: One of "working", "episodic", "semantic", "perceptual".
    """

    id: str
    content: str
    metadata: dict[str, Any]
    importance: float
    created_at: datetime
    last_accessed_at: datetime
    memory_type: Literal["working", "episodic", "semantic", "perceptual"]

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Run all validation checks on the item fields."""
        # UUID format check
        try:
            uuid.UUID(self.id, version=4)
        except (ValueError, AttributeError) as e:
            raise MemoryValidationError(f"Invalid UUID v4 format: {self.id!r}") from e

        # Content length check
        if len(self.content) > MAX_CONTENT_LENGTH:
            raise MemoryValidationError(
                f"Content length {len(self.content)} exceeds maximum "
                f"of {MAX_CONTENT_LENGTH} characters"
            )

        # Importance range check
        if not (0.0 <= self.importance <= 1.0):
            raise MemoryValidationError(f"Importance {self.importance} must be in range [0.0, 1.0]")

        # Memory type check
        if self.memory_type not in VALID_MEMORY_TYPES:
            raise MemoryValidationError(
                f"Invalid memory_type {self.memory_type!r}. "
                f"Must be one of: {sorted(VALID_MEMORY_TYPES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the MemoryItem to a dictionary.

        Returns:
            A dictionary representation suitable for JSON serialization.
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "memory_type": self.memory_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Deserialize a MemoryItem from a dictionary.

        Args:
            data: Dictionary with MemoryItem fields. Timestamps should be
                ISO 8601 formatted strings.

        Returns:
            A new MemoryItem instance.

        Raises:
            MemoryValidationError: If the data fails validation.
            KeyError: If required fields are missing.
        """
        created_at = data["created_at"]
        last_accessed_at = data["last_accessed_at"]

        # Parse ISO format strings to datetime
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(last_accessed_at, str):
            last_accessed_at = datetime.fromisoformat(last_accessed_at)

        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data["metadata"],
            importance=data["importance"],
            created_at=created_at,
            last_accessed_at=last_accessed_at,
            memory_type=data["memory_type"],
        )


@dataclass
class MemoryConfig:
    """System configuration for the memory subsystem.

    Attributes:
        working_memory_ttl: Default TTL in seconds for working memory items.
        decay_rate: Exponential decay rate. Default produces 50% decay after 24h.
        embedding_backend: Primary embedding backend name.
        qdrant_host: Qdrant server hostname.
        qdrant_port: Qdrant server port.
        neo4j_uri: Neo4j connection URI.
        sqlite_path: Path to the SQLite database file.
        max_token_budget: Default context token budget.
    """

    working_memory_ttl: int = 3600
    decay_rate: float = field(default_factory=lambda: math.log(2) / 86400)
    embedding_backend: str = "dashscope"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "bolt://localhost:7687"
    sqlite_path: str = "./memory.db"
    max_token_budget: int = 4096


class BaseMemory(ABC):
    """Abstract base class for all memory type implementations.

    Defines the common interface that Working, Episodic, Semantic, and
    Perceptual memory types must implement.
    """

    @abstractmethod
    async def store(self, item: MemoryItem) -> bool:
        """Store a memory item.

        Args:
            item: The MemoryItem to store.

        Returns:
            True if storage was successful, False otherwise.
        """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[tuple[MemoryItem, float]]:
        """Retrieve memory items relevant to a query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of (MemoryItem, relevance_score) tuples ordered by
            descending relevance score.
        """

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete a memory item by its ID.

        Args:
            item_id: The UUID of the item to delete.

        Returns:
            True if deletion was successful, False if item was not found.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Remove all items from this memory store."""


class ScoringMixin:
    """Mixin class providing unified scoring computations for memory retrieval.

    All memory types share these scoring functions to ensure cross-memory
    retrieval scores are comparable.
    """

    @staticmethod
    def compute_time_decay(elapsed_seconds: float, decay_rate: float | None = None) -> float:
        """Compute exponential time decay factor.

        Uses the formula: exp(-decay_rate * elapsed_seconds)

        Args:
            elapsed_seconds: Time elapsed since last access in seconds.
                Must be non-negative.
            decay_rate: Decay rate parameter. Defaults to ln(2)/86400
                which produces 50% decay after 24 hours.

        Returns:
            A float in the range (0.0, 1.0] representing the time decay factor.
        """
        if decay_rate is None:
            decay_rate = math.log(2) / 86400
        return math.exp(-decay_rate * elapsed_seconds)

    @staticmethod
    def compute_relevance_score(
        similarity: float,
        time_factor: float,
        importance: float,
        sim_weight: float = 1.0,
        time_weight: float = 0.0,
    ) -> float:
        """Compute the unified relevance score for a memory item.

        Formula: (similarity * sim_weight + time_factor * time_weight)
                 * (0.8 + importance * 0.4)

        The result is clamped to [0.0, 1.0].

        Args:
            similarity: Similarity score in [0, 1] (e.g., cosine similarity).
            time_factor: Time decay factor in [0, 1].
            importance: Item importance score in [0, 1].
            sim_weight: Weight for the similarity component.
            time_weight: Weight for the time factor component.

        Returns:
            A float in [0.0, 1.0] representing the final relevance score.
        """
        raw_score = (similarity * sim_weight + time_factor * time_weight) * (0.8 + importance * 0.4)
        return max(0.0, min(1.0, raw_score))


def create_memory_item(
    content: str,
    memory_type: Literal["working", "episodic", "semantic", "perceptual"],
    metadata: dict[str, Any] | None = None,
    importance: float = 0.5,
) -> MemoryItem:
    """Factory function to create a new MemoryItem with auto-generated fields.

    Args:
        content: The memory content string.
        memory_type: The type of memory to create.
        metadata: Optional metadata dictionary. Defaults to empty dict.
        importance: Importance score in [0.0, 1.0]. Defaults to 0.5.

    Returns:
        A new MemoryItem with a generated UUID and current UTC timestamps.
    """
    now = datetime.now(timezone.utc)
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=content,
        metadata=metadata if metadata is not None else {},
        importance=importance,
        created_at=now,
        last_accessed_at=now,
        memory_type=memory_type,
    )
