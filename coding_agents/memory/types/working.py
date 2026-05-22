"""Working Memory implementation for the Cognitive Coding Agent.

This module provides a session-scoped, in-memory storage subsystem with
TTL-based auto-cleanup, TF-IDF semantic retrieval, and a keyword-matching
fallback when TF-IDF similarity falls below a configured threshold.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from coding_agents.memory.base import BaseMemory, MemoryItem, ScoringMixin

# TTL boundaries in seconds (1 second to 24 hours)
MIN_TTL_SECONDS: int = 1
MAX_TTL_SECONDS: int = 86_400

# TF-IDF similarity threshold below which the keyword fallback is used
TFIDF_FALLBACK_THRESHOLD: float = 0.1


class WorkingMemory(BaseMemory, ScoringMixin):
    """Session-scoped, in-memory working memory with TTL and TF-IDF retrieval.

    Items are stored in a dictionary keyed by their unique identifier together
    with an expiry timestamp computed from the configured TTL. Retrieval first
    cleans up expired items, then attempts a TF-IDF cosine-similarity search.
    If the top similarity falls below ``TFIDF_FALLBACK_THRESHOLD``, retrieval
    falls back to a simple keyword-overlap score.

    The final relevance score for each candidate uses the Working Memory
    formula::

        relevance = (similarity * time_decay) * (0.8 + importance * 0.4)

    where ``time_decay = exp(-decay_rate * elapsed_seconds_since_last_access)``.

    Attributes:
        ttl: Configured time-to-live in seconds.
        decay_rate: Exponential decay rate used in ``time_decay``.
    """

    def __init__(self, ttl: int = 3600, decay_rate: float | None = None) -> None:
        """Initialize the working memory.

        Args:
            ttl: Time-to-live for stored items, in seconds. Must be in the
                inclusive range [1, 86400].
            decay_rate: Optional exponential decay rate. When ``None`` the
                default ``ln(2) / 86400`` is used, producing 50% decay after
                24 hours of inactivity.

        Raises:
            ValueError: If ``ttl`` falls outside the supported range.
        """
        if not isinstance(ttl, int) or isinstance(ttl, bool):
            raise ValueError(f"TTL must be an int, got {type(ttl).__name__}")
        if ttl < MIN_TTL_SECONDS or ttl > MAX_TTL_SECONDS:
            raise ValueError(
                f"TTL must be in range [{MIN_TTL_SECONDS}, {MAX_TTL_SECONDS}] seconds, got {ttl}"
            )

        self.ttl: int = ttl
        self.decay_rate: float = decay_rate if decay_rate is not None else math.log(2) / 86_400
        self._storage: dict[str, tuple[MemoryItem, datetime]] = {}

    async def store(self, item: MemoryItem) -> bool:
        """Store a memory item with an expiry of ``now + ttl``.

        Storing an item with an existing identifier replaces the previous
        entry and resets the expiry timestamp.

        Args:
            item: The memory item to store.

        Returns:
            ``True`` once the item has been recorded.
        """
        expiry = datetime.now(timezone.utc) + timedelta(seconds=self.ttl)
        self._storage[item.id] = (item, expiry)
        return True

    async def retrieve(self, query: str, top_k: int = 10) -> list[tuple[MemoryItem, float]]:
        """Retrieve the top-K items most relevant to ``query``.

        The retrieval pipeline is:

        1. Remove any expired items.
        2. Compute TF-IDF cosine similarity between the query and each stored
           item.
        3. If the maximum TF-IDF similarity is below
           :data:`TFIDF_FALLBACK_THRESHOLD`, fall back to keyword-overlap
           scoring (matched query words divided by total query words).
        4. Combine each similarity with a time-decay factor and the item's
           importance to produce the final relevance score.
        5. Return the top ``top_k`` results ordered by descending score (with
           higher importance breaking ties), and refresh ``last_accessed_at``
           on each returned item.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(item, relevance_score)`` tuples ordered by descending
            relevance. Items with a non-positive similarity are excluded.
        """
        self._cleanup()

        if not self._storage or top_k <= 0:
            return []

        snapshot = list(self._storage.values())
        contents = [item.content for item, _ in snapshot]

        similarities = self._compute_tfidf_similarities(query, contents)
        max_similarity = max(similarities) if similarities else 0.0
        if max_similarity < TFIDF_FALLBACK_THRESHOLD:
            similarities = self._compute_keyword_similarities(query, contents)

        now = datetime.now(timezone.utc)
        scored: list[tuple[MemoryItem, float]] = []
        for (item, _expiry), similarity in zip(snapshot, similarities, strict=True):
            if similarity <= 0.0:
                continue
            elapsed = (now - item.last_accessed_at).total_seconds()
            elapsed = max(elapsed, 0.0)
            time_decay = math.exp(-self.decay_rate * elapsed)
            raw_score = (similarity * time_decay) * (0.8 + item.importance * 0.4)
            score = max(0.0, min(1.0, raw_score))
            scored.append((item, score))

        # Sort by descending score; on ties, higher importance first.
        scored.sort(key=lambda pair: (-pair[1], -pair[0].importance))
        results = scored[:top_k]

        for item, _score in results:
            item.last_accessed_at = now

        return results

    async def delete(self, item_id: str) -> bool:
        """Remove an item by identifier.

        Args:
            item_id: Unique identifier of the item to remove.

        Returns:
            ``True`` if the item was present and removed, ``False`` otherwise.
        """
        if item_id in self._storage:
            del self._storage[item_id]
            return True
        return False

    async def clear(self) -> None:
        """Remove all stored items."""
        self._storage.clear()

    def _cleanup(self) -> None:
        """Remove items whose expiry timestamp is in the past."""
        now = datetime.now(timezone.utc)
        expired_ids = [
            item_id for item_id, (_item, expiry) in self._storage.items() if expiry <= now
        ]
        for item_id in expired_ids:
            del self._storage[item_id]

    @staticmethod
    def _compute_tfidf_similarities(query: str, contents: list[str]) -> list[float]:
        """Compute TF-IDF cosine similarities between ``query`` and each content.

        Args:
            query: The search query string.
            contents: The corpus of stored item contents.

        Returns:
            A list of cosine similarity values aligned with ``contents``. If
            TF-IDF vectorization fails (e.g., the corpus contains only stop
            words), a list of zeros is returned.
        """
        if not contents:
            return []
        try:
            vectorizer = TfidfVectorizer()
            corpus = [*contents, query]
            matrix = vectorizer.fit_transform(corpus)
            query_vector = matrix[-1]
            document_vectors = matrix[:-1]
            similarities = cosine_similarity(query_vector, document_vectors)
            flattened = cast(Any, similarities).flatten()
            return [float(value) for value in flattened]
        except ValueError:
            # Empty vocabulary, e.g. all stop words.
            return [0.0] * len(contents)

    @staticmethod
    def _compute_keyword_similarities(query: str, contents: list[str]) -> list[float]:
        """Compute keyword-overlap similarity between ``query`` and each content.

        Tokenizes the query on whitespace (lowercased) and counts how many of
        those tokens appear in each content (case-insensitive). The similarity
        for each content is ``matched_tokens / total_query_tokens``.

        Args:
            query: The search query string.
            contents: The corpus of stored item contents.

        Returns:
            A list of similarity values in ``[0.0, 1.0]`` aligned with
            ``contents``.
        """
        tokens = query.lower().split()
        if not tokens:
            return [0.0] * len(contents)
        total = len(tokens)
        results: list[float] = []
        for content in contents:
            lowered = content.lower()
            matched = sum(1 for token in tokens if token in lowered)
            results.append(matched / total)
        return results
