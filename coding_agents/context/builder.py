"""Context Builder implementing the GSSC pipeline for the Cognitive Coding Agent.

This module provides the ``ContextBuilder`` class which executes the four-stage
GSSC (Gather-Select-Structure-Compress) pipeline to construct optimal context
for LLM consumption. It integrates with the ``MemoryManager`` for cross-memory
retrieval and uses tiktoken for accurate token counting.

Public API:
    - ``ContextResult``: Dataclass holding the pipeline output.
    - ``ContextBuilder``: Main class orchestrating the GSSC pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from coding_agents.memory.base import MemoryItem
from coding_agents.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# Minimum and maximum configurable token budgets
MIN_TOKEN_BUDGET: int = 1000
MAX_TOKEN_BUDGET: int = 128000

# Fallback character-to-token ratio when tiktoken is unavailable
_CHARS_PER_TOKEN: int = 4

# High-importance threshold — items at or above this are never removed during compression
_HIGH_IMPORTANCE_THRESHOLD: float = 0.8


def _load_tiktoken_encoding() -> Any | None:
    """Attempt to load the tiktoken cl100k_base encoding.

    Returns:
        The tiktoken encoding object, or None if tiktoken is unavailable.
    """
    try:
        import tiktoken  # noqa: WPS433

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning(
            "tiktoken unavailable; falling back to character-count estimation."
        )
        return None


# Module-level encoding instance (loaded once)
_ENCODING: Any | None = _load_tiktoken_encoding()


def _count_tokens(text: str) -> int:
    """Count the number of tokens in *text*.

    Uses tiktoken with the cl100k_base model when available; otherwise
    falls back to a character-count estimation (4 chars ≈ 1 token).

    Args:
        text: The string to tokenize.

    Returns:
        Estimated or exact token count.
    """
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    # Fallback: character-count estimation
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class ContextResult:
    """Result of the GSSC context-building pipeline.

    Attributes:
        content: Structured context string ready for LLM consumption.
        token_count: Actual token count of *content*.
        sources: Source attribution list. Each entry is a dict with keys
            ``memory_type``, ``item_id``, and ``score``.
        items_included: Number of memory items included in the context.
    """

    content: str
    token_count: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    items_included: int = 0


class ContextBuilder:
    """Orchestrates the GSSC pipeline to build optimal LLM context.

    The pipeline stages execute in strict order:
        1. **Gather** — collect candidate items from all memory types.
        2. **Select** — rank by Relevance_Score and pick top items within
           the token budget.
        3. **Structure** — organize into a coherent format with section
           headers and source attribution.
        4. **Compress** — reduce to fit within the target token limit,
           preserving high-importance items.

    Args:
        memory_manager: An initialized ``MemoryManager`` instance used
            for cross-memory search during the Gather stage.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        """Initialize the ContextBuilder.

        Args:
            memory_manager: The ``MemoryManager`` providing cross-memory
                search capabilities.
        """
        self._memory_manager = memory_manager

    async def build(self, query: str, max_tokens: int = 4096) -> ContextResult:
        """Execute the full GSSC pipeline and return a ``ContextResult``.

        Args:
            query: The user query to build context for.
            max_tokens: Maximum token budget for the final context.
                Must be between 1000 and 128000.

        Returns:
            A ``ContextResult`` containing the structured, compressed
            context along with metadata.
        """
        # Clamp token budget to valid range
        max_tokens = max(MIN_TOKEN_BUDGET, min(MAX_TOKEN_BUDGET, max_tokens))

        # Stage 1: Gather
        candidates = await self.gather(query)

        if not candidates:
            return ContextResult(content="", token_count=0, sources=[], items_included=0)

        # Stage 2: Select
        selected = self.select(candidates, token_budget=max_tokens)

        if not selected:
            return ContextResult(content="", token_count=0, sources=[], items_included=0)

        # Stage 3: Structure
        structured = self.structure(selected)

        # Stage 4: Compress
        compressed = self.compress(structured, max_tokens=max_tokens)

        # Build source attribution
        sources: list[dict[str, Any]] = [
            {
                "memory_type": item.memory_type,
                "item_id": item.id,
                "score": score,
            }
            for item, score in selected
        ]

        token_count = _count_tokens(compressed)

        return ContextResult(
            content=compressed,
            token_count=token_count,
            sources=sources,
            items_included=len(selected),
        )

    async def gather(self, query: str) -> list[tuple[MemoryItem, float]]:
        """Gather candidate memory items from all memory types.

        Delegates to ``MemoryManager.cross_memory_search`` to collect
        candidates across all available (non-degraded) memory subsystems.

        Args:
            query: The search query string.

        Returns:
            A list of ``(MemoryItem, relevance_score)`` tuples.
        """
        return await self._memory_manager.cross_memory_search(query)

    def select(
        self,
        candidates: list[tuple[MemoryItem, float]],
        token_budget: int,
    ) -> list[tuple[MemoryItem, float]]:
        """Select top candidates within the token budget.

        Sorts candidates by descending Relevance_Score and greedily adds
        items until the next item would exceed the token budget.

        Args:
            candidates: List of ``(MemoryItem, score)`` tuples from the
                Gather stage.
            token_budget: Maximum number of tokens allowed for the
                selected items' content.

        Returns:
            A subset of candidates fitting within the token budget,
            ordered by descending score.
        """
        # Sort by descending score; ties broken by higher importance
        sorted_candidates = sorted(
            candidates, key=lambda x: (-x[1], -x[0].importance)
        )

        selected: list[tuple[MemoryItem, float]] = []
        tokens_used = 0

        for item, score in sorted_candidates:
            item_tokens = _count_tokens(item.content)
            if tokens_used + item_tokens > token_budget:
                break
            selected.append((item, score))
            tokens_used += item_tokens

        return selected

    def structure(self, selected: list[tuple[MemoryItem, float]]) -> str:
        """Organize selected items into a coherent structured format.

        Groups items by memory type and formats each with a section header
        and source attribution.

        Format::

            ## Context from [memory_type] Memory
            [Source: item_id | Score: 0.85]
            <content>

            ---

        Args:
            selected: List of ``(MemoryItem, score)`` tuples to structure.

        Returns:
            A formatted string with section headers and source attribution.
        """
        if not selected:
            return ""

        # Group items by memory type while preserving order within groups
        groups: dict[str, list[tuple[MemoryItem, float]]] = {}
        for item, score in selected:
            groups.setdefault(item.memory_type, []).append((item, score))

        parts: list[str] = []
        for memory_type, items in groups.items():
            parts.append(f"## Context from {memory_type} Memory")
            for item, score in items:
                parts.append(f"[Source: {item.id} | Score: {score:.2f}]")
                parts.append(item.content)
                parts.append("")
                parts.append("---")

        return "\n".join(parts)

    def compress(self, structured: str, max_tokens: int) -> str:
        """Compress structured context to fit within the token limit.

        If the structured content already fits, it is returned as-is.
        Otherwise, iteratively removes the lowest-scored items from the
        end until the content fits. Items with importance >= 0.8 are
        never removed (high-importance preservation).

        Args:
            structured: The structured context string from the Structure
                stage.
            max_tokens: Maximum token count for the output.

        Returns:
            A string that fits within *max_tokens*.
        """
        if _count_tokens(structured) <= max_tokens:
            return structured

        # Parse the structured content back into sections for selective removal.
        # Each section is delimited by "---" and starts with a source line.
        sections = self._parse_sections(structured)

        if not sections:
            # Cannot parse — truncate as last resort
            return self._truncate_to_tokens(structured, max_tokens)

        # Sort sections by score ascending (lowest first) for removal priority,
        # but never remove high-importance items.
        removable = [
            s for s in sections if s["importance"] < _HIGH_IMPORTANCE_THRESHOLD
        ]
        # Sort removable by score ascending (remove lowest-scored first)
        removable.sort(key=lambda s: s["score"])

        # Iteratively remove lowest-scored sections until within budget
        removed_ids: set[int] = set()
        for section in removable:
            current_text = self._rebuild_from_sections(sections, removed_ids)
            if _count_tokens(current_text) <= max_tokens:
                return current_text
            removed_ids.add(section["index"])

        # After removing all removable sections, rebuild
        result = self._rebuild_from_sections(sections, removed_ids)
        if _count_tokens(result) <= max_tokens:
            return result

        # Final fallback: truncate to fit
        return self._truncate_to_tokens(result, max_tokens)

    @staticmethod
    def _parse_sections(structured: str) -> list[dict[str, Any]]:
        """Parse structured text into individual sections with metadata.

        Each section is expected to have the format:
            [Source: <id> | Score: <score>]
            <content>

        Args:
            structured: The full structured context string.

        Returns:
            A list of dicts with keys: index, text, score, importance, id.
        """
        lines = structured.split("\n")
        sections: list[dict[str, Any]] = []
        current_section_lines: list[str] = []
        current_score: float = 0.0
        current_id: str = ""
        in_section = False
        section_index = 0

        for line in lines:
            if line.startswith("[Source:") and "|" in line:
                # Start of a new item section
                if in_section and current_section_lines:
                    sections.append({
                        "index": section_index,
                        "text": "\n".join(current_section_lines),
                        "score": current_score,
                        "importance": current_score,  # Use score as proxy
                        "id": current_id,
                    })
                    section_index += 1
                    current_section_lines = []

                in_section = True
                # Parse score from "[Source: <id> | Score: <score>]"
                try:
                    parts = line.split("|")
                    id_part = parts[0].replace("[Source:", "").strip()
                    score_part = parts[1].replace("Score:", "").replace("]", "").strip()
                    current_score = float(score_part)
                    current_id = id_part
                except (IndexError, ValueError):
                    current_score = 0.0
                    current_id = ""
                current_section_lines.append(line)
            elif line.strip() == "---":
                # End of section
                if in_section and current_section_lines:
                    sections.append({
                        "index": section_index,
                        "text": "\n".join(current_section_lines),
                        "score": current_score,
                        "importance": current_score,
                        "id": current_id,
                    })
                    section_index += 1
                    current_section_lines = []
                    in_section = False
            elif line.startswith("## Context from"):
                # Section header — include in current section if active,
                # otherwise store as a standalone header
                if in_section and current_section_lines:
                    sections.append({
                        "index": section_index,
                        "text": "\n".join(current_section_lines),
                        "score": current_score,
                        "importance": current_score,
                        "id": current_id,
                    })
                    section_index += 1
                    current_section_lines = []
                # Headers are not removable — give them max importance
                sections.append({
                    "index": section_index,
                    "text": line,
                    "score": 1.0,
                    "importance": 1.0,
                    "id": "__header__",
                })
                section_index += 1
                in_section = False
            else:
                if in_section:
                    current_section_lines.append(line)

        # Flush remaining section
        if in_section and current_section_lines:
            sections.append({
                "index": section_index,
                "text": "\n".join(current_section_lines),
                "score": current_score,
                "importance": current_score,
                "id": current_id,
            })

        return sections

    @staticmethod
    def _rebuild_from_sections(
        sections: list[dict[str, Any]], removed_ids: set[int]
    ) -> str:
        """Rebuild structured text from sections, excluding removed ones.

        Args:
            sections: All parsed sections.
            removed_ids: Set of section indices to exclude.

        Returns:
            Rebuilt structured text.
        """
        parts: list[str] = []
        for section in sections:
            if section["index"] not in removed_ids:
                parts.append(section["text"])
                # Add separator after content sections (not headers)
                if not section["text"].startswith("## Context from"):
                    parts.append("---")
        return "\n".join(parts)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens as a last resort.

        Args:
            text: The text to truncate.
            max_tokens: Target token limit.

        Returns:
            Truncated text fitting within the limit.
        """
        if _ENCODING is not None:
            try:
                tokens = _ENCODING.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                result: str = _ENCODING.decode(tokens[:max_tokens])
                return result
            except Exception:
                pass
        # Fallback: character-based truncation
        max_chars = max_tokens * _CHARS_PER_TOKEN
        return text[:max_chars]


__all__ = ["ContextBuilder", "ContextResult"]
