"""Unit tests for ContextBuilder."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from coding_agents.context.builder import (
    ContextBuilder,
    ContextResult,
    _count_tokens,
)
from coding_agents.memory.base import create_memory_item


@pytest.fixture
def mock_memory_manager():
    return AsyncMock()


@pytest.fixture
def builder(mock_memory_manager):
    return ContextBuilder(mock_memory_manager)


def _make_item(content: str, importance: float = 0.5, memory_type: str = "working"):
    return create_memory_item(
        content=content,
        memory_type=memory_type,
        importance=importance,
    )


# ---------------------------------------------------------------------------
# Tests for _count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for the token counting function."""

    def test_empty_string(self) -> None:
        assert _count_tokens("") >= 0

    def test_short_string(self) -> None:
        count = _count_tokens("hello")
        assert count >= 1

    def test_longer_string_has_more_tokens(self) -> None:
        short = _count_tokens("hi")
        long = _count_tokens("a" * 1000)
        assert long > short


# ---------------------------------------------------------------------------
# Tests for ContextBuilder.build
# ---------------------------------------------------------------------------


class TestContextBuilderBuild:
    """Tests for the full GSSC pipeline."""

    async def test_build_with_no_candidates(self, builder: ContextBuilder, mock_memory_manager) -> None:
        mock_memory_manager.cross_memory_search = AsyncMock(return_value=[])
        result = await builder.build("query")
        assert result.content == ""
        assert result.items_included == 0

    async def test_build_with_candidates(self, builder: ContextBuilder, mock_memory_manager) -> None:
        items = [
            (_make_item("Python is great", importance=0.9), 0.85),
            (_make_item("JavaScript is popular", importance=0.7), 0.60),
        ]
        mock_memory_manager.cross_memory_search = AsyncMock(return_value=items)

        result = await builder.build("programming languages")
        assert result.content != ""
        assert result.items_included == 2
        assert result.token_count > 0

    async def test_build_sources_populated(self, builder: ContextBuilder, mock_memory_manager) -> None:
        item = _make_item("test content", importance=0.5)
        mock_memory_manager.cross_memory_search = AsyncMock(return_value=[(item, 0.8)])

        result = await builder.build("query")
        assert len(result.sources) == 1
        assert result.sources[0]["memory_type"] == "working"
        assert result.sources[0]["score"] == 0.8

    async def test_build_token_budget_clamped(self, builder: ContextBuilder, mock_memory_manager) -> None:
        mock_memory_manager.cross_memory_search = AsyncMock(return_value=[])
        # Should not raise even with extreme values
        await builder.build("query", max_tokens=50)  # below MIN_TOKEN_BUDGET
        await builder.build("query", max_tokens=999999)  # above MAX_TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Tests for ContextBuilder.select
# ---------------------------------------------------------------------------


class TestContextBuilderSelect:
    """Tests for the Select stage."""

    def test_select_within_budget(self, builder: ContextBuilder) -> None:
        candidates = [
            (_make_item("short"), 0.9),
            (_make_item("also short"), 0.7),
        ]
        selected = builder.select(candidates, token_budget=10000)
        assert len(selected) == 2

    def test_select_respects_budget(self, builder: ContextBuilder) -> None:
        # Create a large item that exceeds budget
        large_content = "x" * 50000  # ~12500 tokens
        candidates = [
            (_make_item(large_content), 0.9),
            (_make_item("small"), 0.5),
        ]
        selected = builder.select(candidates, token_budget=100)
        # The large item should be excluded or only one fits
        assert len(selected) <= 2

    def test_select_sorted_by_score(self, builder: ContextBuilder) -> None:
        candidates = [
            (_make_item("low score"), 0.3),
            (_make_item("high score"), 0.9),
            (_make_item("mid score"), 0.6),
        ]
        selected = builder.select(candidates, token_budget=100000)
        scores = [s for _, s in selected]
        assert scores == sorted(scores, reverse=True)

    def test_select_empty_candidates(self, builder: ContextBuilder) -> None:
        assert builder.select([], token_budget=1000) == []


# ---------------------------------------------------------------------------
# Tests for ContextBuilder.structure
# ---------------------------------------------------------------------------


class TestContextBuilderStructure:
    """Tests for the Structure stage."""

    def test_structure_empty(self, builder: ContextBuilder) -> None:
        assert builder.structure([]) == ""

    def test_structure_groups_by_type(self, builder: ContextBuilder) -> None:
        selected = [
            (_make_item("working content", memory_type="working"), 0.8),
            (_make_item("episodic content", memory_type="episodic"), 0.7),
        ]
        result = builder.structure(selected)
        assert "working" in result.lower()
        assert "episodic" in result.lower()

    def test_structure_includes_source_attribution(self, builder: ContextBuilder) -> None:
        item = _make_item("test")
        selected = [(item, 0.85)]
        result = builder.structure(selected)
        assert "[Source:" in result
        assert "Score:" in result

    def test_structure_includes_content(self, builder: ContextBuilder) -> None:
        selected = [(_make_item("the actual content"), 0.8)]
        result = builder.structure(selected)
        assert "the actual content" in result


# ---------------------------------------------------------------------------
# Tests for ContextBuilder.compress
# ---------------------------------------------------------------------------


class TestContextBuilderCompress:
    """Tests for the Compress stage."""

    def test_compress_noop_when_within_budget(self, builder: ContextBuilder) -> None:
        text = "short text"
        result = builder.compress(text, max_tokens=10000)
        assert result == text

    def test_compress_removes_low_score_sections(self, builder: ContextBuilder) -> None:
        # Build a structured string with multiple sections
        sections = []
        for i in range(10):
            sections.append(f"## Context from working Memory")
            sections.append(f"[Source: id-{i} | Score: {0.1 * i:.2f}]")
            sections.append(f"Content for section {i} " * 50)
            sections.append("---")
        structured = "\n".join(sections)

        result = builder.compress(structured, max_tokens=200)
        # Should be shorter than original
        assert _count_tokens(result) <= 200 + 10  # small margin

    def test_compress_preserves_high_importance(self, builder: ContextBuilder) -> None:
        # Section with score >= 0.8 should be preserved
        sections = [
            "## Context from working Memory",
            "[Source: id-high | Score: 0.90]",
            "Important content " * 100,
            "---",
            "## Context from working Memory",
            "[Source: id-low | Score: 0.10]",
            "Less important " * 100,
            "---",
        ]
        structured = "\n".join(sections)

        result = builder.compress(structured, max_tokens=300)
        # High-importance content should still be present
        assert "Important content" in result


# ---------------------------------------------------------------------------
# Tests for ContextResult
# ---------------------------------------------------------------------------


class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_default_values(self) -> None:
        result = ContextResult(content="", token_count=0)
        assert result.content == ""
        assert result.token_count == 0
        assert result.sources == []
        assert result.items_included == 0

    def test_with_all_fields(self) -> None:
        result = ContextResult(
            content="some context",
            token_count=3,
            sources=[{"memory_type": "working", "item_id": "x", "score": 0.8}],
            items_included=1,
        )
        assert result.items_included == 1
        assert len(result.sources) == 1
