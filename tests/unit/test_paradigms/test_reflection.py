"""Unit tests for the Reflection paradigm."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.reflection import (
    ReflectionParadigm,
    _is_lgtm,
)


# ---------------------------------------------------------------------------
# Tests for _is_lgtm
# ---------------------------------------------------------------------------


class TestIsLgtm:
    """Tests for the _is_lgtm helper."""

    def test_exact_lgtm(self) -> None:
        assert _is_lgtm("LGTM") is True

    def test_lowercase_lgtm(self) -> None:
        assert _is_lgtm("lgtm") is True

    def test_lgtm_with_punctuation(self) -> None:
        assert _is_lgtm("lgtm!") is True

    def test_lgtm_in_short_text(self) -> None:
        assert _is_lgtm("Looks good, lgtm") is True

    def test_no_issues_found(self) -> None:
        assert _is_lgtm("No issues found") is True

    def test_no_significant_issues(self) -> None:
        assert _is_lgtm("No significant issues") is True

    def test_detailed_feedback_is_not_lgtm(self) -> None:
        assert _is_lgtm(
            "The code has several issues: 1. Missing error handling, 2. No input validation"
        ) is False

    def test_empty_string(self) -> None:
        assert _is_lgtm("") is False

    def test_lgtm_embedded_in_long_text(self) -> None:
        # Long text with LGTM somewhere should NOT be treated as approval
        long_text = "I reviewed the code carefully. " * 10 + "lgtm"
        assert _is_lgtm(long_text) is False


# ---------------------------------------------------------------------------
# Tests for ReflectionParadigm.run
# ---------------------------------------------------------------------------


class TestReflectionParadigmRun:
    """Tests for the Reflection paradigm execution."""

    async def test_lgtm_on_first_review(self) -> None:
        """Reviewer approves on first try — no refinement needed."""
        mock_llm = AsyncMock()
        # ReAct phase: immediate answer
        # Then review: LGTM
        mock_llm.chat_completion = AsyncMock(side_effect=[
            '{"answer": "Initial answer"}',  # ReAct
            'LGTM',  # Review
        ])

        registry = ActionRegistry()
        paradigm = ReflectionParadigm(mock_llm, registry)

        result = await paradigm.run("task")
        assert result.answer == "Initial answer"
        # Should have draft and review in trace
        trace_types = [s["type"] for s in result.reasoning_trace]
        assert "draft" in trace_types
        assert "review" in trace_types

    async def test_refine_once_then_lgtm(self) -> None:
        """Reviewer finds issues, agent refines, then reviewer approves."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            '{"answer": "Draft v1"}',  # ReAct
            'Missing edge case handling',  # Review (not LGTM)
            'Draft v2 with edge cases handled',  # Refine
            'LGTM',  # Second review
        ])

        registry = ActionRegistry()
        paradigm = ReflectionParadigm(mock_llm, registry)

        result = await paradigm.run("task")
        assert result.answer == "Draft v2 with edge cases handled"
        trace_types = [s["type"] for s in result.reasoning_trace]
        assert "refinement" in trace_types

    async def test_max_reflection_iterations(self) -> None:
        """Reviewer never approves — reaches max iterations."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            '{"answer": "Draft"}',       # ReAct
            'Issue 1',                    # Review 1
            'Refined v2',                 # Refine 1
            'Issue 2',                    # Review 2
            'Refined v3',                 # Refine 2
            'Issue 3',                    # Review 3
            'Refined v4',                 # Refine 3 (max reached)
        ])

        registry = ActionRegistry()
        paradigm = ReflectionParadigm(mock_llm, registry)

        result = await paradigm.run("task")
        assert result.answer == "Refined v4"
        # Should have 3 reviews and 3 refinements
        reviews = [s for s in result.reasoning_trace if s["type"] == "review"]
        refinements = [s for s in result.reasoning_trace if s["type"] == "refinement"]
        assert len(reviews) == 3
        assert len(refinements) == 3

    async def test_context_passed_to_react(self) -> None:
        """Context is forwarded to the ReAct phase."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            '{"answer": "ok"}',
            'LGTM',
        ])

        registry = ActionRegistry()
        paradigm = ReflectionParadigm(mock_llm, registry)

        await paradigm.run("task", context="some context")

        # First call should be the ReAct system prompt
        first_call = mock_llm.chat_completion.call_args_list[0][0][0]
        user_msg = [m for m in first_call if m.role == "user"][0]
        assert "some context" in user_msg.content

    async def test_react_with_action_then_reflect(self) -> None:
        """ReAct phase uses actions, then reflection reviews."""
        from coding_agents.core.actions.read_file import ReadFileAction

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            # ReAct: action then answer
            'Thought: read file\n{"action": "read_file", "params": {"path": "x"}}',
            '{"answer": "File says hello"}',
            # Review
            'LGTM',
        ])

        registry = ActionRegistry()
        registry.register(ReadFileAction())
        paradigm = ReflectionParadigm(mock_llm, registry)

        result = await paradigm.run("Read file x")
        assert result.answer == "File says hello"
