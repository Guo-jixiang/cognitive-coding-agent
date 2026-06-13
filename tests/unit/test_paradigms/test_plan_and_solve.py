"""Unit tests for the Plan-and-Solve paradigm."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.plan_and_solve import (
    PlanAndSolveParadigm,
    _extract_action,
    _parse_plan,
)


# ---------------------------------------------------------------------------
# Tests for _parse_plan
# ---------------------------------------------------------------------------


class TestParsePlan:
    """Tests for _parse_plan."""

    def test_parse_numbered_list_dot(self) -> None:
        text = "1. First step\n2. Second step\n3. Third step"
        steps = _parse_plan(text)
        assert steps == ["First step", "Second step", "Third step"]

    def test_parse_numbered_list_paren(self) -> None:
        text = "1) Alpha\n2) Beta\n3) Gamma"
        steps = _parse_plan(text)
        assert steps == ["Alpha", "Beta", "Gamma"]

    def test_parse_mixed_formatting(self) -> None:
        text = "1.  Step one\n2.   Step two"
        steps = _parse_plan(text)
        assert steps == ["Step one", "Step two"]

    def test_parse_ignores_non_numbered_lines(self) -> None:
        text = "Here is my plan:\n1. Do this\nSome intro text\n2. Do that"
        steps = _parse_plan(text)
        assert steps == ["Do this", "Do that"]

    def test_parse_empty_text(self) -> None:
        assert _parse_plan("") == []

    def test_parse_no_numbered_lines(self) -> None:
        assert _parse_plan("Just plain text") == []


# ---------------------------------------------------------------------------
# Tests for _extract_action
# ---------------------------------------------------------------------------


class TestExtractAction:
    """Tests for _extract_action."""

    def test_extract_action_from_json_line(self) -> None:
        text = 'I will use the tool\n{"action": "read_file", "params": {"path": "x"}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "read_file"

    def test_no_action_returns_none(self) -> None:
        assert _extract_action("Just plain text") is None

    def test_extract_action_full_text_json(self) -> None:
        text = '{"action": "search_code", "params": {"pattern": "foo"}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "search_code"

    def test_extract_ignores_json_without_action_key(self) -> None:
        text = '{"answer": "not an action"}'
        assert _extract_action(text) is None

    def test_extract_ignores_invalid_json(self) -> None:
        text = '{bad json}'
        assert _extract_action(text) is None


# ---------------------------------------------------------------------------
# Tests for PlanAndSolveParadigm.run
# ---------------------------------------------------------------------------


class TestPlanAndSolveRun:
    """Tests for the Plan-and-Solve paradigm execution."""

    async def test_simple_plan_execution(self) -> None:
        """Plan with two steps, no actions needed."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            # Plan generation
            "1. Analyze the problem\n2. Provide solution",
            # Step 1 execution
            "Analysis complete: the problem is clear.",
            # Step 2 execution
            "Solution: use a hash map.",
            # Aggregation
            "The solution is to use a hash map for O(1) lookups.",
        ])

        registry = ActionRegistry()
        paradigm = PlanAndSolveParadigm(mock_llm, registry)

        result = await paradigm.run("Optimize this code")
        assert result.answer == "The solution is to use a hash map for O(1) lookups."
        # iterations = 1 (plan) + 2 (steps) + 1 (aggregate) = 4
        assert result.iterations == 4

    async def test_plan_with_action_execution(self) -> None:
        """Plan step uses an action."""
        from coding_agents.core.actions.read_file import ReadFileAction

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            # Plan
            "1. Read the file\n2. Summarize",
            # Step 1: uses action
            '{"action": "read_file", "params": {"path": "test.py"}}',
            # Step 2: no action
            "The file contains a hello world program.",
            # Aggregation
            "Summary: hello world program in test.py.",
        ])

        registry = ActionRegistry()
        registry.register(ReadFileAction())
        paradigm = PlanAndSolveParadigm(mock_llm, registry)

        result = await paradigm.run("Summarize test.py")
        assert "hello world" in result.answer.lower()

    async def test_empty_plan_falls_back_to_single_step(self) -> None:
        """If plan parsing returns empty, the raw response is used as a single step."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            "Just do it all at once.",  # Plan (unparseable)
            "Done.",  # Step execution
            "Final result.",  # Aggregation
        ])

        registry = ActionRegistry()
        paradigm = PlanAndSolveParadigm(mock_llm, registry)

        result = await paradigm.run("task")
        assert result.answer == "Final result."

    async def test_context_passed_to_plan(self) -> None:
        """Context is included in the plan generation prompt."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            "1. Step one",
            "Result one",
            "Final answer",
        ])

        registry = ActionRegistry()
        paradigm = PlanAndSolveParadigm(mock_llm, registry)

        await paradigm.run("task", context="important context")

        # Check first call includes context
        first_call_messages = mock_llm.chat_completion.call_args_list[0][0][0]
        user_msg = [m for m in first_call_messages if m.role == "user"][0]
        assert "important context" in user_msg.content

    async def test_unknown_action_in_step(self) -> None:
        """Step references an unknown action."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            "1. Use unknown tool",
            '{"action": "nonexistent", "params": {}}',
            "Final answer despite error.",
        ])

        registry = ActionRegistry()
        paradigm = PlanAndSolveParadigm(mock_llm, registry)

        result = await paradigm.run("task")
        # Should still complete
        assert result.answer == "Final answer despite error."
        # Trace should have a step_observation with error
        observations = [
            s for s in result.reasoning_trace if s["type"] == "step_observation"
        ]
        assert len(observations) >= 1
        assert "Unknown action" in str(observations[0]["content"]["observation"])
