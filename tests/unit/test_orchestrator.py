"""Unit tests for the Orchestrator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agents.core.agents.base import SubAgentResult, SubTask
from coding_agents.core.agents.orchestrator import Orchestrator


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def mock_memory_manager():
    mm = AsyncMock()
    mm.store = AsyncMock(return_value=MagicMock(id="mem-001"))
    return mm


@pytest.fixture
def mock_context_builder():
    cb = AsyncMock()
    cb.build = AsyncMock(return_value=MagicMock(content="ctx", items_included=1))
    return cb


@pytest.fixture
def orchestrator(mock_llm, mock_memory_manager, mock_context_builder):
    return Orchestrator(
        llm_client=mock_llm,
        memory_manager=mock_memory_manager,
        context_builder=mock_context_builder,
    )


# ---------------------------------------------------------------------------
# Tests for _is_simple_message
# ---------------------------------------------------------------------------


class TestIsSimpleMessage:
    """Tests for simple message detection."""

    def test_hello(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("hello") is True

    def test_hi(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("Hi") is True

    def test_thanks(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("Thanks!") is True

    def test_bye(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("bye.") is True

    def test_ok(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("ok") is True

    def test_complex_message_not_simple(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("Write a Python function to sort a list") is False

    def test_empty_string(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("") is False

    def test_whitespace_handling(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._is_simple_message("  hello  ") is True


# ---------------------------------------------------------------------------
# Tests for _parse_subtasks_json
# ---------------------------------------------------------------------------


class TestParseSubtasksJson:
    """Tests for JSON parsing of subtasks."""

    def test_valid_json_array(self, orchestrator: Orchestrator) -> None:
        response = json.dumps([
            {"description": "Analyze code", "agent_type": "analyzer", "context": "", "depends_on": []},
            {"description": "Write code", "agent_type": "coder", "context": "", "depends_on": [0]},
        ])
        subtasks = orchestrator._parse_subtasks_json(response)
        assert len(subtasks) == 2
        assert subtasks[0].agent_type == "analyzer"
        assert subtasks[1].depends_on == [0]

    def test_markdown_wrapped_json(self, orchestrator: Orchestrator) -> None:
        response = '```json\n[{"description": "task", "agent_type": "coder"}]\n```'
        subtasks = orchestrator._parse_subtasks_json(response)
        assert len(subtasks) == 1

    def test_invalid_json_returns_empty(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._parse_subtasks_json("not json") == []

    def test_empty_array(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._parse_subtasks_json("[]") == []

    def test_invalid_agent_type_defaults_to_researcher(self, orchestrator: Orchestrator) -> None:
        response = json.dumps([{"description": "task", "agent_type": "invalid_type"}])
        subtasks = orchestrator._parse_subtasks_json(response)
        assert len(subtasks) == 1
        assert subtasks[0].agent_type == "researcher"

    def test_missing_description_skipped(self, orchestrator: Orchestrator) -> None:
        response = json.dumps([{"agent_type": "coder"}])
        subtasks = orchestrator._parse_subtasks_json(response)
        assert len(subtasks) == 0

    def test_non_dict_items_skipped(self, orchestrator: Orchestrator) -> None:
        response = json.dumps(["not a dict", {"description": "valid", "agent_type": "coder"}])
        subtasks = orchestrator._parse_subtasks_json(response)
        assert len(subtasks) == 1

    def test_depends_on_validated(self, orchestrator: Orchestrator) -> None:
        response = json.dumps([
            {"description": "a", "agent_type": "coder", "depends_on": [0, 5, -1]},
        ])
        subtasks = orchestrator._parse_subtasks_json(response)
        # Only index 0 is valid (out-of-range and negative are filtered)
        assert subtasks[0].depends_on == [0] or subtasks[0].depends_on == []

    def test_json_with_surrounding_text(self, orchestrator: Orchestrator) -> None:
        response = 'Here are the tasks:\n[{"description": "do it", "agent_type": "coder"}]\nDone.'
        subtasks = orchestrator._parse_subtasks_json(response)
        assert len(subtasks) == 1


# ---------------------------------------------------------------------------
# Tests for _reflect_on_results
# ---------------------------------------------------------------------------


class TestReflectOnResults:
    """Tests for the reflection/synthesis step."""

    async def test_single_successful_result_returns_directly(self, orchestrator: Orchestrator) -> None:
        results = [SubAgentResult(success=True, output="the answer")]
        answer = await orchestrator._reflect_on_results("task", results)
        assert answer == "the answer"

    async def test_no_successful_results(self, orchestrator: Orchestrator) -> None:
        results = [SubAgentResult(success=False, output="", error="failed")]
        answer = await orchestrator._reflect_on_results("task", results)
        assert "unable" in answer.lower() or "error" in answer.lower()

    async def test_multiple_results_calls_llm(self, orchestrator: Orchestrator, mock_llm) -> None:
        mock_llm.invoke = AsyncMock(return_value="synthesized answer")
        results = [
            SubAgentResult(success=True, output="result 1"),
            SubAgentResult(success=True, output="result 2"),
        ]
        answer = await orchestrator._reflect_on_results("task", results)
        assert answer == "synthesized answer"
        mock_llm.invoke.assert_awaited_once()

    async def test_reflection_failure_falls_back(self, orchestrator: Orchestrator, mock_llm) -> None:
        mock_llm.invoke = AsyncMock(side_effect=Exception("LLM fail"))
        results = [
            SubAgentResult(success=True, output="result A"),
            SubAgentResult(success=True, output="result B"),
        ]
        answer = await orchestrator._reflect_on_results("task", results)
        assert "result A" in answer
        assert "result B" in answer


# ---------------------------------------------------------------------------
# Tests for Orchestrator.run (full lifecycle)
# ---------------------------------------------------------------------------


class TestOrchestratorRun:
    """Tests for the full orchestration loop."""

    async def test_simple_message_bypasses_decomposition(
        self, orchestrator: Orchestrator, mock_llm
    ) -> None:
        mock_llm.invoke = AsyncMock(return_value="Hello there!")
        result = await orchestrator.run("hello")
        assert result.answer == "Hello there!"
        # Should not have called decomposition
        assert len(result.reasoning_trace) == 1
        assert result.reasoning_trace[0]["type"] == "thought"

    async def test_simple_message_llm_failure_returns_default(
        self, orchestrator: Orchestrator, mock_llm
    ) -> None:
        mock_llm.invoke = AsyncMock(side_effect=Exception("fail"))
        result = await orchestrator.run("hi")
        assert "help" in result.answer.lower() or "hello" in result.answer.lower()
