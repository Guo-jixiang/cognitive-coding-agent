"""Unit tests for the ReAct paradigm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.react import (
    ReActParadigm,
    _build_actions_description,
    _parse_llm_output,
)
from coding_agents.llm.client import ChatMessage


# ---------------------------------------------------------------------------
# Tests for _parse_llm_output
# ---------------------------------------------------------------------------


class TestParseLlmOutput:
    """Tests for the _parse_llm_output helper."""

    def test_parse_thought_and_action(self) -> None:
        text = (
            'Thought: I need to read the file\n'
            '{"action": "read_file", "params": {"path": "test.py"}}'
        )
        thought, action_dict, final_answer = _parse_llm_output(text)
        assert thought == "I need to read the file"
        assert action_dict is not None
        assert action_dict["action"] == "read_file"
        assert action_dict["params"]["path"] == "test.py"
        assert final_answer is None

    def test_parse_final_answer(self) -> None:
        text = '{"answer": "The result is 42"}'
        thought, action_dict, final_answer = _parse_llm_output(text)
        assert thought is None
        assert action_dict is None
        assert final_answer == "The result is 42"

    def test_parse_thought_only_with_answer(self) -> None:
        text = 'Thought: I have enough info\n{"answer": "done"}'
        thought, action_dict, final_answer = _parse_llm_output(text)
        assert thought == "I have enough info"
        assert final_answer == "done"
        assert action_dict is None

    def test_parse_no_json_returns_raw(self) -> None:
        text = "Just some plain text response"
        thought, action_dict, final_answer = _parse_llm_output(text)
        assert thought is None
        assert action_dict is None
        assert final_answer is None

    def test_parse_multiline_thought(self) -> None:
        text = (
            'Thought: First line\n'
            'Second line of thought\n'
            '{"action": "search_code", "params": {"pattern": "foo"}}'
        )
        thought, action_dict, _ = _parse_llm_output(text)
        assert "First line" in thought
        assert "Second line" in thought

    def test_parse_invalid_json_returns_none(self) -> None:
        text = "Thought: thinking\n{invalid json here}"
        thought, action_dict, final_answer = _parse_llm_output(text)
        assert thought == "thinking"
        assert action_dict is None
        assert final_answer is None

    def test_parse_answer_in_full_text_json(self) -> None:
        # When JSON is the entire text (no Thought: prefix)
        text = '{"answer": "42"}'
        _, _, final_answer = _parse_llm_output(text)
        assert final_answer == "42"

    def test_parse_action_without_params(self) -> None:
        text = '{"action": "list_directory"}'
        _, action_dict, _ = _parse_llm_output(text)
        assert action_dict is not None
        assert action_dict["action"] == "list_directory"


# ---------------------------------------------------------------------------
# Tests for _build_actions_description
# ---------------------------------------------------------------------------


class TestBuildActionsDescription:
    """Tests for _build_actions_description."""

    def test_empty_registry(self) -> None:
        registry = ActionRegistry()
        desc = _build_actions_description(registry)
        assert "No actions available" in desc

    def test_with_registered_action(self) -> None:
        from coding_agents.core.actions.read_file import ReadFileAction

        registry = ActionRegistry()
        registry.register(ReadFileAction())
        desc = _build_actions_description(registry)
        assert "read_file" in desc
        assert "Read the content" in desc


# ---------------------------------------------------------------------------
# Tests for ReActParadigm.run
# ---------------------------------------------------------------------------


class TestReActParadigmRun:
    """Tests for the ReAct paradigm execution loop."""

    async def test_immediate_final_answer(self) -> None:
        """LLM returns a final answer on the first call."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value='{"answer": "Hello, world!"}'
        )

        registry = ActionRegistry()
        paradigm = ReActParadigm(mock_llm, registry)

        result = await paradigm.run("Say hello")
        assert result.answer == "Hello, world!"
        assert result.iterations == 1

    async def test_action_then_answer(self) -> None:
        """LLM performs one action then returns a final answer."""
        from coding_agents.core.actions.read_file import ReadFileAction

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            'Thought: I need to read a file\n{"action": "read_file", "params": {"path": "test.py"}}',
            '{"answer": "File content found"}',
        ])

        registry = ActionRegistry()
        registry.register(ReadFileAction())
        paradigm = ReActParadigm(mock_llm, registry)

        result = await paradigm.run("Read test.py")
        assert result.answer == "File content found"
        assert result.iterations == 2
        # Should have thought, action, and observation in trace
        trace_types = [step["type"] for step in result.reasoning_trace]
        assert "thought" in trace_types
        assert "action" in trace_types
        assert "observation" in trace_types

    async def test_unknown_action_returns_error_observation(self) -> None:
        """Unknown action produces an error observation."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(side_effect=[
            '{"action": "nonexistent_tool", "params": {}}',
            '{"answer": "done"}',
        ])

        registry = ActionRegistry()
        paradigm = ReActParadigm(mock_llm, registry)

        result = await paradigm.run("Do something")
        # The observation should mention the unknown action
        observations = [
            s for s in result.reasoning_trace if s["type"] == "observation"
        ]
        assert any("Unknown action" in str(o["content"]) for o in observations)

    async def test_max_iterations_returns_fallback(self) -> None:
        """Reaching max iterations returns a fallback message."""
        mock_llm = AsyncMock()
        # Always return an action, never a final answer
        mock_llm.chat_completion = AsyncMock(
            return_value='{"action": "read_file", "params": {"path": "x"}}'
        )

        registry = ActionRegistry()
        from coding_agents.core.actions.read_file import ReadFileAction
        registry.register(ReadFileAction())

        paradigm = ReActParadigm(mock_llm, registry)
        result = await paradigm.run("Never-ending task")
        assert "Maximum iterations" in result.answer
        assert result.iterations == 20

    async def test_context_included_in_user_message(self) -> None:
        """Context is prepended to the user message."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value='{"answer": "ok"}'
        )

        registry = ActionRegistry()
        paradigm = ReActParadigm(mock_llm, registry)

        await paradigm.run("task", context="some context")

        # Check the messages passed to the LLM
        call_args = mock_llm.chat_completion.call_args
        messages = call_args[0][0]
        user_msg = [m for m in messages if m.role == "user"][0]
        assert "some context" in user_msg.content
        assert "task" in user_msg.content

    async def test_plain_text_response_treated_as_answer(self) -> None:
        """If LLM returns plain text (no JSON), it's treated as the answer."""
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value="Here is my plain text answer."
        )

        registry = ActionRegistry()
        paradigm = ReActParadigm(mock_llm, registry)

        result = await paradigm.run("question")
        assert result.answer == "Here is my plain text answer."
        assert result.iterations == 1
