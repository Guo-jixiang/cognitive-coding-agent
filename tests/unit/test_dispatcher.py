"""Unit tests for SubAgentDispatcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agents.core.agents.base import SubAgentResult, SubTask
from coding_agents.core.agents.dispatcher import AGENT_MAP, SubAgentDispatcher
from coding_agents.core.agents.messages import ResultMessage, TaskMessage


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def dispatcher(mock_llm):
    return SubAgentDispatcher(mock_llm)


# ---------------------------------------------------------------------------
# Tests for AGENT_MAP
# ---------------------------------------------------------------------------


class TestAgentMap:
    """Tests for the AGENT_MAP registry."""

    def test_all_expected_agents_registered(self) -> None:
        expected = {"planner", "coder", "reviewer", "tester", "analyzer", "debugger", "executor", "researcher"}
        assert set(AGENT_MAP.keys()) == expected

    def test_agent_map_values_are_classes(self) -> None:
        for name, cls in AGENT_MAP.items():
            assert isinstance(cls, type), f"{name} is not a class"


# ---------------------------------------------------------------------------
# Tests for SubAgentDispatcher.dispatch
# ---------------------------------------------------------------------------


class TestDispatcherDispatch:
    """Tests for the dispatch method."""

    async def test_single_task_no_dependencies(self, dispatcher: SubAgentDispatcher, mock_llm) -> None:
        """Single task with no dependencies should execute successfully."""
        subtasks = [SubTask(description="analyze code", agent_type="researcher")]

        # Mock the agent run
        mock_result = SubAgentResult(success=True, output="analysis done", summary="done")

        with patch.object(dispatcher, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ResultMessage(task_id=0, result=mock_result)
            results = await dispatcher.dispatch(subtasks)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "analysis done"

    async def test_two_independent_tasks_parallel(self, dispatcher: SubAgentDispatcher) -> None:
        """Two tasks with no dependencies run in the same wave."""
        subtasks = [
            SubTask(description="task A", agent_type="researcher"),
            SubTask(description="task B", agent_type="analyzer"),
        ]

        results_list = [
            ResultMessage(task_id=0, result=SubAgentResult(success=True, output="A done")),
            ResultMessage(task_id=1, result=SubAgentResult(success=True, output="B done")),
        ]

        call_count = 0

        async def mock_run_agent(msg: TaskMessage) -> ResultMessage:
            nonlocal call_count
            result = results_list[msg.task_id]
            call_count += 1
            return result

        with patch.object(dispatcher, "_run_agent", side_effect=mock_run_agent):
            results = await dispatcher.dispatch(subtasks)

        assert len(results) == 2
        assert results[0].output == "A done"
        assert results[1].output == "B done"
        assert call_count == 2

    async def test_dependent_tasks_sequential_waves(self, dispatcher: SubAgentDispatcher) -> None:
        """Task B depends on task A — they run in separate waves."""
        subtasks = [
            SubTask(description="first", agent_type="researcher"),
            SubTask(description="second", agent_type="coder", depends_on=[0]),
        ]

        async def mock_run_agent(msg: TaskMessage) -> ResultMessage:
            if msg.task_id == 0:
                return ResultMessage(task_id=0, result=SubAgentResult(success=True, output="first done"))
            return ResultMessage(task_id=1, result=SubAgentResult(success=True, output="second done"))

        with patch.object(dispatcher, "_run_agent", side_effect=mock_run_agent):
            results = await dispatcher.dispatch(subtasks)

        assert results[0].output == "first done"
        assert results[1].output == "second done"

    async def test_unresolvable_dependencies(self, dispatcher: SubAgentDispatcher) -> None:
        """Tasks with circular or out-of-range dependencies get failure results."""
        subtasks = [
            SubTask(description="A", agent_type="researcher", depends_on=[1]),
            SubTask(description="B", agent_type="researcher", depends_on=[0]),
        ]

        results = await dispatcher.dispatch(subtasks)
        # At least one should fail due to unresolved dependencies
        assert all(r.success is False for r in results)

    async def test_agent_exception_produces_error_result(self, dispatcher: SubAgentDispatcher) -> None:
        """If an agent raises, the dispatcher catches it and returns an error result."""
        subtasks = [SubTask(description="crash", agent_type="researcher")]

        async def mock_run_agent(msg: TaskMessage) -> ResultMessage:
            raise RuntimeError("agent crashed")

        with patch.object(dispatcher, "_run_agent", side_effect=mock_run_agent):
            results = await dispatcher.dispatch(subtasks)

        assert len(results) == 1
        assert results[0].success is False
        assert "crashed" in results[0].error

    async def test_unknown_agent_type(self, dispatcher: SubAgentDispatcher) -> None:
        """Unknown agent type produces an error result."""
        subtasks = [SubTask(description="task", agent_type="nonexistent")]

        # Use the real _run_agent which handles unknown types
        results = await dispatcher.dispatch(subtasks)
        assert len(results) == 1
        assert results[0].success is False
        assert "Unknown agent type" in results[0].error

    async def test_result_queue_receives_messages(self, dispatcher: SubAgentDispatcher) -> None:
        """Results are also pushed to the shared result queue."""
        subtasks = [SubTask(description="task", agent_type="researcher")]

        async def mock_run_agent(msg: TaskMessage) -> ResultMessage:
            return ResultMessage(task_id=0, result=SubAgentResult(success=True, output="ok"))

        with patch.object(dispatcher, "_run_agent", side_effect=mock_run_agent):
            await dispatcher.dispatch(subtasks)

        assert not dispatcher.result_queue.empty()
        msg = await dispatcher.result_queue.get()
        assert msg.task_id == 0

    async def test_empty_subtasks(self, dispatcher: SubAgentDispatcher) -> None:
        """Empty subtask list returns empty results."""
        results = await dispatcher.dispatch([])
        assert results == []
