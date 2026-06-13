"""Unit tests for SubAgent message protocol and base classes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from coding_agents.core.agents.base import (
    SubAgentConfig,
    SubAgentResult,
    SubTask,
    _create_isolated_registry,
)
from coding_agents.core.agents.messages import ResultMessage, TaskMessage


# ---------------------------------------------------------------------------
# Tests for TaskMessage
# ---------------------------------------------------------------------------


class TestTaskMessage:
    """Tests for TaskMessage frozen dataclass."""

    def test_creation(self) -> None:
        msg = TaskMessage(task_id=0, description="do stuff", agent_type="coder", context="ctx")
        assert msg.task_id == 0
        assert msg.description == "do stuff"
        assert msg.agent_type == "coder"
        assert msg.context == "ctx"

    def test_frozen(self) -> None:
        msg = TaskMessage(task_id=0, description="x", agent_type="coder", context="")
        with pytest.raises(AttributeError):
            msg.task_id = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests for ResultMessage
# ---------------------------------------------------------------------------


class TestResultMessage:
    """Tests for ResultMessage frozen dataclass."""

    def test_creation(self) -> None:
        result = SubAgentResult(success=True, output="done")
        msg = ResultMessage(task_id=0, result=result)
        assert msg.task_id == 0
        assert msg.result.success is True

    def test_frozen(self) -> None:
        result = SubAgentResult(success=True, output="ok")
        msg = ResultMessage(task_id=0, result=result)
        with pytest.raises(AttributeError):
            msg.task_id = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests for SubAgentResult
# ---------------------------------------------------------------------------


class TestSubAgentResult:
    """Tests for SubAgentResult dataclass."""

    def test_minimal(self) -> None:
        r = SubAgentResult(success=True, output="ok")
        assert r.success is True
        assert r.output == "ok"
        assert r.files_modified == []
        assert r.summary == ""
        assert r.error is None

    def test_failure(self) -> None:
        r = SubAgentResult(success=False, output="", error="something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_with_files_modified(self) -> None:
        r = SubAgentResult(success=True, output="ok", files_modified=["a.py", "b.py"])
        assert len(r.files_modified) == 2


# ---------------------------------------------------------------------------
# Tests for SubAgentConfig
# ---------------------------------------------------------------------------


class TestSubAgentConfig:
    """Tests for SubAgentConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = SubAgentConfig(role="coder", system_prompt="prompt", allowed_tools=["read_file"])
        assert cfg.paradigm == "react"

    def test_custom_paradigm(self) -> None:
        cfg = SubAgentConfig(
            role="planner",
            system_prompt="plan",
            allowed_tools=[],
            paradigm="plan_and_solve",
        )
        assert cfg.paradigm == "plan_and_solve"


# ---------------------------------------------------------------------------
# Tests for SubTask
# ---------------------------------------------------------------------------


class TestSubTask:
    """Tests for SubTask dataclass."""

    def test_defaults(self) -> None:
        st = SubTask(description="do it", agent_type="coder")
        assert st.context == ""
        assert st.depends_on == []

    def test_with_dependencies(self) -> None:
        st = SubTask(description="step 2", agent_type="coder", depends_on=[0, 1])
        assert st.depends_on == [0, 1]


# ---------------------------------------------------------------------------
# Tests for _create_isolated_registry
# ---------------------------------------------------------------------------


class TestCreateIsolatedRegistry:
    """Tests for the isolated registry factory."""

    def test_creates_registry_with_specified_tools(self) -> None:
        registry = _create_isolated_registry(["read_file", "write_file"])
        assert registry.get("read_file") is not None
        assert registry.get("write_file") is not None
        assert registry.get("execute_command") is None

    def test_empty_tools(self) -> None:
        registry = _create_isolated_registry([])
        assert registry.list_schemas() == []

    def test_unknown_tool_ignored(self) -> None:
        registry = _create_isolated_registry(["read_file", "nonexistent"])
        assert registry.get("read_file") is not None
        assert registry.get("nonexistent") is None

    def test_all_tools(self) -> None:
        all_tools = ["read_file", "write_file", "execute_command", "search_code", "list_directory"]
        registry = _create_isolated_registry(all_tools)
        assert len(registry.list_schemas()) == 5
