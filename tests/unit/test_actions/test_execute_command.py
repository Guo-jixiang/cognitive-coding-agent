"""Unit tests for ExecuteCommandAction."""

from __future__ import annotations

import sys

import pytest

from coding_agents.core.actions.execute_command import ExecuteCommandAction


@pytest.fixture
def action() -> ExecuteCommandAction:
    return ExecuteCommandAction()


class TestExecuteCommandSchema:
    """Tests for the schema definition."""

    def test_schema_name(self, action: ExecuteCommandAction) -> None:
        assert action.schema().name == "execute_command"

    def test_schema_requires_command(self, action: ExecuteCommandAction) -> None:
        params = action.schema().parameters
        assert "command" in params["required"]


class TestExecuteCommandExecute:
    """Tests for the execute method."""

    async def test_successful_command(self, action: ExecuteCommandAction) -> None:
        result = await action.execute({"command": "echo hello"})
        assert result.success is True
        assert "hello" in result.output

    async def test_failing_command(self, action: ExecuteCommandAction) -> None:
        # Use a command that exits with non-zero code
        if sys.platform == "win32":
            result = await action.execute({"command": "cmd /c exit 1"})
        else:
            result = await action.execute({"command": "false"})
        assert result.success is False
        assert "exited with code" in result.error

    async def test_missing_command_param(self, action: ExecuteCommandAction) -> None:
        result = await action.execute({})
        assert result.success is False
        assert "command" in result.error.lower()

    async def test_empty_command(self, action: ExecuteCommandAction) -> None:
        result = await action.execute({"command": ""})
        assert result.success is False

    async def test_stderr_captured(self, action: ExecuteCommandAction) -> None:
        if sys.platform == "win32":
            result = await action.execute({"command": "cmd /c echo error_output>&2"})
        else:
            result = await action.execute({"command": "echo error_output >&2"})
        assert result.success is True
        assert "error_output" in result.output

    async def test_timeout_parameter(self, action: ExecuteCommandAction) -> None:
        # Use a very short timeout for a command that takes longer
        result = await action.execute({
            "command": "python -c \"import time; time.sleep(10)\"",
            "timeout": 0.1,
        })
        assert result.success is False
        assert "timed out" in result.error.lower()

    async def test_invalid_timeout_falls_back(self, action: ExecuteCommandAction) -> None:
        # Invalid timeout should fall back to default (30s)
        result = await action.execute({"command": "echo ok", "timeout": "not_a_number"})
        assert result.success is True
        assert "ok" in result.output

    async def test_stdout_and_stderr_combined(self, action: ExecuteCommandAction) -> None:
        if sys.platform == "win32":
            result = await action.execute({"command": "cmd /c echo out & echo err>&2"})
        else:
            result = await action.execute({"command": "echo out; echo err >&2"})
        assert result.success is True
        assert "out" in result.output
        assert "err" in result.output
