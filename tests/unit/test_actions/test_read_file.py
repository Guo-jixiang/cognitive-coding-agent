"""Unit tests for ReadFileAction."""

from __future__ import annotations

import pytest

from coding_agents.core.actions.read_file import ReadFileAction


@pytest.fixture
def action() -> ReadFileAction:
    return ReadFileAction()


class TestReadFileActionSchema:
    """Tests for the schema definition."""

    def test_schema_name(self, action: ReadFileAction) -> None:
        assert action.schema().name == "read_file"

    def test_schema_has_path_parameter(self, action: ReadFileAction) -> None:
        params = action.schema().parameters
        assert "path" in params["properties"]
        assert "path" in params["required"]


class TestReadFileActionExecute:
    """Tests for the execute method."""

    async def test_read_existing_file(self, action: ReadFileAction, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")

        result = await action.execute({"path": str(test_file)})
        assert result.success is True
        assert result.output == "hello world"
        assert result.error is None

    async def test_read_missing_path_param(self, action: ReadFileAction) -> None:
        result = await action.execute({})
        assert result.success is False
        assert "path" in result.error.lower()

    async def test_read_empty_path_param(self, action: ReadFileAction) -> None:
        result = await action.execute({"path": ""})
        assert result.success is False

    async def test_read_nonexistent_file(self, action: ReadFileAction) -> None:
        result = await action.execute({"path": "/nonexistent/file.txt"})
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_read_directory_as_file(self, action: ReadFileAction, tmp_path) -> None:
        result = await action.execute({"path": str(tmp_path)})
        assert result.success is False
        assert "not a file" in result.error.lower()

    async def test_read_unicode_content(self, action: ReadFileAction, tmp_path) -> None:
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("你好世界 🌍", encoding="utf-8")

        result = await action.execute({"path": str(test_file)})
        assert result.success is True
        assert "你好世界" in result.output

    async def test_read_empty_file(self, action: ReadFileAction, tmp_path) -> None:
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        result = await action.execute({"path": str(test_file)})
        assert result.success is True
        assert result.output == ""
