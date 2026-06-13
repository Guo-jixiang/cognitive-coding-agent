"""Unit tests for WriteFileAction."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agents.core.actions.write_file import WriteFileAction


@pytest.fixture
def action() -> WriteFileAction:
    return WriteFileAction()


class TestWriteFileActionSchema:
    """Tests for the schema definition."""

    def test_schema_name(self, action: WriteFileAction) -> None:
        assert action.schema().name == "write_file"

    def test_schema_requires_path_and_content(self, action: WriteFileAction) -> None:
        params = action.schema().parameters
        assert "path" in params["required"]
        assert "content" in params["required"]


class TestWriteFileActionExecute:
    """Tests for the execute method."""

    async def test_write_creates_file(self, action: WriteFileAction, tmp_path) -> None:
        target = tmp_path / "output.txt"
        result = await action.execute({"path": str(target), "content": "hello"})
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "hello"

    async def test_write_creates_parent_dirs(self, action: WriteFileAction, tmp_path) -> None:
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        result = await action.execute({"path": str(target), "content": "nested"})
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "nested"

    async def test_write_overwrites_existing(self, action: WriteFileAction, tmp_path) -> None:
        target = tmp_path / "overwrite.txt"
        target.write_text("old", encoding="utf-8")

        result = await action.execute({"path": str(target), "content": "new"})
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "new"

    async def test_write_missing_path(self, action: WriteFileAction) -> None:
        result = await action.execute({"content": "data"})
        assert result.success is False
        assert "path" in result.error.lower()

    async def test_write_missing_content(self, action: WriteFileAction, tmp_path) -> None:
        result = await action.execute({"path": str(tmp_path / "f.txt")})
        assert result.success is False
        assert "content" in result.error.lower()

    async def test_write_content_with_none(self, action: WriteFileAction, tmp_path) -> None:
        result = await action.execute({"path": str(tmp_path / "f.txt"), "content": None})
        assert result.success is False

    async def test_write_output_message_includes_count(self, action: WriteFileAction, tmp_path) -> None:
        target = tmp_path / "count.txt"
        result = await action.execute({"path": str(target), "content": "abcde"})
        assert result.success is True
        assert "5" in result.output

    async def test_write_empty_content(self, action: WriteFileAction, tmp_path) -> None:
        target = tmp_path / "empty.txt"
        result = await action.execute({"path": str(target), "content": ""})
        assert result.success is True
        assert target.read_text(encoding="utf-8") == ""
