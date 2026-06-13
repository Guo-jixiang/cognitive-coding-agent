"""Unit tests for ListDirectoryAction."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agents.core.actions.list_directory import ListDirectoryAction


@pytest.fixture
def action() -> ListDirectoryAction:
    return ListDirectoryAction()


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a sample directory structure."""
    (tmp_path / "file_a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "file_b.py").write_text("b", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")
    return tmp_path


class TestListDirectorySchema:
    """Tests for the schema definition."""

    def test_schema_name(self, action: ListDirectoryAction) -> None:
        assert action.schema().name == "list_directory"


class TestListDirectoryExecute:
    """Tests for the execute method."""

    async def test_list_current_directory(
        self, action: ListDirectoryAction, sample_dir: Path
    ) -> None:
        result = await action.execute({"path": str(sample_dir)})
        assert result.success is True
        assert "[FILE]" in result.output
        assert "[DIR]" in result.output

    async def test_list_contains_files(self, action: ListDirectoryAction, sample_dir: Path) -> None:
        result = await action.execute({"path": str(sample_dir)})
        assert "file_a.txt" in result.output
        assert "file_b.py" in result.output

    async def test_list_contains_subdirectory(
        self, action: ListDirectoryAction, sample_dir: Path
    ) -> None:
        result = await action.execute({"path": str(sample_dir)})
        assert "subdir" in result.output

    async def test_recursive_listing(
        self, action: ListDirectoryAction, sample_dir: Path
    ) -> None:
        result = await action.execute({"path": str(sample_dir), "recursive": True})
        assert result.success is True
        assert "nested.txt" in result.output

    async def test_non_recursive_does_not_show_nested(
        self, action: ListDirectoryAction, sample_dir: Path
    ) -> None:
        result = await action.execute({"path": str(sample_dir), "recursive": False})
        assert result.success is True
        assert "nested.txt" not in result.output

    async def test_empty_directory(self, action: ListDirectoryAction, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = await action.execute({"path": str(empty)})
        assert result.success is True
        assert "empty" in result.output.lower()

    async def test_nonexistent_directory(self, action: ListDirectoryAction) -> None:
        result = await action.execute({"path": "/nonexistent/dir"})
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_file_as_path(self, action: ListDirectoryAction, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        result = await action.execute({"path": str(f)})
        assert result.success is False
        assert "not a directory" in result.error.lower()

    async def test_default_path_is_current_dir(self, action: ListDirectoryAction) -> None:
        result = await action.execute({})
        assert result.success is True

    async def test_sorted_output(self, action: ListDirectoryAction, tmp_path: Path) -> None:
        (tmp_path / "z.txt").write_text("z", encoding="utf-8")
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "m.txt").write_text("m", encoding="utf-8")

        result = await action.execute({"path": str(tmp_path)})
        assert result.success is True
        lines = result.output.strip().split("\n")
        names = [line.split(" ", 1)[1] for line in lines]
        assert names == sorted(names)
