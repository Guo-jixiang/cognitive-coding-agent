"""Unit tests for SearchCodeAction."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agents.core.actions.search_code import SearchCodeAction


@pytest.fixture
def action() -> SearchCodeAction:
    return SearchCodeAction()


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a small sample project for searching."""
    (tmp_path / "main.py").write_text("def hello():\n    print('hello')\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.py").write_text("class Foo:\n    pass\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Project\nSome text\n", encoding="utf-8")
    return tmp_path


class TestSearchCodeSchema:
    """Tests for the schema definition."""

    def test_schema_name(self, action: SearchCodeAction) -> None:
        assert action.schema().name == "search_code"

    def test_schema_requires_pattern(self, action: SearchCodeAction) -> None:
        params = action.schema().parameters
        assert "pattern" in params["required"]


class TestSearchCodeExecute:
    """Tests for the execute method."""

    async def test_find_function_definition(
        self, action: SearchCodeAction, sample_project: Path
    ) -> None:
        result = await action.execute({
            "pattern": "def hello",
            "directory": str(sample_project),
        })
        assert result.success is True
        assert "hello" in result.output

    async def test_no_matches(self, action: SearchCodeAction, sample_project: Path) -> None:
        result = await action.execute({
            "pattern": "nonexistent_function_xyz",
            "directory": str(sample_project),
        })
        assert result.success is True
        assert "no matches" in result.output.lower()

    async def test_search_with_glob_filter(
        self, action: SearchCodeAction, sample_project: Path
    ) -> None:
        result = await action.execute({
            "pattern": "Project",
            "directory": str(sample_project),
            "file_glob": "*.md",
        })
        assert result.success is True
        assert "Project" in result.output

    async def test_search_excludes_non_matching_glob(
        self, action: SearchCodeAction, sample_project: Path
    ) -> None:
        result = await action.execute({
            "pattern": "Project",
            "directory": str(sample_project),
            "file_glob": "*.py",
        })
        assert result.success is True
        assert "no matches" in result.output.lower()

    async def test_recursive_search(
        self, action: SearchCodeAction, sample_project: Path
    ) -> None:
        result = await action.execute({
            "pattern": "class Foo",
            "directory": str(sample_project),
        })
        assert result.success is True
        assert "Foo" in result.output

    async def test_missing_pattern(self, action: SearchCodeAction) -> None:
        result = await action.execute({})
        assert result.success is False
        assert "pattern" in result.error.lower()

    async def test_invalid_regex(self, action: SearchCodeAction, sample_project: Path) -> None:
        result = await action.execute({
            "pattern": "[invalid",
            "directory": str(sample_project),
        })
        assert result.success is False
        assert "regex" in result.error.lower() or "invalid" in result.error.lower()

    async def test_nonexistent_directory(self, action: SearchCodeAction) -> None:
        result = await action.execute({
            "pattern": "test",
            "directory": "/nonexistent/dir",
        })
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_file_as_directory(self, action: SearchCodeAction, sample_project: Path) -> None:
        result = await action.execute({
            "pattern": "test",
            "directory": str(sample_project / "main.py"),
        })
        assert result.success is False
        assert "not a directory" in result.error.lower()

    async def test_regex_special_characters(
        self, action: SearchCodeAction, sample_project: Path
    ) -> None:
        result = await action.execute({
            "pattern": r"print\(.+\)",
            "directory": str(sample_project),
        })
        assert result.success is True
        assert "print" in result.output
