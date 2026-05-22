"""SearchCodeAction — searches files matching a regex pattern in a directory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction


class SearchCodeAction(BaseAction):
    """Action that searches for a regex pattern across files in a directory."""

    def schema(self) -> ActionSchema:
        """Return the schema for the search_code action."""
        return ActionSchema(
            name="search_code",
            description="Search for a regex pattern in files within a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "The directory to search in (default: '.').",
                        "default": ".",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Glob pattern for files to search (default: '*.py').",
                        "default": "*.py",
                    },
                },
                "required": ["pattern"],
            },
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Search for a pattern in files matching the glob within the directory.

        Args:
            params: Dictionary with "pattern", optional "directory" and "file_glob".

        Returns:
            ActionResult with matching lines on success, or error on failure.
        """
        pattern_str = params.get("pattern")
        if not pattern_str:
            return ActionResult(
                success=False, output="", error="Missing required parameter: pattern"
            )

        directory = str(params.get("directory", "."))
        file_glob = str(params.get("file_glob", "*.py"))

        try:
            regex = re.compile(str(pattern_str))
        except re.error as e:
            return ActionResult(
                success=False, output="", error=f"Invalid regex pattern: {e}"
            )

        dir_path = Path(directory)
        if not dir_path.exists():
            return ActionResult(
                success=False, output="", error=f"Directory not found: {directory}"
            )
        if not dir_path.is_dir():
            return ActionResult(
                success=False, output="", error=f"Path is not a directory: {directory}"
            )

        matches: list[str] = []
        try:
            for file_path in dir_path.rglob(file_glob):
                if not file_path.is_file():
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    for line_num, line in enumerate(content.splitlines(), start=1):
                        if regex.search(line):
                            matches.append(f"{file_path}:{line_num}: {line}")
                except (PermissionError, OSError):
                    continue
        except OSError as e:
            return ActionResult(
                success=False, output="", error=f"Error searching directory: {e}"
            )

        if not matches:
            return ActionResult(success=True, output="No matches found.")

        return ActionResult(success=True, output="\n".join(matches))
