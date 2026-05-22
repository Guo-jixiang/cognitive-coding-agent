"""ReadFileAction — reads file content by path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction


class ReadFileAction(BaseAction):
    """Action that reads the content of a file given its path."""

    def schema(self) -> ActionSchema:
        """Return the schema for the read_file action."""
        return ActionSchema(
            name="read_file",
            description="Read the content of a file at the specified path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read.",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Read file content from the given path.

        Args:
            params: Dictionary with a "path" key specifying the file to read.

        Returns:
            ActionResult with file content on success, or error on failure.
        """
        path_str = params.get("path")
        if not path_str:
            return ActionResult(success=False, output="", error="Missing required parameter: path")

        try:
            file_path = Path(str(path_str))
            if not file_path.exists():
                return ActionResult(
                    success=False, output="", error=f"File not found: {path_str}"
                )
            if not file_path.is_file():
                return ActionResult(
                    success=False, output="", error=f"Path is not a file: {path_str}"
                )
            content = file_path.read_text(encoding="utf-8")
            return ActionResult(success=True, output=content)
        except PermissionError:
            return ActionResult(
                success=False, output="", error=f"Permission denied: {path_str}"
            )
        except OSError as e:
            return ActionResult(success=False, output="", error=f"OS error reading file: {e}")
        except Exception as e:  # noqa: BLE001
            return ActionResult(success=False, output="", error=f"Unexpected error: {e}")
