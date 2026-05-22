"""WriteFileAction — writes content to a file path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction


class WriteFileAction(BaseAction):
    """Action that writes content to a file at the specified path."""

    def schema(self) -> ActionSchema:
        """Return the schema for the write_file action."""
        return ActionSchema(
            name="write_file",
            description="Write content to a file at the specified path. Creates parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Write content to the specified file path.

        Args:
            params: Dictionary with "path" and "content" keys.

        Returns:
            ActionResult indicating success or failure.
        """
        path_str = params.get("path")
        content = params.get("content")

        if not path_str:
            return ActionResult(success=False, output="", error="Missing required parameter: path")
        if content is None:
            return ActionResult(
                success=False, output="", error="Missing required parameter: content"
            )

        try:
            file_path = Path(str(path_str))
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(str(content), encoding="utf-8")
            return ActionResult(
                success=True, output=f"Successfully wrote {len(str(content))} characters to {path_str}"
            )
        except PermissionError:
            return ActionResult(
                success=False, output="", error=f"Permission denied: {path_str}"
            )
        except OSError as e:
            return ActionResult(success=False, output="", error=f"OS error writing file: {e}")
        except Exception as e:  # noqa: BLE001
            return ActionResult(success=False, output="", error=f"Unexpected error: {e}")
