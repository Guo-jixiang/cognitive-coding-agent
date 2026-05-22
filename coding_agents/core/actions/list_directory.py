"""ListDirectoryAction — lists directory contents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction


class ListDirectoryAction(BaseAction):
    """Action that lists the contents of a directory."""

    def schema(self) -> ActionSchema:
        """Return the schema for the list_directory action."""
        return ActionSchema(
            name="list_directory",
            description="List the contents of a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (default: '.').",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively (default: false).",
                        "default": False,
                    },
                },
                "required": [],
            },
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """List the contents of the specified directory.

        Args:
            params: Dictionary with optional "path" and "recursive" keys.

        Returns:
            ActionResult with directory listing on success, or error on failure.
        """
        path_str = str(params.get("path", "."))
        recursive = bool(params.get("recursive", False))

        dir_path = Path(path_str)
        if not dir_path.exists():
            return ActionResult(
                success=False, output="", error=f"Directory not found: {path_str}"
            )
        if not dir_path.is_dir():
            return ActionResult(
                success=False, output="", error=f"Path is not a directory: {path_str}"
            )

        try:
            entries: list[str] = []
            if recursive:
                for item in sorted(dir_path.rglob("*")):
                    relative = item.relative_to(dir_path)
                    prefix = "[DIR] " if item.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {relative}")
            else:
                for item in sorted(dir_path.iterdir()):
                    prefix = "[DIR] " if item.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {item.name}")

            if not entries:
                return ActionResult(success=True, output="Directory is empty.")

            return ActionResult(success=True, output="\n".join(entries))

        except PermissionError:
            return ActionResult(
                success=False, output="", error=f"Permission denied: {path_str}"
            )
        except OSError as e:
            return ActionResult(success=False, output="", error=f"OS error listing directory: {e}")
        except Exception as e:  # noqa: BLE001
            return ActionResult(success=False, output="", error=f"Unexpected error: {e}")
