"""ExecuteCommandAction — runs a shell command via asyncio subprocess."""

from __future__ import annotations

import asyncio
from typing import Any

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction

_DEFAULT_TIMEOUT = 30


class ExecuteCommandAction(BaseAction):
    """Action that executes a shell command and returns stdout + stderr."""

    def schema(self) -> ActionSchema:
        """Return the schema for the execute_command action."""
        return ActionSchema(
            name="execute_command",
            description="Execute a shell command and return its output (stdout + stderr).",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 30).",
                        "default": _DEFAULT_TIMEOUT,
                    },
                },
                "required": ["command"],
            },
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute a shell command and capture its output.

        Args:
            params: Dictionary with "command" and optional "timeout" keys.

        Returns:
            ActionResult with combined stdout/stderr on success, or error on timeout/failure.
        """
        command = params.get("command")
        if not command:
            return ActionResult(
                success=False, output="", error="Missing required parameter: command"
            )

        timeout = params.get("timeout", _DEFAULT_TIMEOUT)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = float(_DEFAULT_TIMEOUT)

        try:
            process = await asyncio.create_subprocess_shell(
                str(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ActionResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout} seconds: {command}",
                )

            output_parts: list[str] = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(stderr.decode("utf-8", errors="replace"))

            combined_output = "\n".join(output_parts).strip()

            if process.returncode == 0:
                return ActionResult(success=True, output=combined_output)
            else:
                return ActionResult(
                    success=False,
                    output=combined_output,
                    error=f"Command exited with code {process.returncode}",
                )

        except OSError as e:
            return ActionResult(success=False, output="", error=f"OS error executing command: {e}")
        except Exception as e:  # noqa: BLE001
            return ActionResult(success=False, output="", error=f"Unexpected error: {e}")
