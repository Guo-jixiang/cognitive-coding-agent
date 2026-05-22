"""CoderAgent — writes clean, well-documented code.

The Coder SubAgent has access to read_file, write_file, and list_directory.
It reads existing files before modifying them and produces clean code.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent. Write clean, well-documented Python code. "
    "Read existing files before modifying them."
)


class CoderAgent(BaseSubAgent):
    """SubAgent that writes and modifies code files.

    Has access to: read_file, write_file, list_directory.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="coder",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["read_file", "write_file", "list_directory"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the CoderAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Execute a coding task using the ReAct paradigm.

        Args:
            task: The coding task to accomplish.
            context: Optional additional context (e.g., file contents).

        Returns:
            A SubAgentResult with the coding outcome.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            # Extract files modified from reasoning trace
            files_modified: list[str] = []
            for step in result.reasoning_trace:
                if step.get("type") == "action":
                    action_info = step.get("content", {})
                    if isinstance(action_info, dict):
                        if action_info.get("action") == "write_file":
                            params = action_info.get("params", {})
                            path = params.get("path", "")
                            if path:
                                files_modified.append(str(path))

            return SubAgentResult(
                success=True,
                output=result.answer,
                files_modified=files_modified,
                summary=f"Completed coding task: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Coding task failed: {exc}",
            )


__all__ = ["CoderAgent"]
