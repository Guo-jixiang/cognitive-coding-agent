"""DebuggerAgent — analyzes errors and finds root causes.

The Debugger SubAgent has access to read_file, search_code, and execute_command.
It analyzes errors, searches for root causes, and runs diagnostic commands.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a debugging agent. Analyze errors, search for root causes, "
    "and run diagnostic commands."
)


class DebuggerAgent(BaseSubAgent):
    """SubAgent that debugs issues and finds root causes.

    Has access to: read_file, search_code, execute_command.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="debugger",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["read_file", "search_code", "execute_command"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the DebuggerAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Debug an issue and report findings.

        Args:
            task: The debugging task description.
            context: Optional additional context (e.g., error messages).

        Returns:
            A SubAgentResult with debugging findings.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            return SubAgentResult(
                success=True,
                output=result.answer,
                summary=f"Completed debugging: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Debugging failed: {exc}",
            )


__all__ = ["DebuggerAgent"]
