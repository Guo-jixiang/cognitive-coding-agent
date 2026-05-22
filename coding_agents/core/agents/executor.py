"""ExecutorAgent — runs commands and reports output.

The Executor SubAgent has access to execute_command and read_file.
It runs commands and reports their output.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are an execution agent. Run commands and report their output. "
    "Read files for context when needed."
)


class ExecutorAgent(BaseSubAgent):
    """SubAgent that executes commands and reports results.

    Has access to: execute_command, read_file.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="executor",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["execute_command", "read_file"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the ExecutorAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Execute commands and report output.

        Args:
            task: The execution task description.
            context: Optional additional context.

        Returns:
            A SubAgentResult with command output.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            return SubAgentResult(
                success=True,
                output=result.answer,
                summary=f"Completed execution: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Execution failed: {exc}",
            )


__all__ = ["ExecutorAgent"]
