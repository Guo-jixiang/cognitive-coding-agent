"""TesterAgent — runs tests and reports results.

The Tester SubAgent has access to read_file and execute_command.
It reads test files and runs pytest commands.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a testing agent. Read test files and run pytest commands. "
    "Report test results clearly."
)


class TesterAgent(BaseSubAgent):
    """SubAgent that runs tests and reports results.

    Has access to: read_file, execute_command.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="tester",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["read_file", "execute_command"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the TesterAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Run tests and report results.

        Args:
            task: The testing task description.
            context: Optional additional context.

        Returns:
            A SubAgentResult with test results.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            return SubAgentResult(
                success=True,
                output=result.answer,
                summary=f"Completed testing: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Testing failed: {exc}",
            )


__all__ = ["TesterAgent"]
