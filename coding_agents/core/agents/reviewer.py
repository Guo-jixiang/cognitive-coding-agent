"""ReviewerAgent — provides code review feedback.

The Reviewer SubAgent has access to read_file and list_directory.
It reads code and provides specific, actionable feedback.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a code reviewer. Read the code and provide specific, actionable "
    "feedback on correctness, style, and edge cases."
)


class ReviewerAgent(BaseSubAgent):
    """SubAgent that reviews code and provides feedback.

    Has access to: read_file, list_directory.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="reviewer",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["read_file", "list_directory"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the ReviewerAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Review code and provide feedback.

        Args:
            task: The review task description.
            context: Optional additional context (e.g., code to review).

        Returns:
            A SubAgentResult with the review feedback.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            return SubAgentResult(
                success=True,
                output=result.answer,
                summary=f"Completed review: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Review failed: {exc}",
            )


__all__ = ["ReviewerAgent"]
