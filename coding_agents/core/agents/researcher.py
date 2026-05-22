"""ResearcherAgent — searches the codebase for relevant information.

The Researcher SubAgent has access to search_code, read_file, and list_directory.
It searches the codebase to find relevant code, patterns, and documentation.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a research agent. Search the codebase to find relevant code, "
    "patterns, and documentation."
)


class ResearcherAgent(BaseSubAgent):
    """SubAgent that researches the codebase for information.

    Has access to: search_code, read_file, list_directory.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="researcher",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["search_code", "read_file", "list_directory"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the ResearcherAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Research the codebase and report findings.

        Args:
            task: The research task description.
            context: Optional additional context.

        Returns:
            A SubAgentResult with research findings.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            return SubAgentResult(
                success=True,
                output=result.answer,
                summary=f"Completed research: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Research failed: {exc}",
            )


__all__ = ["ResearcherAgent"]
