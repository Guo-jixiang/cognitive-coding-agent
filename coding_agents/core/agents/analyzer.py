"""AnalyzerAgent — analyzes code architecture and patterns.

The Analyzer SubAgent has access to read_file, search_code, and list_directory.
It searches and reads code to understand architecture, dependencies, and patterns.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a code analysis agent. Search and read code to understand "
    "architecture, dependencies, and patterns."
)


class AnalyzerAgent(BaseSubAgent):
    """SubAgent that analyzes code structure and patterns.

    Has access to: read_file, search_code, list_directory.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="analyzer",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=["read_file", "search_code", "list_directory"],
        paradigm="react",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the AnalyzerAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Analyze code and report findings.

        Args:
            task: The analysis task description.
            context: Optional additional context.

        Returns:
            A SubAgentResult with analysis findings.
        """
        full_task = f"{self._config.system_prompt}\n\nTask: {task}"

        try:
            result = await self._paradigm.run(task=full_task, context=context)
            return SubAgentResult(
                success=True,
                output=result.answer,
                summary=f"Completed analysis: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Analysis failed: {exc}",
            )


__all__ = ["AnalyzerAgent"]
