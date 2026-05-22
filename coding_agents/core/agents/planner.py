"""PlannerAgent — produces structured plans without tool access.

The Planner SubAgent has NO tool access. It uses the LLM directly to
decompose tasks into clear, ordered steps.
"""

from __future__ import annotations

from coding_agents.core.agents.base import BaseSubAgent, SubAgentConfig, SubAgentResult
from coding_agents.llm.client import ChatMessage, LLMClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a planning agent. Decompose tasks into clear, ordered steps. "
    "Output a numbered plan."
)


class PlannerAgent(BaseSubAgent):
    """SubAgent that produces structured plans via LLM reasoning.

    The Planner has no tool access — it only reasons about how to
    decompose a task into actionable steps.
    """

    DEFAULT_CONFIG = SubAgentConfig(
        role="planner",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        allowed_tools=[],
        paradigm="plan_and_solve",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        config: SubAgentConfig | None = None,
    ) -> None:
        """Initialize the PlannerAgent.

        Args:
            llm_client: The LLM client for chat completions.
            config: Optional custom configuration. Uses DEFAULT_CONFIG if None.
        """
        effective_config = config or self.DEFAULT_CONFIG
        super().__init__(effective_config, llm_client)

    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Produce a structured plan for the given task.

        Calls the LLM directly (no tools needed) to generate a plan.

        Args:
            task: The task to plan for.
            context: Optional additional context.

        Returns:
            A SubAgentResult containing the plan as output.
        """
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self._config.system_prompt),
        ]

        user_content = f"Task: {task}"
        if context:
            user_content = f"Context:\n{context}\n\n{user_content}"
        messages.append(ChatMessage(role="user", content=user_content))

        try:
            response = await self._llm_client.invoke(messages)
            return SubAgentResult(
                success=True,
                output=response,
                summary=f"Generated plan for: {task[:100]}",
            )
        except Exception as exc:
            return SubAgentResult(
                success=False,
                output="",
                error=f"Planning failed: {exc}",
            )


__all__ = ["PlannerAgent"]
