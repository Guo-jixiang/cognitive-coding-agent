"""Agent paradigms module — reasoning strategies for problem-solving."""

from __future__ import annotations

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.base import BaseParadigm, ParadigmResult
from coding_agents.core.paradigms.plan_and_solve import PlanAndSolveParadigm
from coding_agents.core.paradigms.react import ReActParadigm
from coding_agents.core.paradigms.reflection import ReflectionParadigm
from coding_agents.llm.client import LLMClient

_PARADIGM_MAP: dict[str, type[BaseParadigm]] = {
    "react": ReActParadigm,
    "plan_and_solve": PlanAndSolveParadigm,
    "reflection": ReflectionParadigm,
}


class ParadigmFactory:
    """Factory for creating paradigm instances by name.

    Supported paradigm names:
        - ``"react"`` → ReActParadigm
        - ``"plan_and_solve"`` → PlanAndSolveParadigm
        - ``"reflection"`` → ReflectionParadigm
    """

    @staticmethod
    def create(
        name: str, llm_client: LLMClient, action_registry: ActionRegistry
    ) -> BaseParadigm:
        """Create a paradigm instance by name.

        Args:
            name: The paradigm identifier (e.g., "react", "plan_and_solve",
                "reflection").
            llm_client: The LLM client for chat completions.
            action_registry: Registry of executable actions.

        Returns:
            An instance of the requested paradigm.

        Raises:
            ValueError: If the paradigm name is not recognized.
        """
        paradigm_cls = _PARADIGM_MAP.get(name)
        if paradigm_cls is None:
            available = ", ".join(sorted(_PARADIGM_MAP.keys()))
            raise ValueError(
                f"Unknown paradigm '{name}'. Available: {available}"
            )
        return paradigm_cls(llm_client, action_registry)


__all__ = [
    "BaseParadigm",
    "ParadigmFactory",
    "ParadigmResult",
    "PlanAndSolveParadigm",
    "ReActParadigm",
    "ReflectionParadigm",
]
