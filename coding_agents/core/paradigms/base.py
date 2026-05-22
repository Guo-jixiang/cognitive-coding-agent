"""Base paradigm definitions: BaseParadigm ABC and ParadigmResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.llm.client import LLMClient


@dataclass
class ParadigmResult:
    """Structured result returned after a paradigm completes execution.

    Attributes:
        answer: The final answer produced by the paradigm.
        reasoning_trace: List of reasoning steps, each a dict with
            'type' and 'content' keys describing the step.
        iterations: Number of iterations or steps taken during execution.
    """

    answer: str
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0


class BaseParadigm(ABC):
    """Abstract base class for all agent reasoning paradigms.

    Subclasses implement a specific problem-solving strategy (e.g., ReAct,
    Plan-and-Solve, Reflection) by overriding the ``run`` method.
    """

    def __init__(
        self, llm_client: LLMClient, action_registry: ActionRegistry
    ) -> None:
        """Initialize the paradigm with shared dependencies.

        Args:
            llm_client: The LLM client for making chat completion requests.
            action_registry: Registry of available actions the paradigm can invoke.
        """
        self._llm_client = llm_client
        self._action_registry = action_registry

    @abstractmethod
    async def run(self, task: str, context: str = "") -> ParadigmResult:
        """Execute the paradigm on the given task.

        Args:
            task: The user task or question to solve.
            context: Optional additional context to include.

        Returns:
            A ParadigmResult containing the answer, reasoning trace,
            and iteration count.
        """
