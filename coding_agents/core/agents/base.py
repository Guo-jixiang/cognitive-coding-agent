"""Base SubAgent framework: SubAgentResult, SubAgentConfig, SubTask, BaseSubAgent.

This module defines the foundational abstractions for the SubAgent architecture.
Each SubAgent operates in an isolated context with its own ActionRegistry
containing only the tools permitted for its role.

Public API:
    - ``SubAgentResult``: Structured result from SubAgent execution.
    - ``SubAgentConfig``: Configuration for a SubAgent's role and capabilities.
    - ``SubTask``: A decomposed task to be routed to a SubAgent.
    - ``BaseSubAgent``: Abstract base class for all SubAgents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from coding_agents.core.actions.execute_command import ExecuteCommandAction
from coding_agents.core.actions.list_directory import ListDirectoryAction
from coding_agents.core.actions.read_file import ReadFileAction
from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.actions.search_code import SearchCodeAction
from coding_agents.core.actions.write_file import WriteFileAction
from coding_agents.core.paradigms import ParadigmFactory
from coding_agents.core.paradigms.base import BaseParadigm
from coding_agents.llm.client import LLMClient


@dataclass
class SubAgentResult:
    """Structured result returned after a SubAgent completes execution.

    Attributes:
        success: Whether the SubAgent completed its task successfully.
        output: The main output content produced by the SubAgent.
        files_modified: List of file paths modified during execution.
        summary: A brief summary of what was accomplished.
        error: Error description if the task failed, None otherwise.
    """

    success: bool
    output: str
    files_modified: list[str] = field(default_factory=list)
    summary: str = ""
    error: str | None = None


@dataclass
class SubAgentConfig:
    """Configuration for a SubAgent's role and capabilities.

    Attributes:
        role: The SubAgent's role identifier (e.g., "coder", "reviewer").
        system_prompt: Role-specific system prompt for the LLM.
        allowed_tools: List of tool names this SubAgent can use.
        paradigm: The reasoning paradigm for this SubAgent (default: "react").
    """

    role: str
    system_prompt: str
    allowed_tools: list[str]
    paradigm: str = "react"


@dataclass
class SubTask:
    """A decomposed task to be routed to a specific SubAgent type.

    Attributes:
        description: What the SubAgent should accomplish.
        agent_type: The type of SubAgent to handle this task.
        context: Additional context relevant to this specific sub-task.
        depends_on: Indices of other sub-tasks that must complete first.
    """

    description: str
    agent_type: str
    context: str = ""
    depends_on: list[int] = field(default_factory=list)


# Mapping of action names to their classes for dynamic instantiation
_ACTION_CLASS_MAP: dict[str, type[ReadFileAction | WriteFileAction | ExecuteCommandAction | SearchCodeAction | ListDirectoryAction]] = {
    "read_file": ReadFileAction,
    "write_file": WriteFileAction,
    "execute_command": ExecuteCommandAction,
    "search_code": SearchCodeAction,
    "list_directory": ListDirectoryAction,
}


def _create_isolated_registry(allowed_tools: list[str]) -> ActionRegistry:
    """Create a fresh ActionRegistry with only the specified tools.

    Args:
        allowed_tools: List of action names to register.

    Returns:
        An ActionRegistry containing only the permitted actions.
    """
    registry = ActionRegistry()
    for tool_name in allowed_tools:
        action_cls = _ACTION_CLASS_MAP.get(tool_name)
        if action_cls is not None:
            registry.register(action_cls())
    return registry


class BaseSubAgent(ABC):
    """Abstract base class for all SubAgents.

    Each SubAgent operates in an isolated context with:
    - Its own ActionRegistry containing only permitted tools.
    - Its own paradigm instance for reasoning.
    - A fresh message list for each run() call (context isolation).

    Args:
        config: The SubAgent's configuration.
        llm_client: The shared LLM client for making requests.
    """

    def __init__(self, config: SubAgentConfig, llm_client: LLMClient) -> None:
        """Initialize the SubAgent with isolated tools and paradigm.

        Args:
            config: Configuration specifying role, tools, and paradigm.
            llm_client: The LLM client for chat completions.
        """
        self._config = config
        self._llm_client = llm_client
        self._registry = _create_isolated_registry(config.allowed_tools)
        self._paradigm: BaseParadigm = ParadigmFactory.create(
            name=config.paradigm,
            llm_client=llm_client,
            action_registry=self._registry,
        )

    @property
    def config(self) -> SubAgentConfig:
        """The SubAgent's configuration."""
        return self._config

    @abstractmethod
    async def run(self, task: str, context: str = "") -> SubAgentResult:
        """Execute a task in an isolated context.

        Each call starts with a fresh message list — no shared history
        between invocations.

        Args:
            task: The task description to accomplish.
            context: Optional additional context for the task.

        Returns:
            A SubAgentResult with the outcome of the execution.
        """


__all__ = [
    "BaseSubAgent",
    "SubAgentConfig",
    "SubAgentResult",
    "SubTask",
]
