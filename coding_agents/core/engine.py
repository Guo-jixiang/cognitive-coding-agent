"""Agent Execution Engine for the Cognitive Coding Agent.

This module implements the central :class:`AgentEngine` that orchestrates
the reasoning loop by coordinating interactions between the LLM client,
reasoning paradigms, tool actions, context builder, and memory systems.

When an Orchestrator is provided, the engine delegates to it for task
decomposition and SubAgent routing. Otherwise, it uses the direct
paradigm-based execution path for backward compatibility.

Public API:
    - ``AgentResponse``: Structured result from agent execution.
    - ``AgentEngine``: Core execution engine class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from coding_agents.context.builder import ContextBuilder
from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms import ParadigmFactory
from coding_agents.llm.client import LLMClient
from coding_agents.memory.manager import MemoryManager

if TYPE_CHECKING:
    from coding_agents.core.agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured result returned after agent execution completes.

    Attributes:
        answer: The final answer text produced by the agent.
        reasoning_trace: Complete list of reasoning steps (thoughts,
            actions, observations, reflections).
        memory_updates: List of memory item IDs created or updated
            during execution.
    """

    answer: str
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[str] = field(default_factory=list)


class AgentEngine:
    """Central execution engine orchestrating LLM, paradigms, actions, and memory.

    The engine coordinates the full reasoning loop:
    1. Builds optimal context via the ContextBuilder.
    2. Creates and executes the selected reasoning paradigm.
    3. Stores reasoning artifacts in episodic memory.
    4. Returns a structured AgentResponse.

    When an Orchestrator is provided, the engine delegates to it for
    SubAgent-based task decomposition and routing. Otherwise, it uses
    the direct paradigm-based execution path.

    Args:
        llm_client: The LLM client for chat completions.
        memory_manager: The memory manager for cross-memory operations.
        context_builder: The GSSC context builder.
        action_registry: Registry of available tool actions.
        max_steps: Maximum reasoning steps before forced termination.
        orchestrator: Optional Orchestrator for SubAgent-based execution.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_manager: MemoryManager,
        context_builder: ContextBuilder,
        action_registry: ActionRegistry,
        max_steps: int = 20,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        """Initialize the AgentEngine with its dependencies.

        Args:
            llm_client: The LLM client for chat completions.
            memory_manager: The memory manager for cross-memory operations.
            context_builder: The GSSC context builder.
            action_registry: Registry of available tool actions.
            max_steps: Maximum reasoning steps before forced termination.
            orchestrator: Optional Orchestrator for SubAgent-based execution.
                If provided, run() delegates to the Orchestrator instead of
                direct paradigm execution.
        """
        self._llm_client = llm_client
        self._memory_manager = memory_manager
        self._context_builder = context_builder
        self._action_registry = action_registry
        self._max_steps = max_steps
        self._orchestrator = orchestrator
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the engine has been initialized."""
        return self._initialized

    @property
    def max_steps(self) -> int:
        """The configured maximum number of reasoning steps."""
        return self._max_steps

    async def initialize(self) -> None:
        """Initialize all dependent components.

        Calls MemoryManager.initialize() to start all memory subsystems
        and verify connectivity.

        Raises:
            RuntimeError: If initialization fails critically.
        """
        try:
            await self._memory_manager.initialize()
            self._initialized = True
            logger.info("AgentEngine initialized successfully.")
        except Exception as exc:
            logger.error("AgentEngine initialization failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"AgentEngine initialization failed: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Gracefully shut down, persisting memory state.

        Calls MemoryManager.shutdown() to persist episodic and semantic
        memory and close connections.
        """
        try:
            await self._memory_manager.shutdown()
            logger.info("AgentEngine shut down successfully.")
        except Exception as exc:
            logger.error("AgentEngine shutdown error: %s", exc, exc_info=True)
        finally:
            self._initialized = False

    async def run(
        self,
        user_message: str,
        paradigm: str = "reflection",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Execute the agent reasoning loop for the given user message.

        If an Orchestrator is configured, delegates to it for SubAgent-based
        task decomposition and routing. Otherwise, uses the direct paradigm
        execution path.

        Steps (direct paradigm path):
            1. Build context via ContextBuilder (query = user_message).
            2. Prepend conversation history to context if provided.
            3. Create the selected paradigm via ParadigmFactory.
            4. Execute the paradigm with task=user_message and built context.
            5. Store a reasoning summary in EpisodicMemory (importance=0.7).
            6. Return AgentResponse with answer, trace, and memory update IDs.

        If the paradigm execution fails, returns a partial response with
        the error information.

        Args:
            user_message: The user's input message or task.
            paradigm: The reasoning paradigm to use. One of "react",
                "plan_and_solve", or "reflection". Defaults to "reflection".
            conversation_history: Optional list of previous conversation
                messages as dicts with "role" and "content" keys. If
                provided, prepended to the context for continuity.

        Returns:
            An AgentResponse containing the final answer, reasoning trace,
            and list of memory item IDs created during execution.
        """
        # Delegate to Orchestrator if available
        if self._orchestrator is not None:
            return await self._orchestrator.run(user_message)

        memory_updates: list[str] = []

        # Step 1: Build context via ContextBuilder
        try:
            context_result = await self._context_builder.build(
                query=user_message, max_tokens=4096
            )
            context_str = context_result.content
        except Exception as exc:
            logger.warning(
                "ContextBuilder failed, using empty context: %s", exc
            )
            context_str = ""

        # Step 2: Prepend conversation history to context
        if conversation_history:
            history_lines: list[str] = ["Previous conversation:"]
            for msg in conversation_history:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)
            if context_str:
                context_str = history_str + "\n\n" + context_str
            else:
                context_str = history_str

        # Step 3: Create paradigm via ParadigmFactory
        try:
            paradigm_instance = ParadigmFactory.create(
                name=paradigm,
                llm_client=self._llm_client,
                action_registry=self._action_registry,
            )
        except ValueError as exc:
            logger.error("Invalid paradigm '%s': %s", paradigm, exc)
            return AgentResponse(
                answer=f"Error: Invalid paradigm '{paradigm}'.",
                reasoning_trace=[{"type": "error", "content": str(exc)}],
                memory_updates=[],
            )

        # Step 4: Execute paradigm
        try:
            result = await paradigm_instance.run(
                task=user_message, context=context_str
            )
        except Exception as exc:
            logger.error("Paradigm execution failed: %s", exc, exc_info=True)
            return AgentResponse(
                answer="I encountered an error while processing your request.",
                reasoning_trace=[
                    {"type": "error", "content": f"Paradigm execution failed: {exc}"}
                ],
                memory_updates=[],
            )

        # Step 5: Store reasoning summary in EpisodicMemory
        try:
            summary = (
                f"Task: {user_message[:200]}\n"
                f"Paradigm: {paradigm}\n"
                f"Answer: {result.answer[:500]}\n"
                f"Steps: {result.iterations}"
            )
            item = await self._memory_manager.store(
                content=summary,
                memory_type="episodic",
                metadata={
                    "source": "agent_engine",
                    "paradigm": paradigm,
                    "iterations": result.iterations,
                    "task_preview": user_message[:100],
                },
                importance=0.7,
            )
            memory_updates.append(item.id)
        except Exception as exc:
            logger.warning(
                "Failed to store reasoning summary in memory: %s", exc
            )

        # Step 6: Return AgentResponse
        return AgentResponse(
            answer=result.answer,
            reasoning_trace=result.reasoning_trace,
            memory_updates=memory_updates,
        )


__all__ = ["AgentEngine", "AgentResponse"]
