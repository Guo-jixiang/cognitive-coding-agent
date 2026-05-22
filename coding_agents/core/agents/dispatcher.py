"""SubAgent dispatcher — parallel execution via asyncio.Queue message passing.

Manages concurrent SubAgent execution by grouping subtasks into dependency
waves and dispatching each wave as parallel asyncio.Tasks. Each SubAgent
runs as an independent worker that reads from its own input queue and
writes results to a shared result queue.

Public API:
    - ``SubAgentDispatcher``: Manages parallel SubAgent execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from coding_agents.core.agents.analyzer import AnalyzerAgent
from coding_agents.core.agents.base import BaseSubAgent, SubAgentResult, SubTask
from coding_agents.core.agents.coder import CoderAgent
from coding_agents.core.agents.debugger import DebuggerAgent
from coding_agents.core.agents.executor import ExecutorAgent
from coding_agents.core.agents.messages import ResultMessage, TaskMessage
from coding_agents.core.agents.planner import PlannerAgent
from coding_agents.core.agents.researcher import ResearcherAgent
from coding_agents.core.agents.reviewer import ReviewerAgent
from coding_agents.core.agents.tester import TesterAgent
from coding_agents.llm.client import LLMClient

logger = logging.getLogger(__name__)


class _SubAgentFactory(Protocol):
    """Protocol describing the constructor signature of concrete SubAgents."""

    def __call__(self, llm_client: LLMClient) -> BaseSubAgent: ...


# Mapping from agent_type string to SubAgent class
AGENT_MAP: dict[str, _SubAgentFactory] = {
    "planner": PlannerAgent,
    "coder": CoderAgent,
    "reviewer": ReviewerAgent,
    "tester": TesterAgent,
    "analyzer": AnalyzerAgent,
    "debugger": DebuggerAgent,
    "executor": ExecutorAgent,
    "researcher": ResearcherAgent,
}


class SubAgentDispatcher:
    """Manages parallel SubAgent execution via asyncio.Queue message passing.

    Groups subtasks into dependency waves. Within each wave, all subtasks
    whose dependencies are satisfied run concurrently as independent
    asyncio.Tasks. Each SubAgent gets its own input queue (TaskMessage)
    and writes results to a shared result queue (ResultMessage).

    Args:
        llm_client: The LLM client shared across all SubAgent instances.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the dispatcher.

        Args:
            llm_client: The LLM client for SubAgent chat completions.
        """
        self._llm_client = llm_client
        self._result_queue: asyncio.Queue[ResultMessage] = asyncio.Queue()

    async def dispatch(self, subtasks: list[SubTask]) -> list[SubAgentResult]:
        """Execute subtasks respecting dependencies, parallelizing where possible.

        Groups subtasks into waves based on ``depends_on``. Within each wave,
        all subtasks run concurrently as independent asyncio.Tasks.
        Each SubAgent gets its own input queue (TaskMessage) and writes
        results to the shared result queue (ResultMessage).

        Args:
            subtasks: The list of SubTasks to execute.

        Returns:
            A list of SubAgentResults in the same order as the input subtasks.
        """
        results: list[SubAgentResult | None] = [None] * len(subtasks)
        executed: set[int] = set()

        while len(executed) < len(subtasks):
            # Find all subtasks whose dependencies are satisfied
            ready: list[tuple[int, SubTask]] = [
                (idx, st)
                for idx, st in enumerate(subtasks)
                if idx not in executed
                and all(d in executed for d in st.depends_on)
            ]

            if not ready:
                # Remaining tasks have unresolvable dependencies
                logger.warning(
                    "Cannot resolve dependencies for %d remaining subtasks.",
                    len(subtasks) - len(executed),
                )
                break

            # Dispatch all ready subtasks in parallel
            tasks: list[asyncio.Task[ResultMessage]] = []
            for idx, subtask in ready:
                # Build context from completed dependencies
                dep_context = subtask.context
                for dep_idx in subtask.depends_on:
                    dep_result = results[dep_idx]
                    if dep_result is not None and dep_result.success:
                        dep_context += (
                            f"\n\nResult from previous step: "
                            f"{dep_result.output[:1000]}"
                        )

                # Create task message
                msg = TaskMessage(
                    task_id=idx,
                    description=subtask.description,
                    agent_type=subtask.agent_type,
                    context=dep_context,
                )

                # Launch SubAgent as independent asyncio.Task
                task = asyncio.create_task(
                    self._run_agent(msg), name=f"subagent-{idx}-{subtask.agent_type}"
                )
                tasks.append(task)

            # Wait for all tasks in this wave to complete
            wave_results: list[ResultMessage | BaseException] = (
                await asyncio.gather(*tasks, return_exceptions=True)
            )

            # Collect results from this wave
            for i, wave_result in enumerate(wave_results):
                if isinstance(wave_result, ResultMessage):
                    results[wave_result.task_id] = wave_result.result
                    executed.add(wave_result.task_id)
                    # Also push to shared result queue for observability
                    await self._result_queue.put(wave_result)
                elif isinstance(wave_result, BaseException):
                    # Determine which task failed
                    failed_idx = ready[i][0]
                    failed_subtask = ready[i][1]
                    error_result = SubAgentResult(
                        success=False,
                        output="",
                        error=f"SubAgent '{failed_subtask.agent_type}' raised: {wave_result}",
                    )
                    results[failed_idx] = error_result
                    executed.add(failed_idx)
                    logger.error(
                        "SubAgent task %d (%s) failed with exception: %s",
                        failed_idx,
                        failed_subtask.agent_type,
                        wave_result,
                    )

        # Fill any remaining None slots with failure results
        final_results: list[SubAgentResult] = []
        for idx, r in enumerate(results):
            if r is None:
                final_results.append(
                    SubAgentResult(
                        success=False,
                        output="",
                        error="Task was not executed (unresolved dependencies).",
                    )
                )
            else:
                final_results.append(r)

        return final_results

    async def _run_agent(self, msg: TaskMessage) -> ResultMessage:
        """Run a single SubAgent in its own async context.

        Creates a fresh SubAgent instance, executes the task, and wraps
        the result in a ResultMessage.

        Args:
            msg: The TaskMessage containing task details.

        Returns:
            A ResultMessage with the execution result.

        Raises:
            KeyError: If the agent_type is not found in AGENT_MAP.
        """
        agent_cls = AGENT_MAP.get(msg.agent_type)
        if agent_cls is None:
            return ResultMessage(
                task_id=msg.task_id,
                result=SubAgentResult(
                    success=False,
                    output="",
                    error=f"Unknown agent type: {msg.agent_type}",
                ),
            )

        agent = agent_cls(llm_client=self._llm_client)
        result = await agent.run(task=msg.description, context=msg.context)
        return ResultMessage(task_id=msg.task_id, result=result)

    @property
    def result_queue(self) -> asyncio.Queue[ResultMessage]:
        """The shared result queue for observability and monitoring.

        External consumers can read from this queue to observe results
        as they arrive from SubAgents.
        """
        return self._result_queue


__all__: list[str] = ["AGENT_MAP", "SubAgentDispatcher"]
