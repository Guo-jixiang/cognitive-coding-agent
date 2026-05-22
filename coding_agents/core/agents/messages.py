"""Message protocol for Orchestrator ↔ SubAgent communication.

Defines the message types used for queue-based message passing between
the Orchestrator's dispatcher and SubAgent workers.

Public API:
    - ``TaskMessage``: Message sent to a SubAgent with task details.
    - ``ResultMessage``: Message returned by a SubAgent with execution results.
"""

from __future__ import annotations

from dataclasses import dataclass

from coding_agents.core.agents.base import SubAgentResult


@dataclass(frozen=True, slots=True)
class TaskMessage:
    """Message dispatched to a SubAgent worker via its input queue.

    Attributes:
        task_id: Index of this task in the subtask list (used to correlate results).
        description: The task description for the SubAgent to execute.
        agent_type: Which SubAgent type should handle this task.
        context: Task-relevant context including dependency results.
    """

    task_id: int
    description: str
    agent_type: str
    context: str


@dataclass(frozen=True, slots=True)
class ResultMessage:
    """Message returned by a SubAgent worker via the shared result queue.

    Attributes:
        task_id: Matches the TaskMessage.task_id for correlation.
        result: The structured SubAgentResult from execution.
    """

    task_id: int
    result: SubAgentResult


__all__ = ["ResultMessage", "TaskMessage"]
