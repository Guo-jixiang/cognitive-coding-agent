"""SubAgent framework — specialized agents with isolated contexts and tools.

This package implements the SubAgent architecture where the Orchestrator
decomposes tasks and routes them to specialized SubAgents, each with
isolated tool access and context.

Communication between the Orchestrator and SubAgents uses asyncio.Queue-based
message passing via TaskMessage/ResultMessage, with parallel dispatch
managed by SubAgentDispatcher.

Public API:
    - ``BaseSubAgent``: Abstract base class for all SubAgents.
    - ``SubAgentResult``: Structured result from SubAgent execution.
    - ``SubAgentConfig``: Configuration for a SubAgent's role and capabilities.
    - ``SubTask``: A decomposed task to be routed to a SubAgent.
    - ``TaskMessage``: Message sent to a SubAgent with task details.
    - ``ResultMessage``: Message returned by a SubAgent with execution results.
    - ``SubAgentDispatcher``: Manages parallel SubAgent execution via queues.
    - ``Orchestrator``: Central task decomposition and routing coordinator.
    - ``PlannerAgent``: Produces structured plans (no tools).
    - ``CoderAgent``: Writes and modifies code files.
    - ``ReviewerAgent``: Reviews code and provides feedback.
    - ``TesterAgent``: Runs tests and reports results.
    - ``AnalyzerAgent``: Analyzes code architecture and patterns.
    - ``DebuggerAgent``: Debugs issues and finds root causes.
    - ``ExecutorAgent``: Executes commands and reports output.
    - ``ResearcherAgent``: Searches the codebase for information.
"""

from __future__ import annotations

from coding_agents.core.agents.analyzer import AnalyzerAgent
from coding_agents.core.agents.base import (
    BaseSubAgent,
    SubAgentConfig,
    SubAgentResult,
    SubTask,
)
from coding_agents.core.agents.coder import CoderAgent
from coding_agents.core.agents.debugger import DebuggerAgent
from coding_agents.core.agents.dispatcher import SubAgentDispatcher
from coding_agents.core.agents.executor import ExecutorAgent
from coding_agents.core.agents.messages import ResultMessage, TaskMessage
from coding_agents.core.agents.orchestrator import Orchestrator
from coding_agents.core.agents.planner import PlannerAgent
from coding_agents.core.agents.researcher import ResearcherAgent
from coding_agents.core.agents.reviewer import ReviewerAgent
from coding_agents.core.agents.tester import TesterAgent

__all__ = [
    "AnalyzerAgent",
    "BaseSubAgent",
    "CoderAgent",
    "DebuggerAgent",
    "ExecutorAgent",
    "Orchestrator",
    "PlannerAgent",
    "ResearcherAgent",
    "ResultMessage",
    "ReviewerAgent",
    "SubAgentConfig",
    "SubAgentDispatcher",
    "SubAgentResult",
    "SubTask",
    "TaskMessage",
    "TesterAgent",
]
