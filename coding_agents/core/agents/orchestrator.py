"""Orchestrator — decomposes tasks and routes to specialized SubAgents.

The Orchestrator is the central coordinator that:
1. Builds context via ContextBuilder.
2. Decomposes user tasks into SubTasks.
3. Dispatches SubTasks via SubAgentDispatcher (parallel, queue-based).
4. Reflects on results to produce a final answer.
5. Stores results in memory.

The Orchestrator does NOT directly execute any file operations, commands,
or code — it only plans, routes, and evaluates.

Public API:
    - ``Orchestrator``: Main orchestration class.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from coding_agents.context.builder import ContextBuilder
from coding_agents.core.agents.base import SubAgentResult, SubTask
from coding_agents.core.agents.dispatcher import SubAgentDispatcher
from coding_agents.core.engine import AgentResponse
from coding_agents.llm.client import ChatMessage, LLMClient
from coding_agents.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# Valid agent types for task decomposition
_VALID_AGENT_TYPES = frozenset(
    {"planner", "coder", "reviewer", "tester", "analyzer", "debugger", "executor", "researcher"}
)

_DECOMPOSITION_SYSTEM_PROMPT = """\
You are a task orchestrator. Decompose the user's request into sub-tasks.
Available agent types: planner, coder, reviewer, tester, analyzer, debugger, executor, researcher

Output a JSON array of sub-tasks:
[{"description": "...", "agent_type": "...", "context": "...", "depends_on": []}]

Rules:
- Use the minimum number of sub-tasks needed
- For simple questions or conversational messages, use a single "researcher" or "analyzer" task
- For coding tasks, typically: analyzer -> planner -> coder -> reviewer -> tester
- depends_on contains indices (0-based) of tasks that must complete first
- Output ONLY the JSON array, no other text
"""

_REFLECTION_SYSTEM_PROMPT = """\
You are a results synthesizer. Given the original task and the results from \
multiple sub-agents, produce a clear, coherent final answer.

If any sub-agent failed, note the failure and provide the best answer possible \
from the successful results. Be concise and direct.
"""

_SIMPLE_MESSAGE_PATTERNS = frozenset(
    {"hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay"}
)


class Orchestrator:
    """Central orchestrator that decomposes tasks and routes to SubAgents.

    The Orchestrator coordinates the full task lifecycle:
    1. Detects simple conversational messages and responds directly.
    2. Builds context via ContextBuilder for complex tasks.
    3. Decomposes tasks into SubTasks with dependency ordering.
    4. Dispatches SubTasks via SubAgentDispatcher (parallel execution).
    5. Reflects on all results to produce a final answer.
    6. Stores important results in memory.

    Args:
        llm_client: The LLM client for task decomposition and reflection.
        memory_manager: The memory manager for storing results.
        context_builder: The context builder for gathering relevant context.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_manager: MemoryManager,
        context_builder: ContextBuilder,
    ) -> None:
        """Initialize the Orchestrator.

        Args:
            llm_client: The LLM client for chat completions.
            memory_manager: The memory manager for cross-memory operations.
            context_builder: The GSSC context builder.
        """
        self._llm_client = llm_client
        self._memory_manager = memory_manager
        self._context_builder = context_builder

    async def run(self, user_message: str) -> AgentResponse:
        """Execute the full orchestration loop for a user message.

        Steps:
            1. Check if the message is simple/conversational.
            2. Build context via ContextBuilder.
            3. Decompose task into SubTasks.
            4. Dispatch SubTasks via SubAgentDispatcher (parallel).
            5. Reflect on results to produce final answer.
            6. Store results in memory.

        Args:
            user_message: The user's input message or task.

        Returns:
            An AgentResponse with the final answer and reasoning trace.
        """
        reasoning_trace: list[dict[str, Any]] = []
        memory_updates: list[str] = []

        # Step 1: Check for simple conversational messages
        if self._is_simple_message(user_message):
            answer = await self._respond_directly(user_message)
            return AgentResponse(
                answer=answer,
                reasoning_trace=[{"type": "thought", "content": "Simple conversational message — responding directly."}],
                memory_updates=[],
            )

        # Step 2: Build context
        context_str = ""
        try:
            context_result = await self._context_builder.build(
                query=user_message, max_tokens=4096
            )
            context_str = context_result.content
            reasoning_trace.append(
                {"type": "thought", "content": f"Built context ({context_result.items_included} items)."}
            )
        except Exception as exc:
            logger.warning("ContextBuilder failed: %s", exc)
            reasoning_trace.append(
                {"type": "thought", "content": "Context building failed, proceeding without context."}
            )

        # Step 3: Decompose task into SubTasks
        try:
            subtasks = await self._decompose_task(user_message, context_str)
            reasoning_trace.append(
                {
                    "type": "plan",
                    "content": f"Decomposed into {len(subtasks)} sub-tasks: "
                    + ", ".join(f"[{st.agent_type}] {st.description[:50]}" for st in subtasks),
                }
            )
        except Exception as exc:
            logger.error("Task decomposition failed: %s", exc)
            # Fallback: use a single researcher agent
            subtasks = [
                SubTask(
                    description=user_message,
                    agent_type="researcher",
                    context=context_str,
                )
            ]
            reasoning_trace.append(
                {"type": "thought", "content": f"Decomposition failed ({exc}), using single researcher."}
            )

        # Step 4: Dispatch SubTasks via SubAgentDispatcher (parallel)
        dispatcher = SubAgentDispatcher(self._llm_client)
        results = await dispatcher.dispatch(subtasks)

        # Record execution trace
        for idx, (subtask, result) in enumerate(zip(subtasks, results, strict=True)):
            reasoning_trace.append(
                {
                    "type": "observation",
                    "content": (
                        f"[{subtask.agent_type}] "
                        f"{'✓' if result.success else '✗'}: "
                        f"{result.summary or result.error or ''}"
                    ),
                }
            )

        # Step 5: Reflect on results
        final_answer = await self._reflect_on_results(user_message, results)
        reasoning_trace.append(
            {"type": "reflection", "content": "Synthesized final answer from sub-agent results."}
        )

        # Step 6: Store results in memory
        try:
            summary = (
                f"Task: {user_message[:200]}\n"
                f"SubTasks: {len(subtasks)}\n"
                f"Successful: {sum(1 for r in results if r.success)}\n"
                f"Answer: {final_answer[:500]}"
            )
            item = await self._memory_manager.store(
                content=summary,
                memory_type="episodic",
                metadata={
                    "source": "orchestrator",
                    "subtask_count": len(subtasks),
                    "task_preview": user_message[:100],
                },
                importance=0.7,
            )
            memory_updates.append(item.id)
        except Exception as exc:
            logger.warning("Failed to store orchestration result in memory: %s", exc)

        return AgentResponse(
            answer=final_answer,
            reasoning_trace=reasoning_trace,
            memory_updates=memory_updates,
        )

    async def _decompose_task(self, message: str, context: str) -> list[SubTask]:
        """Decompose a user message into a list of SubTasks.

        Calls the LLM with a system prompt asking it to produce a JSON
        array of sub-tasks with agent_type, description, context, and
        depends_on fields.

        Args:
            message: The user's message to decompose.
            context: Relevant context from the ContextBuilder.

        Returns:
            A list of SubTask instances.

        Raises:
            ValueError: If the LLM output cannot be parsed as valid sub-tasks.
        """
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_DECOMPOSITION_SYSTEM_PROMPT),
        ]

        user_content = f"User request: {message}"
        if context:
            user_content += f"\n\nAvailable context:\n{context[:2000]}"
        messages.append(ChatMessage(role="user", content=user_content))

        response = await self._llm_client.invoke(messages)

        # Parse JSON from response
        subtasks = self._parse_subtasks_json(response)
        if not subtasks:
            # Fallback: single researcher task
            return [
                SubTask(
                    description=message,
                    agent_type="researcher",
                    context=context,
                )
            ]
        return subtasks

    def _parse_subtasks_json(self, response: str) -> list[SubTask]:
        """Parse LLM response into SubTask list.

        Handles cases where the JSON might be wrapped in markdown code blocks.

        Args:
            response: The raw LLM response text.

        Returns:
            A list of SubTask instances, or empty list if parsing fails.
        """
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        # Try to find JSON array in the text
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx == -1 or end_idx == -1:
            return []

        json_str = text[start_idx : end_idx + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        subtasks: list[SubTask] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", ""))
            agent_type = str(item.get("agent_type", "researcher"))
            context = str(item.get("context", ""))
            depends_on_raw = item.get("depends_on", [])

            # Validate agent_type
            if agent_type not in _VALID_AGENT_TYPES:
                agent_type = "researcher"

            # Validate depends_on
            depends_on: list[int] = []
            if isinstance(depends_on_raw, list):
                for dep in depends_on_raw:
                    if isinstance(dep, int) and 0 <= dep < len(data):
                        depends_on.append(dep)

            if description:
                subtasks.append(
                    SubTask(
                        description=description,
                        agent_type=agent_type,
                        context=context,
                        depends_on=depends_on,
                    )
                )

        return subtasks

    async def _reflect_on_results(
        self, original_task: str, results: list[SubAgentResult]
    ) -> str:
        """Synthesize SubAgent results into a final answer.

        Calls the LLM as a reviewer to combine all sub-agent outputs
        into a coherent response.

        Args:
            original_task: The original user message.
            results: List of SubAgentResults from all sub-tasks.

        Returns:
            The synthesized final answer string.
        """
        # If only one successful result, return it directly
        successful_results = [r for r in results if r.success]
        if len(successful_results) == 1:
            return successful_results[0].output

        # If no successful results, report failure
        if not successful_results:
            errors = [r.error for r in results if r.error]
            return f"I was unable to complete the task. Errors encountered: {'; '.join(errors)}"

        # Multiple results — synthesize
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_REFLECTION_SYSTEM_PROMPT),
        ]

        results_text = ""
        for i, result in enumerate(results):
            status = "SUCCESS" if result.success else "FAILED"
            output = result.output[:1000] if result.success else (result.error or "Unknown error")
            results_text += f"\nSub-task {i + 1} [{status}]: {output}\n"

        user_content = (
            f"Original task: {original_task}\n\n"
            f"Sub-agent results:\n{results_text}\n\n"
            "Please synthesize these results into a clear, coherent final answer."
        )
        messages.append(ChatMessage(role="user", content=user_content))

        try:
            return await self._llm_client.invoke(messages)
        except Exception as exc:
            logger.error("Reflection failed: %s", exc)
            # Fallback: concatenate successful outputs
            return "\n\n".join(r.output for r in successful_results)

    def _is_simple_message(self, message: str) -> bool:
        """Detect simple conversational messages that don't need decomposition.

        Args:
            message: The user's message.

        Returns:
            True if the message is a simple greeting or acknowledgment.
        """
        normalized = message.strip().lower().rstrip("!.,?")
        return normalized in _SIMPLE_MESSAGE_PATTERNS

    async def _respond_directly(self, message: str) -> str:
        """Respond directly to a simple conversational message.

        Args:
            message: The simple message to respond to.

        Returns:
            A direct conversational response.
        """
        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content="You are a helpful coding assistant. Respond naturally to conversational messages.",
            ),
            ChatMessage(role="user", content=message),
        ]

        try:
            return await self._llm_client.invoke(messages)
        except Exception as exc:
            logger.error("Direct response failed: %s", exc)
            return "Hello! How can I help you today?"


__all__ = ["Orchestrator"]
