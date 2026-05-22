"""Plan-and-Solve paradigm: decompose into a plan, then execute steps."""

from __future__ import annotations

import json
import re
from typing import Any

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.base import BaseParadigm, ParadigmResult
from coding_agents.llm.client import ChatMessage, LLMClient

_PLAN_SYSTEM_PROMPT = """\
You are a planning agent. Given a task, decompose it into a numbered list of \
concrete steps. Each step should be a single, actionable instruction.

Output ONLY a numbered list like:
1. First step
2. Second step
3. Third step

Do not include any other text before or after the list.
"""

_EXECUTE_SYSTEM_PROMPT = """\
You are an execution agent. You are given a specific step to execute as part \
of a larger plan.

Available actions:
{actions_description}

If you need to use an action, output JSON on its own line:
{{"action": "<action_name>", "params": {{<parameters>}}}}

If you can complete the step without an action, just provide your result directly.

After completing the step, provide a brief summary of what was accomplished.
"""


def _parse_plan(text: str) -> list[str]:
    """Parse a numbered plan from LLM output into a list of step strings."""
    steps: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # Match lines starting with a number followed by . or )
        match = re.match(r"^\d+[.)]\s*(.+)$", line)
        if match:
            steps.append(match.group(1).strip())
    return steps


def _extract_action(text: str) -> dict[str, Any] | None:
    """Try to extract an action JSON from the text."""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                if "action" in parsed:
                    return parsed  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue
    # Try parsing the entire text
    try:
        parsed = json.loads(text.strip())
        if "action" in parsed:
            return parsed  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    return None


class PlanAndSolveParadigm(BaseParadigm):
    """Plan-and-Solve paradigm: decompose then execute sequentially.

    First, the LLM decomposes the task into a numbered plan of steps.
    Then each step is executed (potentially involving tool actions).
    Finally, results are aggregated into a final answer.
    """

    def __init__(
        self, llm_client: LLMClient, action_registry: ActionRegistry
    ) -> None:
        """Initialize the Plan-and-Solve paradigm.

        Args:
            llm_client: The LLM client for chat completions.
            action_registry: Registry of executable actions.
        """
        super().__init__(llm_client, action_registry)

    async def run(self, task: str, context: str = "") -> ParadigmResult:
        """Execute the Plan-and-Solve strategy on the given task.

        Args:
            task: The user task or question to solve.
            context: Optional additional context to include.

        Returns:
            A ParadigmResult with the final answer, reasoning trace,
            and number of iterations (plan + execution steps).
        """
        reasoning_trace: list[dict[str, Any]] = []
        iterations = 0

        # Phase 1: Generate plan
        plan_user_content = f"Task: {task}"
        if context:
            plan_user_content = f"Context:\n{context}\n\n{plan_user_content}"

        plan_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_PLAN_SYSTEM_PROMPT),
            ChatMessage(role="user", content=plan_user_content),
        ]

        plan_text = await self._llm_client.chat_completion(plan_messages)
        iterations += 1

        steps = _parse_plan(plan_text)
        if not steps:
            # If parsing fails, treat the entire response as a single step
            steps = [plan_text.strip()]

        reasoning_trace.append(
            {"type": "plan", "content": {"raw": plan_text, "steps": steps}}
        )

        # Phase 2: Execute each step
        from coding_agents.core.paradigms.react import _build_actions_description

        actions_desc = _build_actions_description(self._action_registry)
        execute_system = _EXECUTE_SYSTEM_PROMPT.format(actions_description=actions_desc)

        step_results: list[str] = []

        for i, step in enumerate(steps):
            iterations += 1

            step_messages: list[ChatMessage] = [
                ChatMessage(role="system", content=execute_system),
                ChatMessage(
                    role="user",
                    content=(
                        f"Plan step {i + 1}/{len(steps)}: {step}\n\n"
                        f"Original task: {task}"
                    ),
                ),
            ]

            step_response = await self._llm_client.chat_completion(step_messages)

            # Check if the step requires an action
            action_dict = _extract_action(step_response)
            if action_dict is not None:
                action_name = str(action_dict.get("action", ""))
                action_params: dict[str, Any] = action_dict.get("params", {})
                if not isinstance(action_params, dict):
                    action_params = {}

                reasoning_trace.append(
                    {
                        "type": "step_action",
                        "content": {
                            "step": i + 1,
                            "description": step,
                            "action": action_name,
                            "params": action_params,
                        },
                    }
                )

                action = self._action_registry.get(action_name)
                if action is not None:
                    result = await action.execute(action_params)
                    observation = (
                        result.output if result.success else f"Error: {result.error}"
                    )
                else:
                    observation = f"Error: Unknown action '{action_name}'"

                reasoning_trace.append(
                    {
                        "type": "step_observation",
                        "content": {"step": i + 1, "observation": observation},
                    }
                )
                step_results.append(f"Step {i + 1}: {observation}")
            else:
                reasoning_trace.append(
                    {
                        "type": "step_result",
                        "content": {"step": i + 1, "description": step, "result": step_response},
                    }
                )
                step_results.append(f"Step {i + 1}: {step_response}")

        # Phase 3: Aggregate results into final answer
        iterations += 1
        aggregate_messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "You are a summarization agent. Given the results of executing "
                    "a plan, synthesize them into a clear, concise final answer."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Original task: {task}\n\n"
                    f"Plan steps and results:\n" + "\n".join(step_results)
                ),
            ),
        ]

        final_answer = await self._llm_client.chat_completion(aggregate_messages)

        return ParadigmResult(
            answer=final_answer.strip(),
            reasoning_trace=reasoning_trace,
            iterations=iterations,
        )
