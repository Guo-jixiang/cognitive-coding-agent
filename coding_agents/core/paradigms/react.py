"""ReAct paradigm: Thought → Action → Observation loop."""

from __future__ import annotations

import json
from typing import Any

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.base import BaseParadigm, ParadigmResult
from coding_agents.llm.client import ChatMessage, LLMClient

_MAX_ITERATIONS = 20

_SYSTEM_PROMPT = """\
You are a helpful AI coding assistant. You help users with programming tasks, \
answer questions, and provide information. You are NOT the user — you are their assistant.

When the user tells you something about themselves (like their name), acknowledge it \
naturally and remember it for the conversation.

You solve tasks using a Thought → Action → Observation loop.

Available actions:
{actions_description}

At each step, you MUST output ONE of the following:

1. A thought followed by an action (JSON on its own line):
   Thought: <your reasoning about what to do next>
   {{"action": "<action_name>", "params": {{<parameters>}}}}

2. A final answer when you have enough information:
   {{"answer": "<your final answer>"}}

Rules:
- Always think before acting.
- Use actions to gather information or perform operations.
- When you have enough information to answer, output the final answer JSON.
- Do NOT output both an action and a final answer in the same response.
- For simple conversational messages, respond directly with a final answer.
"""


def _build_actions_description(action_registry: ActionRegistry) -> str:
    """Build a human-readable description of available actions."""
    schemas = action_registry.list_schemas()
    if not schemas:
        return "No actions available."
    lines: list[str] = []
    for schema in schemas:
        params_str = json.dumps(schema.parameters, indent=2)
        lines.append(f"- {schema.name}: {schema.description}\n  Parameters: {params_str}")
    return "\n".join(lines)


def _parse_llm_output(text: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Parse LLM output into thought, action/answer components.

    Returns:
        A tuple of (thought, action_dict, final_answer).
        - thought: extracted thought text, or None
        - action_dict: parsed action JSON with 'action' and 'params', or None
        - final_answer: the final answer string, or None
    """
    thought: str | None = None
    action_dict: dict[str, Any] | None = None
    final_answer: str | None = None

    # Extract thought
    lines = text.strip().split("\n")
    thought_lines: list[str] = []
    json_line: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Thought:"):
            thought_lines.append(stripped[len("Thought:"):].strip())
        elif stripped.startswith("{"):
            json_line = stripped
        elif thought_lines and not stripped.startswith("{"):
            # Continuation of thought
            thought_lines.append(stripped)

    if thought_lines:
        thought = " ".join(thought_lines)

    # Try to find JSON in the text
    if json_line is None:
        # Search for JSON anywhere in the text
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                json_line = stripped
                break

    if json_line is not None:
        try:
            parsed = json.loads(json_line)
            if "answer" in parsed:
                final_answer = str(parsed["answer"])
            elif "action" in parsed:
                action_dict = parsed
        except json.JSONDecodeError:
            pass

    # If no JSON found inline, try parsing the entire text as JSON
    if action_dict is None and final_answer is None:
        try:
            parsed = json.loads(text.strip())
            if "answer" in parsed:
                final_answer = str(parsed["answer"])
            elif "action" in parsed:
                action_dict = parsed
        except json.JSONDecodeError:
            pass

    return thought, action_dict, final_answer


class ReActParadigm(BaseParadigm):
    """ReAct paradigm: Thought → Action → Observation loop.

    The LLM reasons about the task, selects actions to execute, observes
    results, and repeats until it produces a final answer or hits the
    maximum iteration limit.
    """

    def __init__(
        self, llm_client: LLMClient, action_registry: ActionRegistry
    ) -> None:
        """Initialize the ReAct paradigm.

        Args:
            llm_client: The LLM client for chat completions.
            action_registry: Registry of executable actions.
        """
        super().__init__(llm_client, action_registry)
        self._max_iterations = _MAX_ITERATIONS

    async def run(self, task: str, context: str = "") -> ParadigmResult:
        """Execute the ReAct loop on the given task.

        Args:
            task: The user task or question to solve.
            context: Optional additional context to include.

        Returns:
            A ParadigmResult with the final answer, reasoning trace,
            and number of iterations performed.
        """
        actions_desc = _build_actions_description(self._action_registry)
        system_prompt = _SYSTEM_PROMPT.format(actions_description=actions_desc)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
        ]

        user_content = f"Task: {task}"
        if context:
            user_content = f"Context:\n{context}\n\n{user_content}"
        messages.append(ChatMessage(role="user", content=user_content))

        reasoning_trace: list[dict[str, Any]] = []
        iterations = 0

        for _ in range(self._max_iterations):
            iterations += 1

            response_text = await self._llm_client.chat_completion(messages)
            thought, action_dict, final_answer = _parse_llm_output(response_text)

            # Record thought
            if thought:
                reasoning_trace.append({"type": "thought", "content": thought})

            # Final answer case
            if final_answer is not None:
                return ParadigmResult(
                    answer=final_answer,
                    reasoning_trace=reasoning_trace,
                    iterations=iterations,
                )

            # Action case
            if action_dict is not None:
                action_name = str(action_dict.get("action", ""))
                action_params: dict[str, Any] = action_dict.get("params", {})
                if not isinstance(action_params, dict):
                    action_params = {}

                reasoning_trace.append(
                    {
                        "type": "action",
                        "content": {"action": action_name, "params": action_params},
                    }
                )

                # Execute the action
                action = self._action_registry.get(action_name)
                if action is not None:
                    result = await action.execute(action_params)
                    observation = result.output if result.success else f"Error: {result.error}"
                else:
                    observation = f"Error: Unknown action '{action_name}'"

                reasoning_trace.append({"type": "observation", "content": observation})

                # Feed observation back to LLM
                messages.append(ChatMessage(role="assistant", content=response_text))
                messages.append(
                    ChatMessage(role="user", content=f"Observation: {observation}")
                )
            else:
                # LLM produced neither action nor answer — treat raw text as answer
                return ParadigmResult(
                    answer=response_text.strip(),
                    reasoning_trace=reasoning_trace,
                    iterations=iterations,
                )

        # Max iterations reached — return best available result
        return ParadigmResult(
            answer="Maximum iterations reached without a final answer.",
            reasoning_trace=reasoning_trace,
            iterations=iterations,
        )
