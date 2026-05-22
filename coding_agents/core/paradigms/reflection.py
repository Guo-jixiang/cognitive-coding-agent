"""Reflection paradigm: Execute → Reflect → Refine loop."""

from __future__ import annotations

import re
from typing import Any

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms.base import BaseParadigm, ParadigmResult
from coding_agents.core.paradigms.react import ReActParadigm
from coding_agents.llm.client import ChatMessage, LLMClient

_MAX_REFLECTION_ITERATIONS = 3

_REVIEWER_SYSTEM_PROMPT = """\
You are a critical code reviewer. Evaluate the following draft answer for:
1. Correctness — Are there factual or logical errors?
2. Completeness — Does it fully address the original task?
3. Code quality — If code is present, is it well-structured and idiomatic?
4. Edge cases — Are important edge cases handled?

If the draft is satisfactory with no significant issues, respond with exactly:
LGTM

Otherwise, provide specific, actionable feedback on what needs to be improved.
Be concise and focus on the most important issues.
"""

_REFINE_SYSTEM_PROMPT = """\
You are a refinement agent. You have received feedback on your previous draft.
Incorporate the feedback to produce an improved version.

Original task: {task}

Previous draft:
{draft}

Reviewer feedback:
{feedback}

Produce an improved answer that addresses the feedback. Output only the \
improved answer without meta-commentary.
"""


def _is_lgtm(review_text: str) -> bool:
    """Check if the reviewer approved the draft."""
    normalized = review_text.strip().lower()
    # Check for explicit LGTM or "no issues found" patterns
    if normalized == "lgtm":
        return True
    if re.search(r"\blgtm\b", normalized) and len(normalized) < 50:
        return True
    if "no issues found" in normalized:
        return True
    if "no significant issues" in normalized and len(normalized) < 80:
        return True
    return False


class ReflectionParadigm(BaseParadigm):
    """Reflection paradigm: Execute → Reflect → Refine.

    Uses ReActParadigm internally for the initial execution phase.
    After getting a draft answer, invokes a separate LLM call with a
    reviewer system prompt to evaluate the draft. If issues are found,
    refines the answer by calling the LLM with the draft and feedback.
    Repeats for a maximum of 3 reflection iterations, stopping early
    if the reviewer approves.
    """

    def __init__(
        self, llm_client: LLMClient, action_registry: ActionRegistry
    ) -> None:
        """Initialize the Reflection paradigm.

        Args:
            llm_client: The LLM client for chat completions.
            action_registry: Registry of executable actions.
        """
        super().__init__(llm_client, action_registry)
        self._react = ReActParadigm(llm_client, action_registry)
        self._max_iterations = _MAX_REFLECTION_ITERATIONS

    async def run(self, task: str, context: str = "") -> ParadigmResult:
        """Execute the Reflection strategy on the given task.

        Args:
            task: The user task or question to solve.
            context: Optional additional context to include.

        Returns:
            A ParadigmResult with the refined answer, full reasoning trace
            (including draft, reviews, and refinements), and iteration count.
        """
        reasoning_trace: list[dict[str, Any]] = []

        # Phase 1: Generate initial draft using ReAct
        react_result = await self._react.run(task, context)
        draft = react_result.answer
        iterations = react_result.iterations

        reasoning_trace.extend(react_result.reasoning_trace)
        reasoning_trace.append({"type": "draft", "content": draft})

        # Phase 2: Reflection loop
        for reflection_round in range(self._max_iterations):
            iterations += 1

            # Review the current draft
            review_messages: list[ChatMessage] = [
                ChatMessage(role="system", content=_REVIEWER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"Original task: {task}\n\n"
                        f"Draft answer to review:\n{draft}"
                    ),
                ),
            ]

            review_text = await self._llm_client.chat_completion(review_messages)
            reasoning_trace.append(
                {
                    "type": "review",
                    "content": {
                        "round": reflection_round + 1,
                        "feedback": review_text,
                    },
                }
            )

            # Check if reviewer approves
            if _is_lgtm(review_text):
                break

            # Refine the draft based on feedback
            iterations += 1
            refine_prompt = _REFINE_SYSTEM_PROMPT.format(
                task=task, draft=draft, feedback=review_text
            )
            refine_messages: list[ChatMessage] = [
                ChatMessage(role="system", content=refine_prompt),
                ChatMessage(
                    role="user",
                    content="Please produce the improved answer now.",
                ),
            ]

            refined = await self._llm_client.chat_completion(refine_messages)
            draft = refined.strip()

            reasoning_trace.append(
                {
                    "type": "refinement",
                    "content": {"round": reflection_round + 1, "refined_answer": draft},
                }
            )

        return ParadigmResult(
            answer=draft,
            reasoning_trace=reasoning_trace,
            iterations=iterations,
        )
