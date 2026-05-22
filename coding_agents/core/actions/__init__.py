"""Agent actions module — structured tool invocations for the reasoning loop."""

from __future__ import annotations

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction
from coding_agents.core.actions.registry import ActionRegistry

__all__ = [
    "ActionRegistry",
    "ActionResult",
    "ActionSchema",
    "BaseAction",
]
