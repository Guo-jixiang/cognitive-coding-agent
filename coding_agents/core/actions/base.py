"""Base action definitions: ActionSchema, ActionResult, and BaseAction ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSchema:
    """Structured description of an action's interface.

    Attributes:
        name: Unique action identifier (e.g., "read_file").
        description: Human-readable description of what the action does.
        parameters: JSON Schema dict describing accepted parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ActionResult:
    """Structured result returned after executing an action.

    Attributes:
        success: Whether the action completed successfully.
        output: Result content on success, empty string on failure.
        error: Error description if success is False, None otherwise.
    """

    success: bool
    output: str
    error: str | None = None


class BaseAction(ABC):
    """Abstract base class for all agent actions.

    Subclasses must implement ``schema()`` to describe their interface
    and ``execute()`` to perform the action logic.
    """

    @abstractmethod
    def schema(self) -> ActionSchema:
        """Return the structured schema for this action."""

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute the action with the given parameters.

        Implementations must never raise exceptions — errors are returned
        as ``ActionResult(success=False, output="", error=<description>)``.
        """
