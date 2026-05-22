"""Action registry for storing and looking up actions by name."""

from __future__ import annotations

from coding_agents.core.actions.base import ActionSchema, BaseAction


class ActionRegistry:
    """Registry that stores actions by their schema name for lookup.

    Actions are registered via ``register()`` and can be retrieved by name
    via ``get()``. The ``list_schemas()`` method returns all registered
    action schemas for tool discovery.
    """

    def __init__(self) -> None:
        """Initialize an empty action registry."""
        self._actions: dict[str, BaseAction] = {}

    def register(self, action: BaseAction) -> None:
        """Register an action by its schema name.

        Args:
            action: The action instance to register.
        """
        name = action.schema().name
        self._actions[name] = action

    def get(self, name: str) -> BaseAction | None:
        """Look up an action by name.

        Args:
            name: The action name to look up.

        Returns:
            The action instance if found, None otherwise.
        """
        return self._actions.get(name)

    def list_schemas(self) -> list[ActionSchema]:
        """Return schemas for all registered actions.

        Returns:
            A list of ActionSchema instances for every registered action.
        """
        return [action.schema() for action in self._actions.values()]
