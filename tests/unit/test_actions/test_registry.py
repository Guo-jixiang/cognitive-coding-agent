"""Unit tests for ActionRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from coding_agents.core.actions.base import ActionResult, ActionSchema, BaseAction
from coding_agents.core.actions.registry import ActionRegistry


class _DummyAction(BaseAction):
    """A minimal action for testing the registry."""

    def __init__(self, name: str = "dummy", description: str = "A dummy action") -> None:
        self._name = name
        self._description = description

    def schema(self) -> ActionSchema:
        return ActionSchema(
            name=self._name,
            description=self._description,
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        return ActionResult(success=True, output="ok")


class TestActionRegistryConstruction:
    """Tests for ActionRegistry initialization."""

    def test_empty_registry(self) -> None:
        registry = ActionRegistry()
        assert registry.list_schemas() == []

    def test_get_returns_none_for_unknown(self) -> None:
        registry = ActionRegistry()
        assert registry.get("nonexistent") is None


class TestActionRegistryRegister:
    """Tests for registering actions."""

    def test_register_and_get(self) -> None:
        registry = ActionRegistry()
        action = _DummyAction("test_action", "A test action")
        registry.register(action)

        retrieved = registry.get("test_action")
        assert retrieved is action

    def test_register_overwrites_same_name(self) -> None:
        registry = ActionRegistry()
        first = _DummyAction("my_action", "first")
        second = _DummyAction("my_action", "second")
        registry.register(first)
        registry.register(second)

        assert registry.get("my_action") is second

    def test_list_schemas_returns_all(self) -> None:
        registry = ActionRegistry()
        registry.register(_DummyAction("action_a", "A"))
        registry.register(_DummyAction("action_b", "B"))

        schemas = registry.list_schemas()
        names = {s.name for s in schemas}
        assert names == {"action_a", "action_b"}

    def test_list_schemas_preserves_description(self) -> None:
        registry = ActionRegistry()
        registry.register(_DummyAction("foo", "Foo description"))

        schemas = registry.list_schemas()
        assert len(schemas) == 1
        assert schemas[0].description == "Foo description"

    def test_multiple_registrations(self) -> None:
        registry = ActionRegistry()
        for i in range(10):
            registry.register(_DummyAction(f"action_{i}"))

        assert len(registry.list_schemas()) == 10
        for i in range(10):
            assert registry.get(f"action_{i}") is not None
