"""Unit tests for ParadigmFactory."""

from __future__ import annotations

import pytest

from coding_agents.core.actions.registry import ActionRegistry
from coding_agents.core.paradigms import ParadigmFactory
from coding_agents.core.paradigms.plan_and_solve import PlanAndSolveParadigm
from coding_agents.core.paradigms.react import ReActParadigm
from coding_agents.core.paradigms.reflection import ReflectionParadigm


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry()


class TestParadigmFactory:
    """Tests for ParadigmFactory.create()."""

    def test_create_react(self, registry: ActionRegistry) -> None:
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        paradigm = ParadigmFactory.create("react", mock_llm, registry)
        assert isinstance(paradigm, ReActParadigm)

    def test_create_plan_and_solve(self, registry: ActionRegistry) -> None:
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        paradigm = ParadigmFactory.create("plan_and_solve", mock_llm, registry)
        assert isinstance(paradigm, PlanAndSolveParadigm)

    def test_create_reflection(self, registry: ActionRegistry) -> None:
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        paradigm = ParadigmFactory.create("reflection", mock_llm, registry)
        assert isinstance(paradigm, ReflectionParadigm)

    def test_create_unknown_raises_value_error(self, registry: ActionRegistry) -> None:
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Unknown paradigm"):
            ParadigmFactory.create("nonexistent", mock_llm, registry)

    def test_create_unknown_lists_available(self, registry: ActionRegistry) -> None:
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        with pytest.raises(ValueError) as exc_info:
            ParadigmFactory.create("bad", mock_llm, registry)
        error_msg = str(exc_info.value)
        assert "react" in error_msg
        assert "plan_and_solve" in error_msg
        assert "reflection" in error_msg
