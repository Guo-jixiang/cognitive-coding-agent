"""Unit tests for AgentEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agents.core.engine import AgentEngine, AgentResponse


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def mock_memory_manager():
    mm = AsyncMock()
    mm.store = AsyncMock(return_value=MagicMock(id="mem-123"))
    return mm


@pytest.fixture
def mock_context_builder():
    cb = AsyncMock()
    cb.build = AsyncMock(return_value=MagicMock(content="context", items_included=2))
    return cb


@pytest.fixture
def mock_action_registry():
    return MagicMock()


@pytest.fixture
def engine(mock_llm, mock_memory_manager, mock_context_builder, mock_action_registry):
    return AgentEngine(
        llm_client=mock_llm,
        memory_manager=mock_memory_manager,
        context_builder=mock_context_builder,
        action_registry=mock_action_registry,
    )


class TestAgentEngineConstruction:
    """Tests for AgentEngine initialization."""

    def test_initial_state_not_initialized(self, engine: AgentEngine) -> None:
        assert engine.is_initialized is False

    def test_max_steps_default(self, engine: AgentEngine) -> None:
        assert engine.max_steps == 20

    def test_max_steps_custom(self, mock_llm, mock_memory_manager, mock_context_builder, mock_action_registry) -> None:
        engine = AgentEngine(
            llm_client=mock_llm,
            memory_manager=mock_memory_manager,
            context_builder=mock_context_builder,
            action_registry=mock_action_registry,
            max_steps=50,
        )
        assert engine.max_steps == 50


class TestAgentEngineLifecycle:
    """Tests for initialize and shutdown."""

    async def test_initialize_sets_flag(self, engine: AgentEngine) -> None:
        await engine.initialize()
        assert engine.is_initialized is True

    async def test_initialize_calls_memory_manager(self, engine: AgentEngine, mock_memory_manager) -> None:
        await engine.initialize()
        mock_memory_manager.initialize.assert_awaited_once()

    async def test_initialize_failure_raises_runtime_error(self, engine: AgentEngine, mock_memory_manager) -> None:
        mock_memory_manager.initialize = AsyncMock(side_effect=Exception("connection failed"))
        with pytest.raises(RuntimeError, match="initialization failed"):
            await engine.initialize()

    async def test_shutdown_clears_flag(self, engine: AgentEngine) -> None:
        await engine.initialize()
        await engine.shutdown()
        assert engine.is_initialized is False

    async def test_shutdown_calls_memory_manager(self, engine: AgentEngine, mock_memory_manager) -> None:
        await engine.initialize()
        await engine.shutdown()
        mock_memory_manager.shutdown.assert_awaited_once()

    async def test_shutdown_tolerates_error(self, engine: AgentEngine, mock_memory_manager) -> None:
        await engine.initialize()
        mock_memory_manager.shutdown = AsyncMock(side_effect=Exception("fail"))
        # Should not raise
        await engine.shutdown()
        assert engine.is_initialized is False


class TestAgentEngineRun:
    """Tests for AgentEngine.run() with direct paradigm execution."""

    async def test_run_delegates_to_orchestrator(self, mock_llm, mock_memory_manager, mock_context_builder, mock_action_registry) -> None:
        orchestrator = AsyncMock()
        orchestrator.run = AsyncMock(return_value=AgentResponse(answer="orchestrated"))
        engine = AgentEngine(
            llm_client=mock_llm,
            memory_manager=mock_memory_manager,
            context_builder=mock_context_builder,
            action_registry=mock_action_registry,
            orchestrator=orchestrator,
        )

        result = await engine.run("hello")
        assert result.answer == "orchestrated"
        orchestrator.run.assert_awaited_once_with("hello")

    async def test_run_with_invalid_paradigm(self, engine: AgentEngine) -> None:
        result = await engine.run("task", paradigm="nonexistent")
        assert "Invalid paradigm" in result.answer
        assert len(result.memory_updates) == 0

    async def test_run_stores_in_memory(self, engine: AgentEngine, mock_memory_manager, mock_llm) -> None:
        mock_llm.chat_completion = AsyncMock(return_value='{"answer": "done"}')
        await engine.run("test task", paradigm="react")

        mock_memory_manager.store.assert_awaited_once()
        call_kwargs = mock_memory_manager.store.call_args
        assert call_kwargs.kwargs.get("memory_type") == "episodic" or call_kwargs[1].get("memory_type") == "episodic"

    async def test_run_returns_agent_response(self, engine: AgentEngine, mock_llm) -> None:
        mock_llm.chat_completion = AsyncMock(return_value='{"answer": "result"}')
        result = await engine.run("task", paradigm="react")
        assert isinstance(result, AgentResponse)
        assert result.answer == "result"

    async def test_run_with_conversation_history(self, engine: AgentEngine, mock_llm) -> None:
        mock_llm.chat_completion = AsyncMock(return_value='{"answer": "ok"}')
        history = [{"role": "user", "content": "prev question"}, {"role": "assistant", "content": "prev answer"}]
        result = await engine.run("new question", paradigm="react", conversation_history=history)
        assert result.answer == "ok"

    async def test_run_context_builder_failure_graceful(self, engine: AgentEngine, mock_context_builder, mock_llm) -> None:
        mock_context_builder.build = AsyncMock(side_effect=Exception("context fail"))
        mock_llm.chat_completion = AsyncMock(return_value='{"answer": "ok"}')
        result = await engine.run("task", paradigm="react")
        assert result.answer == "ok"

    async def test_run_memory_store_failure_graceful(self, engine: AgentEngine, mock_memory_manager, mock_llm) -> None:
        mock_memory_manager.store = AsyncMock(side_effect=Exception("store fail"))
        mock_llm.chat_completion = AsyncMock(return_value='{"answer": "ok"}')
        result = await engine.run("task", paradigm="react")
        assert result.answer == "ok"
        assert len(result.memory_updates) == 0


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_default_values(self) -> None:
        resp = AgentResponse(answer="test")
        assert resp.answer == "test"
        assert resp.reasoning_trace == []
        assert resp.memory_updates == []

    def test_with_all_fields(self) -> None:
        resp = AgentResponse(
            answer="answer",
            reasoning_trace=[{"type": "thought", "content": "thinking"}],
            memory_updates=["id-1", "id-2"],
        )
        assert len(resp.reasoning_trace) == 1
        assert len(resp.memory_updates) == 2
