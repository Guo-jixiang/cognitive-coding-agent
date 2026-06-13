"""Unit tests for LLMClient."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agents.llm.client import ChatMessage, LLMClient, LLMClientError


# ---------------------------------------------------------------------------
# Tests for ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_creation(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_system_role(self) -> None:
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_assistant_role(self) -> None:
        msg = ChatMessage(role="assistant", content="I can help.")
        assert msg.role == "assistant"


# ---------------------------------------------------------------------------
# Tests for LLMClientError
# ---------------------------------------------------------------------------


class TestLLMClientError:
    """Tests for LLMClientError exception."""

    def test_basic_creation(self) -> None:
        err = LLMClientError("something failed")
        assert str(err) == "something failed"
        assert err.provider_url == ""
        assert err.model_id == ""
        assert err.original_error is None

    def test_with_all_fields(self) -> None:
        original = ValueError("orig")
        err = LLMClientError(
            "fail",
            provider_url="http://localhost",
            model_id="test-model",
            original_error=original,
        )
        assert err.provider_url == "http://localhost"
        assert err.model_id == "test-model"
        assert err.original_error is original


# ---------------------------------------------------------------------------
# Tests for LLMClient construction
# ---------------------------------------------------------------------------


class TestLLMClientConstruction:
    """Tests for LLMClient initialization and auto-detection."""

    def test_explicit_provider_and_key(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert client.provider == "openai"
        assert client.api_key == "sk-test"

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMClientError, match="API key"):
                LLMClient(provider="openai", api_key=None, base_url="https://api.openai.com/v1")

    def test_missing_base_url_falls_back_to_provider_default(self) -> None:
        # base_url=None falls back to the provider's default URL from _PROVIDER_DEFAULTS
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(provider="openai", api_key="sk-test", base_url=None)
            assert "openai.com" in client.base_url

    def test_ollama_gets_default_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(provider="ollama")
            assert client.api_key == "ollama"

    def test_vllm_gets_default_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(provider="vllm")
            assert client.api_key == "vllm"

    def test_provider_url_property(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert client.provider_url == "https://api.openai.com/v1"

    def test_model_id_property(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-4",
        )
        assert client.model_id == "gpt-4"

    def test_default_model_from_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(provider="deepseek", api_key="test-key")
            assert client.model == "deepseek-chat"


# ---------------------------------------------------------------------------
# Tests for _auto_detect_provider
# ---------------------------------------------------------------------------


class TestAutoDetectProvider:
    """Tests for provider auto-detection logic."""

    def test_detect_openai_from_env(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-xxx"}, clear=False):
            client = LLMClient()
            assert client.provider == "openai"

    def test_detect_deepseek_from_env(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-xxx"}, clear=False):
            client = LLMClient()
            assert client.provider == "deepseek"

    def test_detect_qwen_from_env(self) -> None:
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "ds-xxx"}, clear=False):
            client = LLMClient()
            assert client.provider == "qwen"

    def test_detect_from_key_format_modelscope(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "ms-test"}, clear=True):
            client = LLMClient(base_url="https://example.com/v1")
            assert client.provider == "modelscope"

    def test_detect_from_key_format_ollama(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "ollama"}, clear=True):
            client = LLMClient(base_url="https://example.com/v1")
            assert client.provider == "ollama"

    def test_detect_from_base_url_openai(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}, clear=True):
            client = LLMClient(base_url="https://api.openai.com/v1")
            assert client.provider == "openai"

    def test_detect_from_base_url_deepseek(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}, clear=True):
            client = LLMClient(base_url="https://api.deepseek.com")
            assert client.provider == "deepseek"

    def test_detect_from_base_url_localhost_ollama(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}, clear=True):
            client = LLMClient(base_url="http://localhost:11434/v1")
            assert client.provider == "ollama"

    def test_detect_from_base_url_localhost_generic(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}, clear=True):
            client = LLMClient(base_url="http://localhost:8000/v1")
            assert client.provider == "local"

    def test_default_to_auto(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}, clear=True):
            client = LLMClient(base_url="https://unknown.example.com/v1")
            assert client.provider == "auto"


# ---------------------------------------------------------------------------
# Tests for _format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    """Tests for message formatting."""

    def test_chat_message_to_dict(self) -> None:
        messages = [ChatMessage(role="user", content="hello")]
        result = LLMClient._format_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_dict_passthrough(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = LLMClient._format_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_mixed_format(self) -> None:
        messages = [
            ChatMessage(role="system", content="sys"),
            {"role": "user", "content": "usr"},
        ]
        result = LLMClient._format_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "usr"}


# ---------------------------------------------------------------------------
# Tests for invoke / chat_completion
# ---------------------------------------------------------------------------


class TestLLMClientInvoke:
    """Tests for the invoke method (requires mocking the OpenAI client)."""

    async def test_invoke_returns_content(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="response text"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client._client = mock_client

        result = await client.invoke([ChatMessage(role="user", content="hi")])
        assert result == "response text"

    async def test_chat_completion_delegates_to_invoke(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="via invoke"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client._client = mock_client

        result = await client.chat_completion([ChatMessage(role="user", content="test")])
        assert result == "via invoke"

    async def test_close_releases_client(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        mock_client = AsyncMock()
        client._client = mock_client

        await client.close()
        mock_client.close.assert_awaited_once()
        assert client._client is None

    async def test_close_noop_when_no_client(self) -> None:
        client = LLMClient(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        # Should not raise
        await client.close()
