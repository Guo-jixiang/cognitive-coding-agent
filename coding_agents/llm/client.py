"""Unified async LLM client — based on OpenAI native API.

Supports multiple LLM providers via OpenAI-compatible endpoints. Configuration
follows a "parameter-first, env-var-fallback" pattern. Streaming is the default
for better user experience.

Design principles:
    - Constructor args take priority; env vars are fallback.
    - Streaming response by default for better UX.
    - Multi-provider support with auto-detection.
    - Unified calling interface.

Supported providers:
    openai, deepseek, qwen, modelscope, kimi, zhipu, ollama, vllm, local, auto
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from openai import APIError, APITimeoutError, AsyncOpenAI

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = Literal[
    "openai", "deepseek", "qwen", "modelscope",
    "kimi", "zhipu", "ollama", "vllm", "local", "auto",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Typed exception for LLM client errors.

    Contains provider details and the original error for debugging.

    Attributes:
        provider_url: The base URL of the LLM provider.
        model_id: The model identifier used in the request.
        original_error: The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        provider_url: str = "",
        model_id: str = "",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_url = provider_url
        self.model_id = model_id
        self.original_error = original_error


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "modelscope": {
        "base_url": "https://api-inference.modelscope.cn/v1/",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "key_env": "MODELSCOPE_API_KEY",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "key_env": "KIMI_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
        "key_env": "ZHIPU_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "key_env": "OLLAMA_API_KEY",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "key_env": "VLLM_API_KEY",
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "model": "local-model",
        "key_env": "LLM_API_KEY",
    },
    "auto": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "key_env": "LLM_API_KEY",
    },
}


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Async LLM client for the Cognitive Coding Agent.

    Designed after the HelloAgents pattern: parameter-first with env-var
    fallback, multi-provider auto-detection, and streaming as default.

    Usage::

        client = LLMClient()  # auto-detect from env vars
        # or explicit:
        client = LLMClient(provider="qwen", model="qwen-plus")

        # Non-streaming
        answer = await client.invoke(messages)

        # Streaming
        async for chunk in client.think(messages):
            print(chunk, end="")

    Args:
        model: Model name. Falls back to LLM_MODEL_ID env var, then provider default.
        api_key: API key. Falls back to provider-specific env var, then LLM_API_KEY.
        base_url: Service URL. Falls back to LLM_BASE_URL, then provider default.
        provider: LLM provider. If None, auto-detected from env vars / base_url.
        temperature: Sampling temperature. Defaults to 0.7.
        max_tokens: Maximum tokens in response. None = provider default.
        timeout: Request timeout in seconds. Falls back to LLM_TIMEOUT (default 60).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: SUPPORTED_PROVIDERS | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self._kwargs = kwargs

        # Auto-detect provider
        self.provider: str = provider or self._auto_detect_provider(api_key, base_url)

        # Resolve credentials based on provider
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # Resolve model
        self.model = model or os.getenv("LLM_MODEL_ID") or self._get_default_model()

        # Validate
        if not self.api_key:
            raise LLMClientError(
                "API key is required. Set LLM_API_KEY or provider-specific env var.",
                provider_url=self.base_url,
                model_id=self.model,
            )
        if not self.base_url:
            raise LLMClientError(
                "Base URL is required. Set LLM_BASE_URL or specify a provider.",
                provider_url="",
                model_id=self.model,
            )

        # Lazy client creation
        self._client: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def provider_url(self) -> str:
        """The base URL of the configured LLM provider."""
        return self.base_url

    @property
    def model_id(self) -> str:
        """The configured model identifier."""
        return self.model

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def think(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM (primary method).

        Calls the LLM with streaming enabled and yields text chunks as
        they arrive. This is the preferred method for interactive use.

        Args:
            messages: Conversation messages (ChatMessage or dict format).
            temperature: Override temperature for this call.

        Yields:
            Text chunks as they arrive from the LLM.

        Raises:
            LLMClientError: If the request fails or times out.
        """
        client = self._get_client()
        formatted = self._format_messages(messages)

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=formatted,  # type: ignore[arg-type]
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:  # type: ignore[union-attr]
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except APITimeoutError as e:
            raise LLMClientError(
                f"Request timed out after {self.timeout}s",
                provider_url=self.base_url,
                model_id=self.model,
                original_error=e,
            ) from e
        except APIError as e:
            raise LLMClientError(
                f"API error: {e.message}",
                provider_url=self.base_url,
                model_id=self.model,
                original_error=e,
            ) from e
        except Exception as e:
            raise LLMClientError(
                f"Unexpected error: {e}",
                provider_url=self.base_url,
                model_id=self.model,
                original_error=e,
            ) from e

    async def invoke(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Non-streaming LLM call, returns the complete response.

        Args:
            messages: Conversation messages (ChatMessage or dict format).
            **kwargs: Override temperature, max_tokens for this call.

        Returns:
            The complete response text from the LLM.

        Raises:
            LLMClientError: If the request fails or times out.
        """
        client = self._get_client()
        formatted = self._format_messages(messages)

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=formatted,  # type: ignore[arg-type]
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=False,
            )
            return response.choices[0].message.content or ""  # type: ignore[union-attr]
        except APITimeoutError as e:
            raise LLMClientError(
                f"Request timed out after {self.timeout}s",
                provider_url=self.base_url,
                model_id=self.model,
                original_error=e,
            ) from e
        except APIError as e:
            raise LLMClientError(
                f"API error: {e.message}",
                provider_url=self.base_url,
                model_id=self.model,
                original_error=e,
            ) from e
        except Exception as e:
            raise LLMClientError(
                f"Unexpected error: {e}",
                provider_url=self.base_url,
                model_id=self.model,
                original_error=e,
            ) from e

    async def chat_completion(
        self, messages: list[ChatMessage], *, stream: bool = False
    ) -> str:
        """Compatibility method — delegates to invoke() or think().

        Args:
            messages: List of ChatMessage instances.
            stream: If True, internally streams and concatenates chunks.

        Returns:
            The complete response text.
        """
        if stream:
            chunks: list[str] = []
            async for chunk in self.think(messages):
                chunks.append(chunk)
            return "".join(chunks)
        return await self.invoke(messages)

    async def chat_completion_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        """Compatibility method — delegates to think().

        Args:
            messages: List of ChatMessage instances.

        Yields:
            Text chunks from the LLM.
        """
        async for chunk in self.think(messages):
            yield chunk

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _get_client(self) -> AsyncOpenAI:
        """Get or lazily create the AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=float(self.timeout),
            )
        return self._client

    @staticmethod
    def _format_messages(
        messages: list[ChatMessage] | list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Normalize messages to dict format for the OpenAI SDK."""
        result: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                result.append({"role": m.role, "content": m.content})
            else:
                result.append(m)
        return result

    def _auto_detect_provider(
        self, api_key: str | None, base_url: str | None
    ) -> str:
        """Auto-detect LLM provider from env vars and credentials.

        Detection logic:
            0. Check LLM_PROVIDER env var (explicit override).
            1. Check provider-specific env vars.
            2. Infer from API key format.
            3. Infer from base_url.
            4. Default to "auto" (uses LLM_* generic vars).
        """
        # 0. Explicit override via LLM_PROVIDER
        explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
        if explicit and explicit in _PROVIDER_DEFAULTS:
            return explicit

        # 1. Check provider-specific env vars
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # 2. Infer from API key format
        actual_key = api_key or os.getenv("LLM_API_KEY", "")
        if actual_key:
            key_lower = actual_key.lower()
            if actual_key.startswith("ms-"):
                return "modelscope"
            if key_lower == "ollama":
                return "ollama"
            if key_lower == "vllm":
                return "vllm"
            if key_lower == "local":
                return "local"

        # 3. Infer from base_url
        actual_url = base_url or os.getenv("LLM_BASE_URL", "")
        if actual_url:
            url_lower = actual_url.lower()
            if "api.openai.com" in url_lower:
                return "openai"
            if "api.deepseek.com" in url_lower:
                return "deepseek"
            if "dashscope.aliyuncs.com" in url_lower:
                return "qwen"
            if "modelscope.cn" in url_lower:
                return "modelscope"
            if "moonshot.cn" in url_lower:
                return "kimi"
            if "bigmodel.cn" in url_lower:
                return "zhipu"
            if "xiaomimimo.com" in url_lower:
                return "auto"
            if "localhost" in url_lower or "127.0.0.1" in url_lower:
                if ":11434" in url_lower:
                    return "ollama"
                return "local"

        # 4. Default
        return "auto"

    def _resolve_credentials(
        self, api_key: str | None, base_url: str | None
    ) -> tuple[str, str]:
        """Resolve API key and base URL based on provider."""
        provider_cfg = _PROVIDER_DEFAULTS.get(self.provider, _PROVIDER_DEFAULTS["auto"])

        # API key: explicit arg > provider env var > LLM_API_KEY
        resolved_key: str = (
            api_key
            or os.getenv(provider_cfg["key_env"], "")
            or os.getenv("LLM_API_KEY", "")
            or ""
        )

        # Special cases for local providers that don't need real keys
        if not resolved_key and self.provider in ("ollama", "vllm", "local"):
            resolved_key = self.provider

        # Base URL: explicit arg > LLM_BASE_URL > provider default
        resolved_url = (
            base_url
            or os.getenv("LLM_BASE_URL", "")
            or provider_cfg["base_url"]
        )

        return resolved_key, resolved_url

    def _get_default_model(self) -> str:
        """Get the default model for the detected provider."""
        provider_cfg = _PROVIDER_DEFAULTS.get(self.provider, _PROVIDER_DEFAULTS["auto"])
        return provider_cfg["model"]


__all__ = ["ChatMessage", "LLMClient", "LLMClientError", "SUPPORTED_PROVIDERS"]
