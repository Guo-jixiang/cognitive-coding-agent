"""Embedding service with multi-backend fallback chain.

This module provides a unified embedding interface (`EmbeddingService`) that
generates fixed-dimension, L2-normalized vector representations of text.
It implements the strategy pattern with a three-tier fallback chain:

    DashScope (cloud API)
        -> LocalTransformer (sentence-transformers)
            -> TF-IDF (scikit-learn TfidfVectorizer)

Each backend implements `BaseEmbeddingBackend` and reports its availability
through `is_available()`. When the primary backend is unavailable or fails,
the service transparently falls back to the next available backend. If every
backend is exhausted, an `EmbeddingUnavailableError` is raised.

Public API:
    - ``BaseEmbeddingBackend``: Abstract base for embedding backends.
    - ``DashScopeEmbedding``: DashScope (Alibaba Cloud) backend via OpenAI SDK.
    - ``LocalTransformerEmbedding``: Local sentence-transformers backend.
    - ``TFIDFEmbedding``: Lightweight TF-IDF backend.
    - ``EmbeddingService``: Unified service with fallback chain.
    - ``EmbeddingUnavailableError``: Raised when all backends fail.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DASHSCOPE_DEFAULT_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_DEFAULT_MODEL: str = "text-embedding-v3"
_DASHSCOPE_DEFAULT_DIMENSION: int = 1024

_LOCAL_DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
_LOCAL_DEFAULT_DIMENSION: int = 384

_TFIDF_DEFAULT_DIMENSION: int = 512


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingUnavailableError(RuntimeError):
    """Raised when every configured embedding backend is unavailable.

    Attributes:
        attempts: A list of ``(backend_name, reason)`` tuples describing why
            each backend in the fallback chain could not produce an embedding.
    """

    def __init__(self, attempts: list[tuple[str, str]]) -> None:
        """Initialize the error with a list of backend attempt records.

        Args:
            attempts: Ordered list of ``(backend_name, reason)`` tuples
                explaining why each backend could not be used.
        """
        self.attempts = attempts
        details = "; ".join(f"{name}: {reason}" for name, reason in attempts) or "no backends"
        super().__init__(f"All embedding backends are unavailable. Attempts: {details}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of ``vector`` as ``float32``.

    If the input has zero norm (which can occur for extremely degenerate
    inputs), a deterministic unit vector is returned instead so that the
    output always satisfies ``np.linalg.norm == 1.0``.

    Args:
        vector: Input vector. Must be 1-dimensional or a flat array-like.

    Returns:
        A 1-D numpy ``float32`` array with L2 norm equal to 1.0.
    """
    arr = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        # Degenerate input: synthesize a deterministic unit vector so callers
        # can rely on the L2-norm-equals-1 invariant.
        result = np.zeros_like(arr)
        if result.size > 0:
            result[0] = 1.0
        return result
    return (arr / norm).astype(np.float32, copy=False)


def _pad_or_truncate(vector: np.ndarray, dimension: int) -> np.ndarray:
    """Pad with zeros or truncate ``vector`` so it has length ``dimension``.

    Args:
        vector: 1-D input vector of arbitrary length.
        dimension: Target output dimension. Must be positive.

    Returns:
        A 1-D ``float32`` vector of length ``dimension``.
    """
    arr = np.asarray(vector, dtype=np.float32).ravel()
    if arr.shape[0] == dimension:
        return arr
    if arr.shape[0] > dimension:
        return arr[:dimension].astype(np.float32, copy=False)
    padded = np.zeros(dimension, dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def _validate_text(text: str) -> None:
    """Validate a single text input before any backend dispatch.

    Args:
        text: The text to validate.

    Raises:
        ValueError: If ``text`` is the empty string.
    """
    if text == "":
        raise ValueError("Embedding input must be a non-empty string.")


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class BaseEmbeddingBackend(ABC):
    """Abstract base class for all embedding backends.

    Subclasses implement a specific embedding strategy (cloud API, local
    model, or lightweight vectorization) and expose a uniform async
    interface so that ``EmbeddingService`` can swap them at runtime.
    """

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a fixed-dimension vector.

        Args:
            text: The text to embed.

        Returns:
            A 1-D numpy array of length ``get_dimension()``.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed (in order).

        Returns:
            A list of 1-D numpy arrays in the same order as ``texts``.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimensionality produced by this backend."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this backend is currently usable.

        Implementations should perform only cheap checks (e.g., env var
        presence or import availability) and must not perform network calls.
        """


# ---------------------------------------------------------------------------
# DashScope backend
# ---------------------------------------------------------------------------


class DashScopeEmbedding(BaseEmbeddingBackend):
    """Embedding backend backed by DashScope's OpenAI-compatible endpoint.

    Reads the following environment variables (with explicit constructor
    arguments taking precedence):

    - ``DASHSCOPE_API_KEY``: API key. Backend is unavailable if missing/empty.
    - ``DASHSCOPE_BASE_URL``: Base URL of the OpenAI-compatible endpoint.
      Defaults to ``"https://dashscope.aliyuncs.com/compatible-mode/v1"``.
    - ``DASHSCOPE_MODEL``: Model identifier. Defaults to ``"text-embedding-v3"``.

    The output dimension is fixed at 1024 (the ``text-embedding-v3`` default).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int = _DASHSCOPE_DEFAULT_DIMENSION,
    ) -> None:
        """Initialize the DashScope backend.

        Args:
            api_key: Explicit API key. Falls back to ``DASHSCOPE_API_KEY``.
            base_url: Explicit base URL. Falls back to ``DASHSCOPE_BASE_URL``,
                then to the DashScope default.
            model: Explicit model name. Falls back to ``DASHSCOPE_MODEL``,
                then to ``"text-embedding-v3"``.
            dimension: Output embedding dimension. Defaults to 1024.
        """
        self._api_key = api_key if api_key is not None else os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = (
            base_url
            if base_url is not None
            else os.environ.get("DASHSCOPE_BASE_URL", _DASHSCOPE_DEFAULT_BASE_URL)
        )
        self._model = (
            model
            if model is not None
            else os.environ.get("DASHSCOPE_MODEL", _DASHSCOPE_DEFAULT_MODEL)
        )
        self._dimension = dimension
        self._client: Any = None

    def is_available(self) -> bool:
        """Return True iff a non-empty API key is configured."""
        return bool(self._api_key)

    def get_dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    def _get_client(self) -> Any:
        """Lazily construct the AsyncOpenAI client on first use."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text via the DashScope API.

        Args:
            text: The text to embed.

        Returns:
            A 1-D L2-normalized ``float32`` vector of length
            ``get_dimension()``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts via the DashScope API.

        The response data is returned in the same order as ``texts``.
        Each output vector is L2-normalized to unit length.

        Args:
            texts: Texts to embed in order.

        Returns:
            A list of 1-D L2-normalized ``float32`` vectors.
        """
        client = self._get_client()
        response = await client.embeddings.create(model=self._model, input=texts)
        vectors: list[np.ndarray] = []
        for item in response.data:
            raw = np.asarray(item.embedding, dtype=np.float32)
            vectors.append(_normalize_vector(raw))
        return vectors


# ---------------------------------------------------------------------------
# Local sentence-transformer backend
# ---------------------------------------------------------------------------


class LocalTransformerEmbedding(BaseEmbeddingBackend):
    """Embedding backend backed by a local sentence-transformers model.

    The model is **lazy-loaded** on first call to ``embed`` or ``embed_batch``
    so that constructing this backend (e.g., during application startup or
    in ``EmbeddingService.__init__``) is cheap.

    The default model is ``"sentence-transformers/all-MiniLM-L6-v2"`` which
    produces 384-dimensional embeddings.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int = _LOCAL_DEFAULT_DIMENSION,
    ) -> None:
        """Initialize the local transformer backend.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                ``"sentence-transformers/all-MiniLM-L6-v2"``.
            dimension: Expected output dimension. Defaults to 384, the
                ``all-MiniLM-L6-v2`` default.
        """
        self._model_name = model_name or _LOCAL_DEFAULT_MODEL
        self._dimension = dimension
        self._model: Any = None

    def is_available(self) -> bool:
        """Return True iff the ``sentence_transformers`` package is importable."""
        return importlib.util.find_spec("sentence_transformers") is not None

    def get_dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    def _get_model(self) -> Any:
        """Lazily load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if isinstance(actual_dim, int) and actual_dim > 0:
                self._dimension = actual_dim
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text using the local model.

        Args:
            text: The text to embed.

        Returns:
            A 1-D L2-normalized ``float32`` vector of length
            ``get_dimension()``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts using the local model.

        ``SentenceTransformer.encode`` is synchronous and CPU/GPU bound, so
        it is dispatched onto a worker thread via ``asyncio.to_thread`` to
        avoid blocking the event loop.

        Args:
            texts: Texts to embed in order.

        Returns:
            A list of 1-D L2-normalized ``float32`` vectors.
        """
        model = self._get_model()

        def _encode() -> Any:
            return model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)

        encoded = await asyncio.to_thread(_encode)
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return [_normalize_vector(matrix[i]) for i in range(matrix.shape[0])]


# ---------------------------------------------------------------------------
# TF-IDF backend
# ---------------------------------------------------------------------------


class TFIDFEmbedding(BaseEmbeddingBackend):
    """Lightweight embedding backend using scikit-learn ``TfidfVectorizer``.

    This backend is the final fallback in the chain. It is always available
    (scikit-learn is a hard dependency) and requires no network access.

    A rolling corpus of every text seen by ``embed`` / ``embed_batch`` is
    maintained internally. On each call the vectorizer is (re-)fit on the
    accumulated corpus with ``max_features = dimension`` so the raw output
    width never exceeds the target dimension. Each transformed row is then
    padded with zeros (or truncated) to exactly ``dimension`` dimensions and
    L2-normalized.
    """

    def __init__(self, dimension: int = _TFIDF_DEFAULT_DIMENSION) -> None:
        """Initialize the TF-IDF backend.

        Args:
            dimension: Fixed output dimension after padding/truncation.
                Must be positive. Defaults to 512.
        """
        if dimension <= 0:
            raise ValueError(f"TF-IDF dimension must be positive, got {dimension}")
        self._dimension = dimension
        self._corpus: list[str] = []
        self._corpus_set: set[str] = set()

    def is_available(self) -> bool:
        """The TF-IDF backend is always available (sklearn is a hard dep)."""
        return True

    def get_dimension(self) -> int:
        """Return the fixed embedding dimension."""
        return self._dimension

    def _extend_corpus(self, texts: list[str]) -> None:
        """Append previously-unseen texts to the rolling corpus."""
        for text in texts:
            if text not in self._corpus_set:
                self._corpus_set.add(text)
                self._corpus.append(text)

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text using a TF-IDF vectorization.

        Args:
            text: The text to embed.

        Returns:
            A 1-D L2-normalized ``float32`` vector of length
            ``get_dimension()``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts using a TF-IDF vectorization.

        The vectorizer is fit on the rolling corpus (extended with the
        input texts) and the inputs are then transformed. Each row is
        padded or truncated to the configured dimension and L2-normalized.

        Args:
            texts: Texts to embed in order.

        Returns:
            A list of 1-D L2-normalized ``float32`` vectors of length
            ``get_dimension()``.
        """
        from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
            TfidfVectorizer,
        )

        self._extend_corpus(texts)

        def _vectorize() -> np.ndarray:
            vectorizer = TfidfVectorizer(max_features=self._dimension)
            try:
                vectorizer.fit(self._corpus)
                sparse_matrix = vectorizer.transform(texts)
                return np.asarray(sparse_matrix.toarray(), dtype=np.float32)
            except ValueError:
                # Empty vocabulary (e.g., inputs are all single chars or all
                # stop words). sklearn's default token pattern requires
                # words of >= 2 characters, so a corpus of {"a"} produces
                # no tokens. Fall back to a deterministic per-text vector
                # derived from a stable hash of each input so the L2-norm-
                # equals-1 invariant still holds and equal inputs yield
                # equal vectors (determinism).
                rows = np.zeros((len(texts), self._dimension), dtype=np.float32)
                for i, text in enumerate(texts):
                    # Use a stable hash to pick a single non-zero coordinate.
                    # SHA-256 keeps determinism across processes/restarts.
                    import hashlib

                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    index = int.from_bytes(digest[:4], "big") % self._dimension
                    rows[i, index] = 1.0
                return rows

        dense = await asyncio.to_thread(_vectorize)
        return [
            _normalize_vector(_pad_or_truncate(dense[i], self._dimension))
            for i in range(dense.shape[0])
        ]


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Unified embedding service implementing a fallback chain over backends.

    The service tries each backend in order. A backend is skipped if its
    ``is_available()`` returns False or if any of its calls raise. If no
    backend produces an embedding, ``EmbeddingUnavailableError`` is raised.

    All output vectors are guaranteed to be L2-normalized to unit length.
    """

    def __init__(self, backends: list[BaseEmbeddingBackend] | None = None) -> None:
        """Initialize the service with an explicit backend list or defaults.

        Args:
            backends: Optional explicit list of backends in priority order.
                If None, the default chain is used:
                ``[DashScopeEmbedding, LocalTransformerEmbedding, TFIDFEmbedding]``.
        """
        if backends is None:
            backends = [
                DashScopeEmbedding(),
                LocalTransformerEmbedding(),
                TFIDFEmbedding(),
            ]
        self._backends: list[BaseEmbeddingBackend] = backends

    @property
    def backends(self) -> list[BaseEmbeddingBackend]:
        """Return a copy of the configured backend list."""
        return list(self._backends)

    def get_dimension(self) -> int:
        """Return the dimension of the first available backend.

        If no backend reports availability, the dimension of the first
        configured backend is returned as a best-effort default.

        Raises:
            EmbeddingUnavailableError: If no backends are configured at all.
        """
        if not self._backends:
            raise EmbeddingUnavailableError([])
        for backend in self._backends:
            if backend.is_available():
                return backend.get_dimension()
        return self._backends[0].get_dimension()

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text, falling back through the backend chain.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            A 1-D L2-normalized numpy array.

        Raises:
            ValueError: If ``text`` is the empty string.
            EmbeddingUnavailableError: If every backend fails or is unavailable.
        """
        _validate_text(text)
        attempts: list[tuple[str, str]] = []
        for backend in self._backends:
            name = type(backend).__name__
            if not backend.is_available():
                attempts.append((name, "is_available() returned False"))
                continue
            try:
                vector = await backend.embed(text)
            except Exception as exc:  # noqa: BLE001 - intentionally broad for fallback
                attempts.append((name, f"{type(exc).__name__}: {exc}"))
                continue
            return _normalize_vector(vector)
        raise EmbeddingUnavailableError(attempts)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts, falling back through the backend chain.

        Args:
            texts: The texts to embed. Each must be non-empty.

        Returns:
            A list of 1-D L2-normalized numpy arrays in the same order as
            ``texts``. An empty input list returns an empty output list
            without invoking any backend.

        Raises:
            ValueError: If any element of ``texts`` is the empty string.
            EmbeddingUnavailableError: If every backend fails or is unavailable.
        """
        for text in texts:
            _validate_text(text)
        if not texts:
            return []
        attempts: list[tuple[str, str]] = []
        for backend in self._backends:
            name = type(backend).__name__
            if not backend.is_available():
                attempts.append((name, "is_available() returned False"))
                continue
            try:
                vectors = await backend.embed_batch(texts)
            except Exception as exc:  # noqa: BLE001 - intentionally broad for fallback
                attempts.append((name, f"{type(exc).__name__}: {exc}"))
                continue
            return [_normalize_vector(v) for v in vectors]
        raise EmbeddingUnavailableError(attempts)


__all__ = [
    "BaseEmbeddingBackend",
    "DashScopeEmbedding",
    "EmbeddingService",
    "EmbeddingUnavailableError",
    "LocalTransformerEmbedding",
    "TFIDFEmbedding",
]
