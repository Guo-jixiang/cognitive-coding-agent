"""Property-based tests for the Embedding service.

These tests validate Properties 9 and 10 from the Cognitive Coding Agent
design document using only the deterministic, dependency-free
``TFIDFEmbedding`` backend. Restricting the property tests to TF-IDF keeps
the suite fast and self-contained: no DashScope API calls, no
``sentence-transformers`` model loads.

Validates: Requirements 6.2, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import asyncio
import string

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from coding_agents.memory.embedding import TFIDFEmbedding

# A small, fixed dimension keeps TF-IDF refits fast across 100 examples.
_TEST_DIMENSION = 64

# Generate non-empty printable ASCII text drawn from letters, digits, and
# spaces. This range is wide enough to exercise tokenizer behaviour while
# remaining narrow enough that ``TfidfVectorizer``'s default token pattern
# (``\b\w\w+\b``) extracts at least some tokens for most inputs. The filter
# guarantees the input is not pure whitespace, which would be rejected by
# ``EmbeddingService`` upstream and is also semantically uninteresting.
text_strategy = st.text(
    alphabet=string.ascii_letters + string.digits + " ",
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")


# Feature: cognitive-coding-agent, Property 9: Embedding Output Invariants
# Validates: Requirements 6.2, 6.6
@given(text=text_strategy)
@settings(max_examples=100, deadline=None)
def test_embedding_output_invariants(text: str) -> None:
    """Output vectors have fixed dimension and unit L2 norm.

    For any non-empty text, ``TFIDFEmbedding.embed`` must return a 1-D
    ``float32`` vector of length ``dimension`` whose L2 norm equals 1.0
    within floating-point tolerance.
    """
    backend = TFIDFEmbedding(dimension=_TEST_DIMENSION)

    vector = asyncio.run(backend.embed(text))

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (_TEST_DIMENSION,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


# Feature: cognitive-coding-agent, Property 10: Embedding Determinism and Batch Equivalence
# Validates: Requirements 6.4, 6.5
@given(text=text_strategy)
@settings(max_examples=100, deadline=None)
def test_embedding_determinism(text: str) -> None:
    """Embedding the same text twice with the same backend yields identical vectors.

    ``TFIDFEmbedding`` maintains a rolling corpus and refits per call, so
    determinism is defined relative to a shared backend instance: once a
    text is in the corpus, subsequent calls see an identical corpus and
    therefore produce an identical vocabulary and identical output.
    """
    backend = TFIDFEmbedding(dimension=_TEST_DIMENSION)

    async def _run() -> tuple[np.ndarray, np.ndarray]:
        first = await backend.embed(text)
        second = await backend.embed(text)
        return first, second

    first, second = asyncio.run(_run())

    assert np.allclose(first, second, atol=1e-5, rtol=0.0)


# Feature: cognitive-coding-agent, Property 10: Embedding Determinism and Batch Equivalence
# Validates: Requirements 6.4, 6.5
@given(texts=st.lists(text_strategy, min_size=1, max_size=8))
@settings(max_examples=100, deadline=None)
def test_embedding_batch_equivalence(texts: list[str]) -> None:
    """``embed_batch(texts)`` equals ``[embed(t) for t in texts]`` for a shared backend.

    To make the comparison meaningful we first call ``embed_batch`` so the
    rolling corpus contains every input text. Subsequent individual
    ``embed`` calls then refit on the *same* corpus, guaranteeing a stable
    vocabulary across both code paths.
    """
    backend = TFIDFEmbedding(dimension=_TEST_DIMENSION)

    async def _run() -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch_results = await backend.embed_batch(texts)
        individual_results = [await backend.embed(t) for t in texts]
        return batch_results, individual_results

    batch_results, individual_results = asyncio.run(_run())

    assert len(batch_results) == len(texts)
    assert len(individual_results) == len(texts)
    for batch_vec, individual_vec in zip(batch_results, individual_results, strict=True):
        assert batch_vec.shape == (_TEST_DIMENSION,)
        assert individual_vec.shape == (_TEST_DIMENSION,)
        assert np.allclose(batch_vec, individual_vec, atol=1e-5, rtol=0.0)
