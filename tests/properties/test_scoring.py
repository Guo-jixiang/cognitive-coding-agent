"""Property-based tests for ScoringMixin formulas.

# Feature: cognitive-coding-agent, Property 3: Relevance Score Formula Correctness
# Feature: cognitive-coding-agent, Property 8: Time Decay Exponential Model
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from coding_agents.memory.base import ScoringMixin

_TOLERANCE: float = 1e-9

# Weight configurations matching the four memory types from design.md.
# Format: (label, sim_weight, time_weight)
_WEIGHT_CONFIGS: tuple[tuple[str, float, float], ...] = (
    ("working", 1.0, 0.0),
    ("episodic", 0.8, 0.2),
    ("semantic", 0.7, 0.3),
    ("perceptual", 0.8, 0.2),
)


# Feature: cognitive-coding-agent, Property 3: Relevance Score Formula Correctness
# Validates: Requirements 2.5, 3.2, 4.2, 5.5, 11.4
@given(
    similarity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    time_factor=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    importance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_relevance_score_formula(
    similarity: float, time_factor: float, importance: float
) -> None:
    """Relevance score matches the documented formula and stays in [0, 1].

    Formula: clamp((similarity * sim_w + time_factor * time_w) * (0.8 + importance * 0.4),
                   0.0, 1.0)
    """
    for label, sim_weight, time_weight in _WEIGHT_CONFIGS:
        raw_score: float = (similarity * sim_weight + time_factor * time_weight) * (
            0.8 + importance * 0.4
        )
        expected: float = max(0.0, min(1.0, raw_score))
        actual: float = ScoringMixin.compute_relevance_score(
            similarity=similarity,
            time_factor=time_factor,
            importance=importance,
            sim_weight=sim_weight,
            time_weight=time_weight,
        )
        assert math.isclose(actual, expected, abs_tol=_TOLERANCE), (
            f"[{label}] sim={similarity}, time={time_factor}, imp={importance}, "
            f"weights=({sim_weight}, {time_weight}): expected {expected}, got {actual}"
        )
        assert 0.0 <= actual <= 1.0, f"[{label}] score {actual} out of [0, 1]"


# Feature: cognitive-coding-agent, Property 8: Time Decay Exponential Model
# Validates: Requirements 5.6, 11.1
@given(
    elapsed_seconds=st.floats(
        min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False
    ),
    decay_rate=st.floats(
        min_value=1e-10, max_value=1e-3, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_time_decay_exponential_model(elapsed_seconds: float, decay_rate: float) -> None:
    """Time decay equals exp(-decay_rate * elapsed_seconds) and is bounded in [0, 1].

    Mathematically the bound is (0, 1], but extreme inputs (decay_rate * elapsed_seconds
    above ~709) cause IEEE 754 underflow to 0.0. The formula equality check is the
    primary assertion; the bound check uses [0, 1] to accommodate underflow.
    """
    actual: float = ScoringMixin.compute_time_decay(
        elapsed_seconds=elapsed_seconds, decay_rate=decay_rate
    )
    expected: float = math.exp(-decay_rate * elapsed_seconds)
    assert math.isclose(actual, expected, abs_tol=_TOLERANCE), (
        f"elapsed={elapsed_seconds}, rate={decay_rate}: expected {expected}, got {actual}"
    )
    assert 0.0 <= actual <= 1.0, f"decay value {actual} out of [0, 1]"


# Feature: cognitive-coding-agent, Property 8: Time Decay Exponential Model
# Validates: Requirements 5.6, 11.1
@given(
    t1=st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
    t2=st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
    decay_rate=st.floats(
        min_value=1e-10, max_value=1e-3, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_time_decay_monotonicity(t1: float, t2: float, decay_rate: float) -> None:
    """For t_earlier <= t_later, time_decay(t_earlier) >= time_decay(t_later)."""
    earlier: float = min(t1, t2)
    later: float = max(t1, t2)
    decay_earlier: float = ScoringMixin.compute_time_decay(
        elapsed_seconds=earlier, decay_rate=decay_rate
    )
    decay_later: float = ScoringMixin.compute_time_decay(
        elapsed_seconds=later, decay_rate=decay_rate
    )
    assert decay_earlier >= decay_later, (
        f"earlier={earlier} (decay={decay_earlier}) should be >= "
        f"later={later} (decay={decay_later})"
    )
