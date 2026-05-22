"""Property-based tests for MemoryItem serialization round-trip.

# Feature: cognitive-coding-agent, Property 1: MemoryItem Serialization Round-Trip
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from coding_agents.memory.base import MemoryItem

_MemoryTypeLiteral = Literal["working", "episodic", "semantic", "perceptual"]

_MEMORY_TYPES: tuple[_MemoryTypeLiteral, ...] = (
    "working",
    "episodic",
    "semantic",
    "perceptual",
)


@st.composite
def memory_items(draw: st.DrawFn) -> MemoryItem:
    """Generate random valid MemoryItems with varied fields.

    Content is bounded to 1,000 characters (within the 100,000 limit) to keep
    test runs fast while still exercising the round-trip property broadly.
    """
    content: str = draw(st.text(max_size=1000))
    metadata: dict[str, Any] = draw(
        st.dictionaries(
            st.text(max_size=50),
            st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False),
            ),
            max_size=10,
        )
    )
    importance: float = draw(st.floats(min_value=0.0, max_value=1.0))
    created_at: datetime = draw(st.datetimes(timezones=st.just(timezone.utc)))
    last_accessed_at: datetime = draw(st.datetimes(timezones=st.just(timezone.utc)))
    memory_type: _MemoryTypeLiteral = draw(st.sampled_from(_MEMORY_TYPES))

    return MemoryItem(
        id=str(uuid.uuid4()),
        content=content,
        metadata=metadata,
        importance=importance,
        created_at=created_at,
        last_accessed_at=last_accessed_at,
        memory_type=memory_type,
    )


# Feature: cognitive-coding-agent, Property 1: MemoryItem Serialization Round-Trip
# Validates: Requirements 1.3
@given(item=memory_items())
@settings(max_examples=100)
def test_memory_item_serialization_round_trip(item: MemoryItem) -> None:
    """A MemoryItem is equivalent to itself after to_dict/from_dict round trip."""
    serialized: dict[str, Any] = item.to_dict()
    restored: MemoryItem = MemoryItem.from_dict(serialized)
    assert restored == item
