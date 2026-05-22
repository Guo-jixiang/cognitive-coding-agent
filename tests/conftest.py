"""Shared test fixtures for the Cognitive Coding Agent test suite."""

from typing import Any

import pytest


@pytest.fixture
def sample_memory_item_data() -> dict[str, Any]:
    """Provide sample data for creating a MemoryItem in tests."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "Sample memory content for testing.",
        "metadata": {"source": "test", "tags": ["unit-test"]},
        "importance": 0.7,
        "memory_type": "working",
    }
