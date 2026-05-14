"""Isolated non-LLM test configuration.

Tests under this directory are intentionally excluded from the repository
default pytest collection. Run them explicitly with:

    python -m pytest tests/isolated_non_llm
"""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.non_llm)
