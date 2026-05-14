# Isolated Non-LLM Tests

This directory contains tests that do not call the real LLM/API, including unit, mock, and schema-only tests.

They are intentionally excluded from the default pytest collection in `pytest.ini`.

Run them explicitly when needed:

```powershell
python -m pytest tests/isolated_non_llm
```

The default test path is now `tests/integration`, which is the real pipeline/API-facing test area.
