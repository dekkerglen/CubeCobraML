"""Entry point for `uvicorn backend.main:app`."""

from backend.app import app  # noqa: F401  re-exported for uvicorn
