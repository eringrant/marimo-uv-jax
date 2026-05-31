"""Tests for marimo_uv_jax package-level helpers."""

from marimo_uv_jax import SCRATCH_DIR
from marimo_uv_jax import ensure_scratch_dir


def test_ensure_scratch_dir_creates_and_returns():
  """ensure_scratch_dir() materializes the dir and returns its path."""
  path = ensure_scratch_dir()
  assert path == SCRATCH_DIR
  assert path.is_dir()
