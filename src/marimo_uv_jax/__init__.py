"""Package-wide constants and configuration."""

import os
from pathlib import Path

__version__ = "0.1.0"

TMP_DIR = Path("/tmp", "marimo-uv-jax")  # noqa: S108

scratch_home = os.environ.get("SCRATCH_HOME")
# `or None`: an unset OR empty SCRATCH_HOME falls back to TMP_DIR (an empty
# string would otherwise yield a cwd-relative scratch path).
SCRATCH_DIR = Path(scratch_home, "marimo-uv-jax") if scratch_home else TMP_DIR


def ensure_scratch_dir() -> Path:
  """Create the scratch directory if needed and return it.

  Call this before writing scratch output. Directory creation is deferred so
  that importing the package has no filesystem side effects.

  Returns:
    The scratch directory path.
  """
  Path.mkdir(SCRATCH_DIR, parents=True, exist_ok=True)
  return SCRATCH_DIR


__all__ = [
  "__version__",
  "TMP_DIR",
  "SCRATCH_DIR",
  "ensure_scratch_dir",
]
