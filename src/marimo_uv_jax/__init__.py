"""Package-wide constants and configuration."""

import os
from pathlib import Path

__version__ = "0.1.0"

# Package-wide path constants
__package_path = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = Path(__package_path, "data")
EXPERIMENTS_DIR = Path(__package_path, "experiments")
REPRODUCTIONS_DIR = Path(__package_path, "reproductions")
TMP_DIR = Path("/tmp", "marimo-uv-jax")  # noqa: S108

scratch_home = os.environ.get("SCRATCH_HOME")
SCRATCH_DIR = (
  Path(scratch_home, "marimo-uv-jax") if scratch_home is not None else TMP_DIR
)

del __package_path


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
  "DATA_DIR",
  "EXPERIMENTS_DIR",
  "REPRODUCTIONS_DIR",
  "TMP_DIR",
  "SCRATCH_DIR",
  "ensure_scratch_dir",
]
