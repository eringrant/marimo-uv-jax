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
Path.mkdir(TMP_DIR, parents=True, exist_ok=True)

scratch_home = os.environ.get("SCRATCH_HOME")
SCRATCH_DIR = (
  Path(scratch_home, "marimo-uv-jax") if scratch_home is not None else TMP_DIR
)
Path.mkdir(SCRATCH_DIR, parents=True, exist_ok=True)

del __package_path

__all__ = [
  "__version__",
  "DATA_DIR",
  "EXPERIMENTS_DIR",
  "REPRODUCTIONS_DIR",
  "TMP_DIR",
  "SCRATCH_DIR",
]
