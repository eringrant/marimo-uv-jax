# On Windows, run recipes through Git Bash (for the clean recipe's rm/find).
set windows-shell := ["bash", "-cu"]

# List available commands
default:
  @just --list

# Install dependencies and pre-commit hooks
setup:
  uv sync --all-groups
  uv run prek install

# Run all pre-commit hooks (ruff format, ruff check, etc.)
lint:
  uv run prek run --all-files

# Type-check
ty:
  uv run --all-groups ty check

# Run tests
test:
  uv run pytest

# Open the marimo editor
marimo:
  uv run marimo edit

# Present a notebook read-only (pick a "slides" layout in the editor for a slideshow)
present NOTEBOOK:
  uv run marimo run {{NOTEBOOK}}

# Remove build and cache artifacts
clean:
  rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .ruff_cache/ .ty_cache/
  find . -type d -name __pycache__ -exec rm -rf {} +

# Run the full CI checks locally
ci: lint ty test
