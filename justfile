# List available commands
default:
  @just --list

# Install all dependencies
install:
  uv sync --all-groups

# Install pre-commit hooks
install-hooks:
  uv run prek install

# Run all pre-commit hooks
lint:
  uv run prek run --all-files

# Run tests
test:
  uv run pytest

# Run type checking
typecheck:
  uv run --group dev ty check

# Format code
format:
  uv run ruff format .

# Check code style
check:
  uv run ruff check .

# Fix code style issues
fix:
  uv run ruff check . --fix

# Run marimo editor
marimo:
  uv run marimo edit

# Run a specific experiment
experiment NAME:
  uv run marimo edit experiments/{{NAME}}.py

# Run a specific reproduction
reproduction NAME:
  uv run marimo edit reproductions/{{NAME}}.py

# Clean build artifacts
clean:
  rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .ruff_cache/ node_modules/
  find . -type d -name __pycache__ -exec rm -rf {} +
  find . -type f -name "*.pyc" -delete

# Run full CI pipeline locally
ci: lint test typecheck

# Setup development environment from scratch
setup: install install-hooks
  @echo "Development environment ready!"
