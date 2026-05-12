# `marimo` + `uv` + JAX template

A starter template for [marimo](https://marimo.io) notebooks using [uv](https://github.com/astral-sh/uv) for dependency management and [JAX](https://github.com/jax-ml/jax) for numerical computing. This template provides a modern Python development setup with best practices for scientific computing and notebook development.

## Features

- 🚀 Python 3.13+ support
- 📦 Fast dependency management with `uv`
- 🔢 JAX for high-performance numerical computing with GPU support
- 🧠 Equinox for neural network modules
- 📊 Visualization with Altair and treescope
- 🧪 Testing setup with pytest and hypothesis
- 🎯 Code quality with Ruff (linting + formatting)
- 🔍 Type checking with ty
- 👷 CI/CD with GitHub Actions
- 📓 Interactive notebook development with marimo

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) installed

## Getting started

1. Clone this repository:

   ```bash
   git clone https://github.com/eringrant/marimo-uv-jax
   cd marimo-uv-jax
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Run the marimo editor:

   ```bash
   uv run marimo edit
   ```

   Or run a specific experiment:

   ```bash
   uv run marimo edit experiments/example.py
   ```

## Development

### Setup

For development, install all dependency groups (including dev and test):

```bash
uv sync --all-groups
```

Install pre-commit hooks:

```bash
uv run prek install
```

### Testing

```bash
uv run pytest tests
```

### Linting and formatting

```bash
# Run all pre-commit hooks on all files
uv run prek run --all-files

# Or run individual tools:
# Check code style
uv run ruff check .
# Format code
uv run ruff format .
```

### Type checking

```bash
# Run type checking with ty
uv run --group dev ty check
```

### Alternative: Using `just`

A `justfile` is provided for convenience. Install [just](https://github.com/casey/just) and run `just` to see all available commands:

```bash
just setup          # Install deps + hooks
just test           # Run tests
just lint           # Run linting
just typecheck      # Run type checking
just ci             # Run full CI pipeline locally
just marimo         # Open marimo editor
just experiment example  # Run specific experiment
```

## License

MIT
