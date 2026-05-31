# `marimo` + `uv` + JAX template

A starter template for [marimo](https://marimo.io) notebooks using
[uv](https://github.com/astral-sh/uv) for dependency management and
[JAX](https://github.com/jax-ml/jax) for numerical computing.

## Getting started

Use this template (GitHub's "Use this template", or clone it), then:

```bash
uv sync                                               # install
uv run marimo edit                                    # open the editor
uv run marimo edit notebooks/experiments/example.py   # or a specific notebook
```

To make it your own, rename the `src/marimo_uv_jax/` directory
and update files returned by:

```bash
git grep -lI 'marimo[-_]uv[-_]jax'
```

## Development

```bash
uv sync --all-groups          # dev + test deps
uv run prek install           # pre-commit hooks
uv run pytest                 # tests
uv run prek run --all-files   # lint + format
uv run --all-groups ty check  # type check
```

A `justfile` wraps these. Install [just](https://github.com/casey/just) and run
it to list recipes:

```bash
just
```
## License

MIT
