"""JAX demonstration notebook with marimo."""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
  mo.md("""
    # JAX + marimo + uv Template

    This template demonstrates using JAX with marimo notebooks and uv for
    dependency management.
    """)


@app.cell
def _():
  import marimo as mo

  return (mo,)


@app.cell
def _():
  import jax
  import jax.numpy as jnp

  from marimo_uv_jax.utils import mse_loss
  from marimo_uv_jax.utils import normalize

  return jax, jnp, mse_loss, normalize


@app.cell
def _(jax):
  # Create a simple dataset
  key = jax.random.PRNGKey(42)
  return (key,)


@app.cell
def _(jax, jnp, key):
  # Generate random data
  x = jax.random.normal(key, (100, 10))
  y = jnp.sum(x, axis=1) + jax.random.normal(key, (100,)) * 0.1
  return x, y


@app.cell
def _(mo, normalize, x):
  # Normalize the data
  x_normalized = normalize(x)
  mo.md(f"Original shape: {x.shape}, Normalized shape: {x_normalized.shape}")
  return (x_normalized,)


@app.cell
def _(jnp, mse_loss, y):
  # Compute a simple loss
  predictions = jnp.zeros_like(y)
  loss = mse_loss(predictions, y)
  loss
  return loss, predictions


if __name__ == "__main__":
  app.run()
