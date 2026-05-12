"""Utility functions for JAX experiments."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def normalize(
  x: Float[Array, "... features"],
) -> Float[Array, "... features"]:
  """Normalize input to zero mean and unit variance.

  Args:
    x: Input array to normalize.

  Returns:
    Normalized array with zero mean and unit variance.
  """
  mean = jnp.mean(x, axis=-1, keepdims=True)
  std = jnp.std(x, axis=-1, keepdims=True)
  return (x - mean) / (std + 1e-8)


def mse_loss(
  predictions: Float[Array, " batch"],
  targets: Float[Array, " batch"],
) -> Float[Array, ""]:
  """Compute mean squared error loss.

  Args:
    predictions: Model predictions.
    targets: Target values.

  Returns:
    Mean squared error.
  """
  return jnp.mean((predictions - targets) ** 2)
