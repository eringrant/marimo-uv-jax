"""Utility functions for JAX experiments."""

import jax.numpy as jnp
import optax
from jaxtyping import Array
from jaxtyping import Float


def normalize(
  x: Float[Array, "*batch features"],
) -> Float[Array, "*batch features"]:
  """Standardize each feature to zero mean and unit variance over the batch.

  Requires at least a batch and a feature axis; a rank-1 array has no batch to
  reduce over (statistics would be per-element, zeroing the output).
  """
  if x.ndim < 2:
    raise ValueError(
      f"normalize expects >= 2 dims (batch, ..., features), got {x.ndim}"
    )
  batch_axes = tuple(range(x.ndim - 1))
  mean = jnp.mean(x, axis=batch_axes, keepdims=True)
  std = jnp.std(x, axis=batch_axes, keepdims=True)
  return (x - mean) / (std + 1e-8)


def mse_loss(
  predictions: Float[Array, "*shape"],
  targets: Float[Array, "*shape"],
) -> Float[Array, ""]:
  """Mean squared error between two equally-shaped arrays."""
  return jnp.mean(optax.squared_error(predictions, targets))
