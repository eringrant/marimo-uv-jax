"""Tests for marimo_uv_jax.utils module."""

from marimo_uv_jax.utils import mse_loss
from marimo_uv_jax.utils import normalize


def test_normalize():
  """Test normalize function."""
  import jax.numpy as jnp

  x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  normalized = normalize(x)
  # Check that mean is approximately 0 and std is approximately 1
  assert jnp.abs(jnp.mean(normalized, axis=-1)).max() < 1e-6
  assert jnp.allclose(jnp.std(normalized, axis=-1), 1.0, atol=1e-6)


def test_mse_loss():
  """Test MSE loss function."""
  import jax.numpy as jnp

  predictions = jnp.array([1.0, 2.0, 3.0])
  targets = jnp.array([1.0, 2.0, 3.0])
  loss = mse_loss(predictions, targets)
  assert jnp.isclose(loss, 0.0)

  predictions = jnp.array([1.0, 2.0, 3.0])
  targets = jnp.array([2.0, 3.0, 4.0])
  loss = mse_loss(predictions, targets)
  assert jnp.isclose(loss, 1.0)
