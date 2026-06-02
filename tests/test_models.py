"""Tests for marimo_uv_jax.models (Equinox MLP + optax training)."""

import jax
import jax.numpy as jnp
import pytest

from marimo_uv_jax.models import MLP
from marimo_uv_jax.models import batched_mse
from marimo_uv_jax.models import fit


def test_mlp_forward_shape():
  """A single forward pass maps in_size -> out_size."""
  model = MLP(3, 2, width=8, depth=2, key=jax.random.key(0))
  assert model(jnp.ones((3,))).shape == (2,)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_mlp_layer_count(depth):
  """An MLP has depth + 1 Linear layers."""
  model = MLP(3, 2, width=8, depth=depth, key=jax.random.key(0))
  assert len(model.layers) == depth + 1


def test_mlp_rejects_depth_below_1():
  """Depth < 1 is rejected, not silently degraded to a linear model."""
  with pytest.raises(ValueError):
    MLP(2, 1, depth=0, key=jax.random.key(0))


def test_batched_mse_matches_manual_mean():
  """batched_mse equals the mean of per-example squared errors."""
  model = MLP(3, 2, width=8, depth=1, key=jax.random.key(0))
  x = jax.random.normal(jax.random.key(1), (16, 3))
  y = jax.random.normal(jax.random.key(2), (16, 2))
  manual = jnp.mean((jax.vmap(model)(x) - y) ** 2)
  assert jnp.allclose(batched_mse(model, x, y), manual)
  assert float(batched_mse(model, x, jax.vmap(model)(x))) == 0.0


def test_fit_reduces_loss():
  """Training drives the loss down and returns one loss per step."""
  mkey, dkey = jax.random.split(jax.random.key(0))
  x = jax.random.normal(dkey, (64, 2))
  y = jnp.sin(x).sum(axis=1, keepdims=True)
  model = MLP(2, 1, width=16, depth=2, key=mkey)
  trained, losses = fit(model, x, y, steps=200, learning_rate=1e-2)
  assert losses.shape == (200,)
  assert float(losses[-1]) < float(losses[0])
  assert float(batched_mse(trained, x, y)) < float(batched_mse(model, x, y))


def test_fit_loss_history_matches_steps():
  """Fit returns exactly `steps` losses (scan length wired to the argument)."""
  x = jax.random.normal(jax.random.key(0), (16, 2))
  y = jnp.zeros((16, 1))
  model = MLP(2, 1, width=4, depth=1, key=jax.random.key(1))
  _, losses = fit(model, x, y, steps=37)
  assert losses.shape == (37,)
