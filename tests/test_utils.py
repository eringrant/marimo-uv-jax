"""Tests for marimo_uv_jax.utils module."""

from datetime import timedelta

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from marimo_uv_jax.utils import mse_loss
from marimo_uv_jax.utils import normalize

# Generous enough to absorb JAX's per-shape compile on these tiny arrays while
# still tripping on an orders-of-magnitude slowdown.
_JAX_DEADLINE = timedelta(seconds=2)


def test_normalize():
  """Standardize each feature (column) over the batch axis."""
  x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
  normalized = normalize(x)
  # Each feature (column) should be zero-mean / unit-variance over the batch.
  assert jnp.abs(jnp.mean(normalized, axis=0)).max() < 1e-5
  assert jnp.allclose(jnp.std(normalized, axis=0), 1.0, atol=1e-4)


def test_normalize_single_feature_column():
  """A (batch, 1) column normalizes over the batch, not to all zeros."""
  x = jnp.linspace(-3.0, 3.0, 128).reshape(-1, 1)
  normalized = normalize(x)
  assert jnp.abs(jnp.mean(normalized)) < 1e-5
  assert jnp.isclose(jnp.std(normalized), 1.0, atol=1e-4)


def test_normalize_reduces_over_all_but_last_axis():
  """With >2 dims, stats are taken over every axis except the feature axis."""
  x = jax.random.normal(jax.random.key(0), (4, 5, 3))
  out = normalize(x)
  assert jnp.abs(jnp.mean(out, axis=(0, 1))).max() < 1e-5
  assert jnp.allclose(jnp.std(out, axis=(0, 1)), 1.0, atol=1e-4)


def test_normalize_constant_feature_is_finite():
  """A zero-variance feature stays finite (the 1e-8 floor), not nan/inf."""
  x = jnp.array([[5.0, 1.0], [5.0, 3.0], [5.0, 5.0]])  # column 0 is constant
  out = normalize(x)
  assert jnp.all(jnp.isfinite(out))
  assert jnp.allclose(out[:, 0], 0.0)


def test_normalize_rejects_rank1():
  """A rank-1 array has no batch axis and is rejected, not silently zeroed."""
  with pytest.raises(ValueError):
    normalize(jnp.array([1.0, 2.0, 3.0]))


def test_mse_loss():
  """Test MSE loss function."""
  predictions = jnp.array([1.0, 2.0, 3.0])
  targets = jnp.array([1.0, 2.0, 3.0])
  loss = mse_loss(predictions, targets)
  assert jnp.isclose(loss, 0.0)

  predictions = jnp.array([1.0, 2.0, 3.0])
  targets = jnp.array([2.0, 3.0, 4.0])
  loss = mse_loss(predictions, targets)
  assert jnp.isclose(loss, 1.0)


_elements = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, width=32)
_finite_vector = hnp.arrays(
  dtype=jnp.float32,
  shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=64),
  elements=_elements,
)


@st.composite
def _same_shape_pair(draw):
  """Draw two float32 vectors that share a shape."""
  shape = draw(hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=64))
  a = draw(hnp.arrays(dtype=jnp.float32, shape=shape, elements=_elements))
  b = draw(hnp.arrays(dtype=jnp.float32, shape=shape, elements=_elements))
  return a, b


@settings(deadline=_JAX_DEADLINE)
@given(_finite_vector)
def test_mse_loss_self_is_zero(arr):
  """MSE of any vector against itself is zero (property test)."""
  x = jnp.asarray(arr)
  assert jnp.isclose(mse_loss(x, x), 0.0)


@settings(deadline=_JAX_DEADLINE)
@given(_same_shape_pair())
def test_mse_loss_scales_quadratically(pair):
  """Scaling both inputs by k scales MSE by k**2."""
  a, b = pair
  x, y = jnp.asarray(a), jnp.asarray(b)
  base = mse_loss(x, y)
  assert jnp.allclose(mse_loss(2.0 * x, 2.0 * y), 4.0 * base, rtol=1e-4)
