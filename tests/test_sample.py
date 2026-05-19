"""Tests for marimo_uv_jax.utils module."""

import jax.numpy as jnp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

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


@given(_finite_vector)
def test_mse_loss_self_is_zero(arr):
  """MSE of any vector against itself is zero (property test)."""
  x = jnp.asarray(arr)
  assert jnp.isclose(mse_loss(x, x), 0.0)


@given(_same_shape_pair())
def test_mse_loss_symmetric(pair):
  """MSE is symmetric in its arguments (property test)."""
  a, b = pair
  x, y = jnp.asarray(a), jnp.asarray(b)
  assert jnp.isclose(mse_loss(x, y), mse_loss(y, x))


def test_ensure_scratch_dir_creates_and_returns():
  """ensure_scratch_dir() materializes the dir and returns its path."""
  from marimo_uv_jax import SCRATCH_DIR
  from marimo_uv_jax import ensure_scratch_dir

  path = ensure_scratch_dir()
  assert path == SCRATCH_DIR
  assert path.is_dir()
