"""A small Equinox MLP and an optax training loop: the JAX showcase.

Exercises the template's core stack end to end: an ``equinox`` model (which is
just a JAX pytree), ``optax`` for optimization, and the JAX/Equinox transforms
(``filter_jit``, ``filter_value_and_grad``, ``vmap``) that make a compiled,
differentiated, batched training step.
"""

from __future__ import annotations

import equinox as eqx
import jax
import optax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PRNGKeyArray

from marimo_uv_jax.utils import mse_loss


class MLP(eqx.Module):
  """A multilayer perceptron with tanh activations.

  An ``eqx.Module`` is a registered pytree: its array fields are leaves (so JAX
  transforms see through it) while everything else is static. ``__call__`` acts
  on a SINGLE example; batch it with ``jax.vmap``.
  """

  layers: list[eqx.nn.Linear]

  def __init__(
    self,
    in_size: int,
    out_size: int,
    *,
    width: int = 32,
    depth: int = 2,
    key: PRNGKeyArray,
  ):
    """Build the layer stack.

    Args:
      in_size: Input feature count.
      out_size: Output feature count.
      width: Hidden layer width.
      depth: Number of hidden layers (>= 1).
      key: PRNG key for weight initialization.
    """
    if depth < 1:
      raise ValueError(f"depth must be >= 1, got {depth}")
    sizes = [in_size, *([width] * depth), out_size]
    keys = jax.random.split(key, len(sizes) - 1)
    self.layers = [
      eqx.nn.Linear(i, o, key=k) for i, o, k in zip(sizes[:-1], sizes[1:], keys)
    ]

  def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
    """Apply the MLP to one example."""
    for layer in self.layers[:-1]:
      x = jax.nn.tanh(layer(x))
    return self.layers[-1](x)


def batched_mse(
  model: MLP,
  x: Float[Array, "batch in_size"],
  y: Float[Array, "batch out_size"],
) -> Float[Array, ""]:
  """Mean squared error of ``model`` over a batch (vmapped over examples)."""
  predictions = jax.vmap(model)(x)
  return mse_loss(predictions, y)


def fit(
  model: MLP,
  x: Float[Array, "batch in_size"],
  y: Float[Array, "batch out_size"],
  *,
  steps: int = 200,
  learning_rate: float = 1e-2,
) -> tuple[MLP, Float[Array, " steps"]]:
  """Train ``model`` on ``(x, y)`` with Adam via ``jax.lax.scan``.

  Returns the trained model and a 1-D array of per-step training losses.
  """
  optim = optax.adam(learning_rate)
  opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

  @eqx.filter_jit
  def run(
    model: MLP, opt_state: optax.OptState
  ) -> tuple[tuple[MLP, optax.OptState], Float[Array, " steps"]]:
    def step(carry, _):
      model, opt_state = carry
      loss, grads = eqx.filter_value_and_grad(batched_mse)(model, x, y)
      updates, opt_state = optim.update(grads, opt_state)
      return (eqx.apply_updates(model, updates), opt_state), loss

    return jax.lax.scan(step, (model, opt_state), xs=None, length=steps)

  (model, _), losses = run(model, opt_state)
  return model, losses
