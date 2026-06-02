"""Reproduction: linear regression solved three ways, and shown to agree.

Closed-form normal equation vs. an optimistix least-squares solver vs. optax
gradient descent, with float64 enabled so the comparison is exact to tolerance.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
  mo.md(
    """
    # Linear regression, three ways

    The same least-squares fit via (1) the closed-form **normal equation**,
    (2) an **optimistix** least-squares solver, and (3) **optax** gradient
    descent. With `jax_enable_x64` they agree to tolerance, a small
    reproducibility check.
    """
  )
  return


@app.cell
def _():
  import marimo as mo

  return (mo,)


@app.cell
def _():
  import jax

  # Scientific computing often wants float64; JAX defaults to float32.
  jax.config.update("jax_enable_x64", True)

  import altair as alt
  import jax.numpy as jnp
  import optax
  import optimistix as optx
  import polars as pl

  return alt, jax, jnp, optax, optx, pl


@app.cell
def _(jax, jnp):
  # Synthetic linear data: y = X @ w_true + b_true + noise.
  _key = jax.random.key(0)
  x_key, noise_key = jax.random.split(_key)
  n, d = 200, 3
  w_true = jnp.array([2.0, -1.0, 0.5])
  b_true = 0.7
  X = jax.random.normal(x_key, (n, d))
  y = X @ w_true + b_true + 0.05 * jax.random.normal(noise_key, (n,))
  return X, b_true, d, w_true, y


@app.cell
def _(X, jnp, y):
  # (1) Closed form: augment X with a bias column and solve least squares.
  # lstsq is better-conditioned than forming the normal equations (X.T @ X).
  _xb = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
  _theta = jnp.linalg.lstsq(_xb, y)[0]
  w_closed, b_closed = _theta[:-1], _theta[-1]
  return b_closed, w_closed


@app.cell
def _(X, d, jnp, optx, y):
  # (2) optimistix: minimize the residual vector with Levenberg-Marquardt.
  def _residuals(params, args):
    w, b = params
    feats, targets = args
    return feats @ w + b - targets

  _solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-10)
  _sol = optx.least_squares(
    _residuals, _solver, (jnp.zeros(d), jnp.array(0.0)), args=(X, y)
  )
  w_optx, b_optx = _sol.value
  return b_optx, w_optx


@app.cell
def _(X, d, jax, jnp, optax, y):
  # (3) optax gradient descent on the mean squared error.
  def _mse(params, feats, targets):
    w, b = params
    return optax.squared_error(feats @ w + b, targets).mean()

  _optim = optax.adam(1e-1)
  _params = (jnp.zeros(d), jnp.array(0.0))
  _opt_state = _optim.init(_params)

  @jax.jit
  def _step(params, opt_state):
    loss, grads = jax.value_and_grad(_mse)(params, X, y)
    updates, opt_state = _optim.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

  for _ in range(2000):
    _params, _opt_state, _ = _step(_params, _opt_state)
  w_gd, b_gd = _params
  return b_gd, w_gd


@app.cell
def _(
  alt,
  b_closed,
  b_gd,
  b_optx,
  b_true,
  jnp,
  pl,
  w_closed,
  w_gd,
  w_optx,
  w_true,
):
  # All three should recover the true coefficients to tolerance.
  _methods = {
    "true": (w_true, b_true),
    "closed-form": (w_closed, b_closed),
    "optimistix": (w_optx, b_optx),
    "optax-gd": (w_gd, b_gd),
  }
  for _name, (_w, _b) in _methods.items():
    if _name != "true":
      assert jnp.allclose(_w, w_true, atol=1e-2), _name
      assert jnp.allclose(_b, b_true, atol=1e-2), _name

  _rows = [
    {"method": _name, "coef": f"w[{_i}]", "value": float(_w[_i])}
    for _name, (_w, _b) in _methods.items()
    for _i in range(_w.shape[0])
  ]
  _df = pl.DataFrame(_rows)
  alt.Chart(_df).mark_bar().encode(
    x="method:N", y="value:Q", color="method:N", column="coef:N"
  ).properties(height=220, title="recovered coefficients agree")
  return


if __name__ == "__main__":
  app.run()
