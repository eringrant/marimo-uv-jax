"""Interactive JAX showcase: train an Equinox MLP with optax, reactively."""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
  mo.md(
    """
    # JAX showcase: an Equinox MLP trained with optax

    Move the slider to retrain. This exercises the template's stack end to end:
    an `equinox` model, `optax` optimization, and `jax` transforms
    (`filter_jit` / `value_and_grad` / `vmap`), all reactive in marimo.
    """
  )
  return


@app.cell
def _():
  import marimo as mo

  return (mo,)


@app.cell
def _():
  import altair as alt
  import jax
  import jax.numpy as jnp
  import polars as pl
  import treescope

  from marimo_uv_jax.models import MLP
  from marimo_uv_jax.models import fit

  return MLP, alt, fit, jax, jnp, pl, treescope


@app.cell
def _(jax, jnp):
  # A 1-D regression target: y = sin(2x) + noise.
  _key = jax.random.key(0)
  data_key, model_key = jax.random.split(_key)
  x = jnp.linspace(-3.0, 3.0, 128).reshape(-1, 1)
  y = jnp.sin(2.0 * x) + 0.1 * jax.random.normal(data_key, x.shape)
  return model_key, x, y


@app.cell
def _(mo):
  steps = mo.ui.slider(50, 600, value=300, step=50, label="training steps")
  steps
  return (steps,)


@app.cell
def _(MLP, fit, model_key, steps, x, y):
  _model = MLP(1, 1, width=32, depth=2, key=model_key)
  trained, losses = fit(_model, x, y, steps=steps.value, learning_rate=1e-2)
  return losses, trained


@app.cell
def _(alt, jax, pl, trained, x, y):
  # Predictions vs targets (vmap the single-example model over the batch).
  _preds = jax.vmap(trained)(x)
  _df = pl.DataFrame(
    {
      "x": x.ravel().tolist(),
      "target": y.ravel().tolist(),
      "prediction": _preds.ravel().tolist(),
    }
  )
  _long = _df.unpivot(index="x", variable_name="series", value_name="value")
  alt.Chart(_long).mark_line().encode(
    x="x:Q", y="value:Q", color="series:N"
  ).properties(height=260, title="fit")
  return


@app.cell
def _(alt, losses, pl):
  # Training loss curve (log scale).
  _df = pl.DataFrame({"step": range(losses.shape[0]), "loss": losses.tolist()})
  alt.Chart(_df).mark_line().encode(
    x="step:Q", y=alt.Y("loss:Q", scale=alt.Scale(type="log"))
  ).properties(height=200, title="training loss")
  return


@app.cell
def _(mo, trained, treescope):
  # Inspect the trained model as a JAX pytree.
  mo.Html(treescope.render_to_html(trained))
  return


if __name__ == "__main__":
  app.run()
