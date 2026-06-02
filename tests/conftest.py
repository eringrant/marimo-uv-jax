"""Shared pytest / hypothesis configuration."""

from jaxtyping import install_import_hook

# Enforce jaxtyping shape/dtype annotations at runtime (via beartype). Must run
# before the test modules import the package.
install_import_hook("marimo_uv_jax", "beartype.beartype")
