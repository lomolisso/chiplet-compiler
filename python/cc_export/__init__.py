"""cc_export: JAX ports → StableHLO with component boundaries (composites) and symbolic input sizes.

The only Python pass of chiplet-compiler See docs/passes/export-frontend.md.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
