"""cc_export: JAX ports → StableHLO with component boundaries (composites) and symbolic input sizes.

The only Python pass of chiplet-compiler. See README.md.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
