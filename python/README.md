# python/

Python is used only where a Python-only tool must run.

| Package | Role |
|---|---|
| `cc_export/` | another change | JAX export front end: composites, symbolic shapes, StableHLO JSON for Rust |
| `cc_softfloat/` | another change | Allo soft-float library (float formats as integer bit operations) |
| `cc_artifacts/` | another change | read-only helpers for notebooks over `build/.<pass>/` |
| `tests/` | pytest suites, run in CI |

The project's `pyproject.toml` / `uv.lock` (repo root) pin the JAX stack used by `models/`.
