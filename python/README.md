# python/

`cc_export/` is the JAX export front end of the `export` pass ([docs/passes/export-frontend.md](../docs/passes/export-frontend.md));
`cc_float/` models float compositions of cores and measures their error ([docs/framework/float.md](../docs/framework/float.md));
`tests/` holds the pytest suites. The locked JAX stack is the repo root's `pyproject.toml` / `uv.lock`.
