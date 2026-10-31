# JAX export → StableHLO (the only Python step)

`python/cc_export/` turns each model in `models/` into StableHLO plus a JSON dump the Rust passes read.
Pinned stack (repo `pyproject.toml`): `jax==0.12.0`, `jaxlib==0.12.0`, `flax==0.12.10`, `numpy==2.5.4`,
Python 3.12; models use `flax.linen` (some vendored upstream code uses Haiku). Quantized GEMMs/convs are written in
explicit QDQ form (`models/quant.py`: i8×i8→i32, f8E4M3FN×f8E4M3FN→f32, int4/int8 weight-only dequant), so the
graph carries plain integer/FP8 dtypes, not `!quant.uniform` types (see `stablehlo.md` §8).
All source references below are to tag `jax-v0.12.0` of https://github.com/jax-ml/jax.

## 1. Memory rules (non-negotiable)

- Export = **tracing + lowering with abstract values**; nothing executes. Feed `jax.ShapeDtypeStruct`s (or
  symbolic shapes) for both inputs and parameters.
- Get parameter shapes with `jax.eval_shape(model.init, rng, x_spec)` returns a pytree of `ShapeDtypeStruct`
  without allocating (https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html). **Never call
  `model.init`/`apply` at full size.**
- Parameters must be function **arguments**, not closed-over concrete arrays: closed-over numpy/jax arrays are
  embedded as dense constants in the module (a multi-GB `.mlir`).
- Numeric checks only at `testing.tiny_dim` sizes (design §8).

## 2. `jax.export` API (`jax/_src/export/_export.py`)

```python
from jax import export
exp = export.export(jax.jit(fwd), platforms=("cpu",))(params_spec, x_spec)   # fwd(params, x)
text = exp.mlir_module()          # StableHLO text (deserialized from the portable artifact)
blob = exp.serialize()            # bytes; deserialize() requires trusted input
```

| `Exported` field | Use |
|---|---|
| `fun_name`, `in_tree`, `out_tree` | map flat args back to the params/x pytree (record key paths in our JSON) |
| `in_avals`, `out_avals` | abstract shapes/dtypes (symbolic dims appear as expressions, e.g. `int32[a,2*b]`) |
| `module_kept_var_idx` | **sorted indices of the args actually kept in `main`**: unused args are dropped; use it to map `main`'s operands to `in_avals` |
| `platforms`, `nr_devices`, `calling_convention_version`, `uses_global_constants` | record in `verify.json` |
| `disabled_safety_checks` | `export(..., disabled_checks=[DisabledSafetyCheck.custom_call(...)])` only if needed |
| `mlir_module_serialized` | VHLO portable artifact bytes; `mlir_module()` returns text, `mlir_module(serialized=False)` a Module |

Facts: `export` accepts only `jax.jit` results; default platform is the local backend; pass `platforms=("cpu",)`
for determinism. Multiple platforms add a platform-index argument to `main` (docs "Module calling convention").
`main` takes [platform index] + effect tokens + kept array args; inner functions such as
`_wrapped_jax_export_main` take dimension-variable arguments (scalar i32/i64, sorted by name); since calling
convention v9 those carry `jax.global_constant`; default convention version 10 since JAX 0.7.0
(https://docs.jax.dev/en/latest/export/export.html).

## 3. Symbolic shapes (https://docs.jax.dev/en/latest/export/shape_poly.html)

```python
B, S = export.symbolic_shape("B, S", constraints=("S >= 16", "S <= 4096"))
x_spec = jax.ShapeDtypeStruct((B, S, 768), jnp.bfloat16)
```
- `symbolic_shape(spec, constraints=..., scope=...)`; all specs of one export must share a `SymbolicScope`
  (pass `scope=` or create with the same call); mixing scopes raises `ValueError`.
- `symbolic_args_specs(args, spec)` builds specs from concrete args (`...` = any number of dims, `_` = one).
- Dim vars are assumed ≥ 1; `b >= 2` is inconclusive → `InconclusiveDimensionOperation`. Fix with explicit
  `constraints` (`>=`, `<=`, `==`), `core.max_dim`/`core.min_dim`, or code changes. Equality constraints act as
  rewrite rules (no top-level `+`/`-` on the LHS). Non-linear (`a*a`) dims are not solvable.
- Dim vars must be derivable from input shapes, else `UnexpectedDimVar`.
- `symbolic_dim_bounds(expr)` gives conservative bounds; use them for iteration-space analysis.
- Debug: `JAX_DUMP_IR_TO=<dir>` dumps the module before shape refinement.

What the lowering emits when shapes are symbolic (grep of `jax/_src/interpreters/mlir.py`, `lax/*.py`,
`export/shape_poly.py`):
- tensor types with `?` dims;
- `stablehlo.get_dimension_size` (+ `convert`) to compute dim vars from argument shapes (`dimension_size_p`);
- shape-dynamic ops: `dynamic_broadcast_in_dim`, `dynamic_reshape`, `dynamic_iota`, `dynamic_pad`,
  `real_dynamic_slice`, `dynamic_reduce_window`, `dynamic_gather`, `dynamic_conv`, `dynamic_top_k`,
  `dynamic_rng_bit_generator`;
- `stablehlo.custom_call @shape_assertion(%pred, …) {error_message = "…{0}…"}` checks (positivity, divisibility,
  constraint consistency): side-effecting, no results; our kernel analysis must skip them but keep them as
  constraints on dim vars.
- `jax.uses_shape_polymorphism` module attribute (calling convention ≥ 8).
Note: `dynamic_slice`/`dynamic_update_slice` are **static-shape** ops with runtime offsets; they appear with or
without symbolic shapes.

## 4. `jax.lax.composite` (`jax/_src/lax/lax.py` L1924–2106)

`lax.composite(decomposition, name, version=0)` returns a function: **positional args = operands, keyword args =
attributes**. Lowers to `stablehlo.composite "name" %ops {composite_attributes = {...}, decomposition =
@<private func>, version = N}`; the decomposition becomes a private `func.func`.

Rules taken from the source:
- Attributes are static: a `Tracer` in kwargs raises `UnexpectedTracerError` ("use numpy arrays instead").
  Allowed values (`mlir.ir_attribute` handlers): Python `bool/int/float/str/bytes`, numpy scalars and arrays
  (→ dense elements), `np.dtype` (→ TypeAttr), `dict` (→ DictAttr), `list`/`tuple` (→ ArrayAttr), MLIR
  attributes. Top-level `None` kwargs are omitted.
- The decomposition is traced eagerly at call time with `pe.trace_to_jaxpr`; **closing over a tracer** (e.g. Flax
  params captured from the outer `jit`) raises `UnexpectedTracerError`; pass them as operands. Concrete
  closed-over constants are hoisted to extra leading operands.
- Dimension-variable values are prepended to the composite operands when shapes are symbolic.
- No JVP/transpose rules (inference only, fine for us).
- Names: use a namespace, e.g. `cc.flax.Dense`, `cc.quant.qdq_dot` (convention; the API does not enforce one).
  Bump `version` when the decomposition semantics change.

### Wrapping Flax (linen) modules as composites (our design)

Use `flax.linen.intercept_methods(interceptor)` ("intercept calls to modules at a distance"; interceptor called
with `(next_fun, args, kwargs, context)`) so model code is untouched:
1. In the interceptor, for `context.method_name == "__call__"` and module types we mark, take
   `unbound, variables = context.module.unbind()` ("unbound copy of a Module and its variables").
2. Call `lax.composite(lambda v, *xs: unbound.apply(v, *xs), name=f"cc.flax.{type(m).__name__}")(variables, *args, **static_attrs)`:
   variables are explicit operands, hyperparameters (features, strides, dtype names) are static attributes.
3. **Re-entrancy guard**: `unbound.apply` re-enters the interceptor for the same module; keep a `contextvars`
   set of "currently decomposing" module paths and call `next_fun` directly when re-entered, so children still get
   their own nested composites but the module itself is not wrapped twice (infinite recursion otherwise).
4. Only inference collections (`params`, plus read-only `batch_stats` etc.); `deterministic=True`; no RNGs.

## 5. Reading the module with JAX's bundled MLIR bindings

```python
from jax._src.interpreters import mlir as jmlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo        # stablehlo bindings (dialects/__init__.py)

with jmlir.make_ir_context() as ctx:             # registers stablehlo, chlo, mhlo, sdy, func, …
    m = ir.Module.parse(exp.mlir_module())
    def visit(op):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    rec = {"name": o.operation.name,
                           "operands": [str(v.get_name()) for v in o.operands],   # SSA ids
                           "results": [str(r.type) for r in o.results],
                           "attrs": {a.name: str(a.attr) for a in o.attributes},
                           "loc": str(o.location)}
                    ...; visit(o)
    visit(m.operation)
```
- Verified with jaxlib 0.12: `Operation.regions/blocks/operations` walk as sketched; `op.attributes` is
  indexed (`attrs[i].name/.attr`, iterating yields names); `ir.Value` is hashable (use it as a dict key instead of
  `get_name()`, which prints the parent op each call); attributes come back down-cast (`IntegerAttr`,
  `DenseI64ArrayAttr`, ...). Implementation: `python/cc_export/graph.py`.
- Types: `ir.RankedTensorType(t)` → `.shape`, `.element_type`; `ir.ShapedType.is_dynamic_size(d)` for `?`.
- StableHLO attribute wrappers (e.g. `hlo.DotDimensionNumbers(attr).lhs_batching_dimensions`,
  `hlo.ConvDimensionNumbers`, `hlo.GatherDimensionNumbers`) give structured access; fall back to `str(attr)` and
  parse in Rust only for opaque attributes. Verified: `DotDimensionNumbers.lhs_batching_dimensions` etc.;
  enum attributes (`ComparisonDirectionAttr`, ...) are read from their printed form.
- `make_ir_context()` reuses a thread-local context ≤10 times; create one context per model export.
- Emit one JSON per model: functions, ops (topologically ordered as in the block), operand SSA ids, types,
  attributes, composite name/attrs/decomposition symbol, locations, `module_kept_var_idx` and pytree paths.
- Keep the `.mlir` text too (for `stablehlo-opt`-based checks and human diffing).

## 6. Pitfalls

| Pitfall | Mitigation |
|---|---|
| Params captured by closure → giant constants / tracer errors | params as explicit args; composites take `variables` as operands |
| Dropped unused args | map via `module_kept_var_idx` |
| Platform-specific lowering | `platforms=("cpu",)` always |
| `eval_shape`/`init` RNG differences | only shapes matter; never materialize |
| Haiku models | `hk.transform` → `apply(params, rng, x)`; same rules. `hk.intercept_methods` works; it also intercepts `hk.Module`'s own methods (`params_dict`), skip those |
| Non-deterministic attr printing | sort dict keys in JSON |

## Sources

- https://docs.jax.dev/en/latest/export/export.html ; https://docs.jax.dev/en/latest/export/shape_poly.html ;
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.composite.html ; https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html
- JAX source @ `jax-v0.12.0`: `jax/_src/export/_export.py` (`Exported`, `export`, `mlir_module`,
  `_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE` incl. `shape_assertion`), `jax/_src/export/shape_poly.py`
  (`dimension_size_p`, `shape_assertion_p`), `jax/_src/lax/lax.py` (`composite`), `jax/_src/interpreters/mlir.py`
  (`ir_attribute` handlers, `make_ir_context`), `jax/_src/lib/mlir/dialects/__init__.py`.
- Flax linen Module API (`unbind`, `variables`, `intercept_methods`): https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html
- Repo: `pyproject.toml`, `models/quant.py`, `python/README.md`.

## Findings (jax 0.12.0)

- `lax.composite` refuses decompositions that close over tracers, has no batching rule and no DCE rule; the
  exporter registers all three (`python/cc_export/composite.py`). With symbolic shapes, `Exported.call` fails in
  XLA's shape refinement on composites ("Error refining shapes"): rewrite composites into `func.call`s first
  (`cc_export/evaluate.py`); `stablehlo-legalize-composite-to-call` is not registered in jaxlib's bindings.
- `jax.make_jaxpr` / `jit` trace caches survive across exports: a port that traces internally gets a stale jaxpr
  unless `jax.clear_caches()` runs between traces with different interception.
- `jax.export.Exported.serialize` needs the `flatbuffers` package (a dependency in `pyproject.toml`).
- Symbolic dims: a feature map upsampled ×2 and concatenated with its skip only type-checks when the input dim is
  written as a multiple of the stride (`32*H`), not with a bare symbol plus a `mod` constraint.

