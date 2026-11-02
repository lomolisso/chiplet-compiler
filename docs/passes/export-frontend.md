# `cc_export`: JAX ports → StableHLO with components and symbolic sizes

`python -m cc_export --model <key> --config reduced|full --out <dir>` exports every phase of `models/<key>` into
`<dir>` (layout and checks: `docs/passes/export.md`). Nothing executes: only `jax.export` (tracing
and lowering) on `ShapeDtypeStruct`s.

| Module | Role |
|---|---|
| `exporter.py` | per phase: symbolic specs, discovery trace, intercepted export, files, self-checks, concrete census, invariance re-export |
| `composite.py` | `cc_composite` (`lax.composite` + closure conversion), the batching and DCE rules for `composite_p`, `wrap`, static attributes |
| `flax_adapter.py`, `haiku_adapter.py`, `keras_adapter.py` | the three interceptors |
| `symbols.py` | symbolic argument specs, config perturbation, where symbols appear, their sizes at the config |
| `graph.py` | module → `graph.json` through JAX's bundled MLIR bindings |
| `analysis.py` | composite instances, coverage, census, canonical hash, first difference |
| `facts.py` | `python -m cc_export.facts <dir>`: re-parse + MLIR verifier + op multiset (the pass's `verify`) |
| `evaluate.py` | `load(exported.bin)`: an `Exported` callable at concrete (tiny) shapes |

## Interception: every module call is a composite

Every module method call (except the phase root) becomes `stablehlo.composite "cc.<framework>.<Class>[.<method>]"`
whose decomposition is the call; calls inside it nest. Array arguments are operands; every other argument stays
static. Composite attributes are the module's configuration (`composite.attr_value`: deterministic, never a `repr`
with an address) plus `cc_path` (the module path, provenance).

- **Flax linen** (`nn.intercept_methods`): `module.variables` are explicit operands, the decomposition applies an
  unbound clone (`module.clone(parent=None)`; a bound module cannot be used in another trace). The clone is the root
  of its own `apply` (scope path `()`), which is the re-entrancy guard; the phase's own `apply` root passes through
  the same way. Mutable collections (a KV `cache`): the clone runs with the scope's `mutable` filter and the
  variables it changes or creates are extra composite results, written back into the scope.
- **Haiku** (`hk.intercept_methods`): the decomposition calls the intercepted method; parameters read from the
  transform frame (through every custom getter the port installs) are values of the enclosing trace and become
  operands by closure conversion. A discovery trace finds the top-level calls: exactly one is the phase root.
- **Keras 3** (`Layer.__call__` patched during an export): the layer's variables are operands at their current
  values, the decomposition calls the layer under a nested `StatelessScope` mapping them; only calls on arrays.

Three rules `jax.lax.composite` lacks in jax 0.12, registered by `composite.py`:
- **Closure conversion**: a decomposition reading a value of the enclosing trace (a global context such as
  graphcast's static graph, a closure, a Haiku parameter) gets it as a leading operand instead of failing.
- **Batching** (`vmap` of a composite = composite of the vmapped decomposition; alphafold2 maps over modules).
- **Dead-code elimination** into decompositions (without it every component input stays live and dead code inside
  components reaches the module).

JAX's trace caches are cleared around every trace: ports that trace internally (`dce` in alphafold2 / graphcast)
would otherwise reuse a trace made without interception.

## Symbolic shapes and size independence

Each port declares (`models/spec.py`): `ModelSpec.symbols = {name: Sym(config field, alt, doc)}`,
`ModelSpec.constraints` (JAX syntax), and per phase `Phase.shapes` (a pytree prefix of the inputs with JAX shape
specs, `_` = concrete) and optionally `Phase.size_variant_of`. Input-dependent axes (batch, sequence, tokens,
points, voxels, frames, edges, image sizes) are symbols; architecture dims stay concrete. One StableHLO module then
serves every size meeting the constraints, so its components and kernels do not depend on input sizes; Python
code that branches on a symbolic dim fails tracing. Two further certificates close the remaining gaps:
- **Invariance**: the phase is re-exported from `build(cfg')`, every symbol's field set to `alt`; the canonical
  graph hash (provenance excluded) and every input shape must be unchanged. This catches code branching on Python
  config values, which tracing cannot see. Config fields that are symbols appear in attributes as the symbol name.
- **Size variants**: phases declared as differing only in input sizes must give the identical graph.

Where the networks need it, input dims are written as multiples (`32*H`: a feature map upsampled ×2 and
concatenated with a skip only matches for multiples of the maximal stride); constraints bound sizes by
architecture tables (`S <= max_position_embeddings`).

## `graph.json`

Rust types and JSON Schema: `crates/cc-passes/src/export/json.rs`, `docs/schemas/export_graph.schema.json`.
Values are integer ids in definition order; an operand's port is its position. Types and attributes are tagged by
`kind` (`{"kind": "tensor", "shape": [16, null], "elem": "f32"}`, `{"kind": "dot_dimension_numbers", ...}`,
`{"kind": "opaque", "value": "..."}` when no structured form exists). Dynamic dims are `null`; their expressions
follow from the dimension-variable arguments (`jax.global_constant = "<symbol>"`) and the size computation.

## Evaluating an export (tiny sizes only, design §5.3)

`evaluate.load(path).call(*args)`: XLA's shape refinement does not specialize composite decompositions, so for
evaluation every composite is rewritten into a `func.call` of its decomposition (the spec's own semantics of a
composite). `python/tests/test_export.py` uses it to check that interception preserves semantics.

## Tests

`python/tests/` (CI): toy Flax / Haiku / Keras ports (`toys.py`): nesting, roots, attributes, mutable caches,
global-context hoisting, a port-side Haiku getter, vmap and DCE of composites, invariance catching a config branch,
Python shape arithmetic failing, size variants, census = `verify/hlo.py`, the concrete census, symbol sizes (scaled
axes, a failing expression), semantics at tiny sizes.
