# `cc-ir`: the IR and the op-semantics table

The IR the analysis passes share (written by `import`, docs/passes/import.md), and the op-semantics
table that encodes the StableHLO specification (https://openxla.org/stablehlo/spec) as data (design §4.2).
Nothing in this crate maps ops or models to hardware.

## Module (`ir.rs`)

An arena: plain vectors addressed by typed `u32` indices (`FuncId`, `RegionId`, `OpId`, `ValueId`), so a
module serializes with serde (postcard on disk) and is deterministic. Graph algorithms build their views
(users, the `absgraph` graphs below) on demand.

| Type | Fields |
|---|---|
| `Module` | `model`, `phase`, `config`; `symbols` (sorted) and `constraints` of the symbolic scope; `entry`; the arenas `funcs`, `regions`, `ops`, `values`; `locs` (interned source locations) |
| `Func` | `name`, `kind` (`entry`, `decomposition` of a component composite, `computation` of a custom call), `body`, `results` (types) |
| `Region` | `owner` (a function or an op), `args`, `ops` in program order (one block; the terminator last) |
| `Op` | `name`, `target` (custom-call target), `operands` (the port is the position), `results`, typed `attrs`, `regions`, `callee`, `parent` region, `size` (size computation), `dim_as_data` (ports where a dimension enters compute), `prov` |
| `Value` | `ty`, `def` (`arg {region, index}` or `result {op, index}`), `dim_var` (the symbol of a dimension-variable argument), `src` (its value id in `graph.json`) |
| `Provenance` | `graph.json` function and op id, interned location, `inlined_from` (the inlined call sites, outermost first) |

Op ids order every definition before its uses. `func.call` never occurs: calls are inlined. A
`stablehlo.composite` refers to its decomposition through `callee`. `Module::expanded_multiset` counts the
ops of the entry with every composite expanded at its call site.

`hash::module_hash` is a structural hash (Merkle style): a value's hash covers how it is computed (op
name, target, attributes, result types, operand hashes in port order, regions, callee), a region's hash
the multiset of its ops' hashes. Op order, value numbering and provenance do not enter it.

## Types (`types.rs`)

- `Elem`: `bool` (`i1`); `int {signed, width}` (signless `iN` is signed in StableHLO, `uiN` unsigned);
  `float {exp, mant, flavour}` with the special-value convention (`ieee`; `finite_nan` = `FN`, e.g. OCP OFP8
  E4M3; `finite_nan_unsigned_zero` = `FNUZ`; …); `complex(elem)`. `Elem::parse` and `Display` use the MLIR
  spellings (`i8`, `ui32`, `bf16`, `f8E4M3FN`, `complex<f32>`).
- `Dim`: `const(u64)` or `sym(SymExpr)`; a `Sym` is never constant.
- `TensorType {shape: Vec<Dim>, elem}`; `Type` is a tensor or, for anything else, its printed form.

## Symbolic expressions (`sym.rs`)

JAX's dimension algebra (`jax/_src/export/shape_poly.py`, jax 0.12.0): a `SymExpr` is a sum of terms
`k * m` with `k: i64` and `m` a monomial, a product of atoms with powers; an atom is a variable or
`floordiv(e1, e2)`, `mod(e1, e2)`, `max(e1, e2)`, `min(e1, e2)` (Python semantics: `floordiv` rounds
toward −∞, `mod` takes the divisor's sign). The form is canonical (terms sorted, no zero coefficients),
so equal normal forms mean equal expressions; the converse is decided, where it is needed, by `SymCtx`.

- Construction folds constants and applies exact identities only: for a constant divisor `c > 0`,
  `floor((c*q + r) / c) = q + floor(r / c)` (coefficients split by Euclidean division),
  `floor((floor(x / a) + k) / c) = floor((x + k*a) / (a*c))`, and `mod` by the same split.
- `parse` reads JAX's syntax (integers, identifiers, `+ - *`, `//`, `%`, parentheses, the four functions);
  `Display` prints it back. `Constraint::parse` reads `e1 >= e2`, `e1 <= e2`, `e1 == e2`.
- `eval` evaluates at an assignment with checked arithmetic.
- `SymCtx` holds the bounds of the scope (every symbol at least 1, refined by single-variable constraints)
  and the constraints as facts `f >= 0`. `nonneg(e)` proves `e >= 0` or `e < 0` by interval arithmetic,
  directly or with one fact subtracted; `ge`, `eq`, `max`, `min` build on it; `with(facts)` adds facts
  (the guards of a case split).

## Attributes (`attrs.rs`)

`Attr` mirrors the attribute kinds of the exported modules: integers, floats, booleans, strings, symbols,
integer and boolean lists, dense elements (values when the export kept them, at most 1024, else a
digest), arrays, dictionaries, types, unit, StableHLO enums, and the structured dimension numbers
(`DotDims`, `ConvDims`, `GatherDims`, `ScatterDims`). `AttrsExt` reads typed values: `int`, `ints` (an
integer list or a dense integer tensor), `enum_value`, `string`, `padding` (`[[lo, hi], …]`).

## The op-semantics table (`ops/`)

`ops::TABLE` is a static list of `OpSpec`, one per op the models use plus the spec's dynamic ops; custom
calls are keyed by target. `ops::lookup(name, target)` returns the entry or `None`, and a caller treats
`None` as an error that names the op. `op_inventory.json` serializes the entries as data.

| Field | Meaning |
|---|---|
| `name`, `target` | `stablehlo.add`; `stablehlo.custom_call` with `target = "shape_assertion"` |
| `spec` | the section anchor in the StableHLO spec (`#add`) |
| `class` | `core` (dot_general, convolution, reduce, reduce_window and their dynamic forms), `elementwise`, `layout` (reshape, transpose, broadcast, slice, concatenate, pad, reverse), `data_movement` (gather, scatter, dynamic slices), `size` (`get_dimension_size`, shape assertions), `control` (returns, calls, composites), `irregular` (sort, fft), `constant` (constant, iota) |
| `operands`, `results` | arity: `exactly(n)` or `at_least(n)` |
| `elementwise`, `commutative` | spec properties (commutative only when it holds for every admitted element type) |
| `regions` | number of regions |
| `shape_operands` | ports that carry shapes: `ports([…])`, `last(n)`, `all`, or `dim_var_args` (composite ports bound to dimension arguments) |
| `elem` | result element type: `same_as(i)`, `bool`, `i32`, `component_of0` (`real`/`imag`), `complex_of0`, `free` (`convert`, `dot_general`'s preferred type, composites) |
| `shape` | the result-shape rule (`ops/shape.rs`), transcribed from the spec's shape constraints and cited per rule; symbolic over `Dim`, with shape operands given as evaluated `SymExpr` vectors |
| `iteration` | the iteration-space semantics (`ops/iter.rs`, below): set for every core, element-wise op, broadcast and `scatter` |
| `dims` | the dimension and element-type equalities between operands and results (`ops/dims.rs`, below) |
| `size_attrs` | attributes whose integers are sizes, offsets, strides or counts along axes, not axis numbers (`slice` bounds and strides, `pad` amounts, convolution and window strides, padding, dilations and group counts, window and slice sizes, `fft_length`) |

`dims`, `size_attrs` and `iteration` are code and lists, not data, and are not in `op_inventory.json`.

### Iteration semantics (`ops/iter.rs`)

A hook sees the ranks of the data operands and results, the literal attributes, which attributes are
present, the operand axes a broadcast expands and whether the op's group counts are 1, and returns the
op's iteration dims (parallel or reduction, where each extent comes from, the dim's position in ω's axis
space) and, per operand and result, an affine map from the iteration vector to the element: one row
`(Σ k · Π factors · i_dim) / divisor` per axis, a factor being a size-attribute entry, a shape-operand
entry or a dim's extent, with flags for padding guards and data-dependent indices. Every formula is the
spec's index formula, cited per op: `dot_general` (result axes, then the contracting pairs),
`convolution` and `dynamic_conv` (strides, rhs and lhs dilation, padding, `window_reversal`, and the
group split of `feature_group_count`), `reduce`, `reduce_window` and `dynamic_reduce_window` (window
strides, dilations, base dilations, padding), `scatter` (window and batching parts affine, the start
index data), `broadcast_in_dim` (projections), and every element-wise op (identity; rank-0 operands read
whole). The maps are parametric, so one map serves every instance of a kernel class; the
`iteration_space` pass composes them over a kernel (docs/passes/iteration_space.md).

### Dimension relations (`ops/dims.rs`)

A rule sees the rank of every operand and result, which of their axes are the literal 1 (a port that
is not part of a graph, such as a shape operand, is absent) and the op's literal attributes, and returns
pairs of equal axes and of equal element types, each transcribed from the spec's constraints and cited:
element-wise ops (every operand of positive rank has the result's shape), `broadcast_in_dim` (operand
axis `i` is result axis `broadcast_dimensions[i]` unless the operand dim is 1), `transpose`,
`concatenate` (all axes but `dimension`), `dot_general` (batch and contracting dims agree; the result is
batch, then lhs free, then rhs free), `convolution` (output batch and feature; the input feature when
`feature_group_count` is 1), `reduce` (kept axes), `reduce_window` and `dynamic_reduce_window` (inputs
agree), `scatter`, `sort`, `reverse`, `dynamic_update_slice`, `pad`, `fft` (leading axes). Every other op
relates nothing (`fresh`): sound, with fewer shared classes. Axes of size 1 are never related.
`dims::relations` adds the element rules of the entry: `same_as(i)` with one result, and for an
element-wise op every operand of operand `i`'s family. The components pass groups the dims of an
abstract graph into classes with these relations (docs/passes/components.md).

## Graphs for matching (`absgraph.rs`)

- `Graph<N>`: nodes with labels `N`; edges `u -> v` with the sorted list of their ends (`EdgeEnd`: the
  result index of `u` and a `PortLabel`: an operand port, `Commutative` for an operand of a commutative
  op, `Region {region, index}` from an owner to a region argument, `Yield {region}` from a region's
  terminator to its owner). Parallel edges are one edge with several ends (`x * x`).
- `wl`: 1-WL colour refinement on the directed, edge-labelled graph (Shervashidze et al., JMLR
  12:2539-2561, 2011, for the undirected node-labelled form): colours from the node's colour and the
  sorted multisets of `(edge label, neighbour colour)` over out- and in-edges, until the number of
  colours stops growing. The graph hash covers the sorted colours and the node and edge counts. Stable
  colours are isomorphism-invariant.
- `isomorphism`: compares node and edge counts, then runs the `vf2` crate (Cordella et al., IEEE TPAMI
  26(10), 2004) in isomorphism mode through an adapter that implements `vf2::Graph` for `Graph` (the
  crate's petgraph integration targets petgraph 0.6). The caller's predicate gets node indices; a
  `budget` bounds the candidate pairs.
- `check_isomorphism`: an O(V + E log d) check of a stored map (bijection, node predicate, every edge to
  an edge with an equal label, equal counts), independent of the search.
- `UnionFind`: deterministic roots (the smaller index wins), classes numbered by first occurrence.

Adding an op: one entry in `TABLE` (with its spec anchor and shape rule), a unit test of the rule, and the
op then appears with its entry in `op_inventory.json`.
