# `import`

1. **Purpose.** Each exported phase (`graph.json`) → a `cc_ir::Module` (docs/framework/ir.md), the IR every
   later analysis pass reads. Calls are inlined with provenance; every value gets a type whose dimensions
   are constants or symbolic expressions over the phase's size symbols; size computation and constant
   computation are tagged; every op is classified by the op-semantics table. Position: `export → import → components`.

2. **Background.** Design §2, §4.2; `docs/tools/stablehlo.md`, `docs/tools/jax_export.md`.
   - StableHLO is SSA in single-block regions; `func.call` has no semantics of its own, so it is inlined;
     a `stablehlo.composite` "can be replaced with its decomposition without changing program semantics"
     (spec `#composite`): the module's `cc.*` composites are the component boundaries and stay ops.
   - With symbolic shapes, a tensor type holds `?` for every input-dependent dim. JAX passes the dimension
     variables to every private function as leading `tensor<i32>` arguments marked
     `jax.global_constant = "<symbol>"` and computes sizes explicitly (`get_dimension_size`, integer
     arithmetic, shape vectors fed to `dynamic_broadcast_in_dim`, `dynamic_reshape`, `dynamic_iota`,
     `dynamic_pad`, `dynamic_gather`, `real_dynamic_slice`, `custom_call @stablehlo.dynamic_reduce_window`,
     checked by `custom_call @shape_assertion`; `jax/_src/export/shape_poly.py`, jax 0.12.0). The import
     recovers each `?` as an expression in JAX's dimension algebra (sums of products of symbols and of
     `floordiv`, `mod`, `max`, `min`; `shape_poly.py` `_DimExpr`) by abstract interpretation of that size
     computation (Cousot & Cousot, POPL 1977) over the expressions.
   - Integer `divide` and `remainder` truncate (spec `#divide`, `#remainder`); JAX lowers Python's floor
     division and modulo as `select(sign(a) != sign(b) and rem(a, b) != 0, divide(a, b) - 1, divide(a, b))`
     and `select((r < 0) != (b < 0) and r != 0, r + b, r)` with `r = rem(a, b)` (`jax/_src/lax/lax.py`).

3. **Inputs and outputs.** Input: `build/.export/out/export/<model>/{export.json, <phase>/{graph.json,
   symbols.json, census.json}}`. Outputs (`build/.import/out/import/`; schemas in `docs/schemas/import_*`):
   - `<model>/<phase>/ir.postcard`: the `cc_ir::Module` (postcard; `import_ir.schema.json` documents its
     JSON form).
   - `<model>/<phase>/ir.postcard.summary.json` (`Summary`): the structural hash, symbols and constraints,
     counts of functions, ops and values, ops per class and per name, size ops and their share of the
     non-structural ops, `dim_as_data` edges per consuming op, component composites by name, values with a
     symbolic dim, constant-computation ops (size computation included) with their share and, without
     size computation, per name, the `const_in` edges (a constant-computation value entering an op that
     is not one, at a non-shape port) per consuming op name, and the inlining ledger (calls, inlined composites, returns removed, per executed call
     site).
   - `op_inventory.json` (`Inventory`): every op (`name`, or `name@target` for custom calls) of every
     phase with its table entry (or `null`), its count and the models using it; `unknown` lists the ops
     without an entry.
   - `views/<model>/<phase>.dot`: the entry function with every composite collapsed into one node (size
     computation and constants left out).

4. **Algorithm.** Per phase, phases in parallel:
   1. *Build* (`algo/build.rs`), one walk from the entry function. The entry's argument types come from
      `graph.json` `inputs[].shape` (parsed into `SymExpr`). Each op is looked up in the op table (an
      unknown op or custom-call target is a hard error naming model, phase, function and op) and checked
      against its arity. A `func.call`, or a composite whose name matches `inline_composites`, is inlined:
      its body is emitted in place with its arguments bound to the call's operands, and the call site
      (`graph.json` op id, callee) is appended to every inlined op's `inlined_from`. A `cc.*` composite
      becomes an op whose decomposition is built at that call site, with the call's operand types; a
      decomposition reused by another call site must be called with the same types. A custom call's
      `called_computations` function is kept as a computation.
   2. *Shapes.* For every op, the table's shape rule derives the result dims from the operand types and,
      for value-dependent shapes, from the operands' computed values. A declared constant dim must equal
      the derived one (or be provably equal under the scope's bounds); a declared `?` takes the derived
      expression. A dim that cannot be derived, or that depends on a run-time value, is a hard error.
   3. *Size values* (`algo/size_value.rs`). The elements of every small static integer or boolean tensor
      (at most 64 elements) are tracked as `SymExpr`s: dimension arguments are their symbol,
      `get_dimension_size` its operand's dim, constants and `iota` their values, and integer arithmetic,
      comparisons, `select`, `convert`, `reshape`, `broadcast_in_dim`, `concatenate`, `slice` and
      `transpose` act element-wise with StableHLO's semantics. Each element is a set of guarded cases
      (`guard → value`, guards are conjunctions of facts `f >= 0`): a comparison or sign the bounds cannot
      decide splits into its cases, infeasible cases are dropped, agreeing cases merge. The two floor
      identities above are applied where the IR has their select structure (an 8-bit exhaustive unit test
      checks both). Integers computed from run-time data are opaque variables; cases are never split on
      them, so a slice of fixed size at a run-time offset still gets the size `limit - start`.
   4. *Bounds* (`cc_ir::SymCtx`): every symbol is at least 1; a constraint with one variable bounds it; every
      constraint is a fact. `e >= 0` is proven by interval arithmetic, directly or after subtracting one
      fact.
   5. *Size tagging* (`algo/size_tag.rs`). An op is size computation if it is a size op by definition
      (class `size`), or if it computes small integer values from a dimension value and every use of its
      results is a shape operand or another size op (dead size code included); this is the greatest
      fixpoint of that rule. A size value consumed at a non-shape port of any other op is a `dim_as_data`
      port: the dimension enters compute there as a run-time scalar (anchor grids, position ids, the mean
      over a symbolic axis). Such uses are allowed and recorded; the consuming op stays compute.
   6. *Constant tagging* (`algo/const_tag.rs`). An op computes a constant when no data argument of the
      phase reaches it: data arguments are the entry's arguments except the dimension variables
      (activations, weights and run-time inputs are all entry arguments of the abstract export). The
      reachability is a forward work list over the inlined graph: a reached value reaches each user at
      a port that is not a shape operand; a reached op reaches its results, its regions' arguments, the
      arguments of its custom-call computation and its region's owner (a value captured from the
      enclosing scope); a reached composite operand reaches the decomposition argument it binds. A
      decomposition shared by several calls is reached from any of them, so its ops are constant only
      when they are constant at every call. Size computation is never reached (its values are
      dimensions, not data). Every unreached op is tagged `constant` except the literals (`constant`,
      `iota`, and layout ops whose data operands are literals: `normalize` folds splat and broadcast
      constants into ports, so they stay ops) and structure (returns, composite calls); the ops in the
      regions of a tagged op are tagged with it. Size computation is therefore a part of constant
      computation. Its typical members are position and rotary tables (`iota → convert → multiply →
      sin/cos`), masks compared from two `iota`s, and the float form of a size (`convert` of a
      dimension, the divisor of a mean over a symbolic axis). Later passes leave tagged ops out of their
      graphs; a tagged value read by compute enters as an input (`const_in`, docs/passes/components.md).
      Folding or precomputing the values is outside this pass: the tag marks what is not hardware.
   7. *Write* the module, its summary and the view; then the inventory over all phases.

   Cost: linear in the inlined op count (each op once; shape rules are O(rank); the constant tag visits
   every use once); a guarded value has at
   most 8 cases and a combination at most 64. The whole model set at `full` (78 phases, 2.03 million ops
   after inlining, half of them size computation, 7445 `dim_as_data` edges) takes about 22 s with 8
   threads (2026-11-01). Constant computation is 1.25 million of those ops (227 k beyond size
   computation) with 29 503 `const_in` edges; the pass then takes about 19 s (2026-11-05).

5. **Config** (`[passes.import]`): `inline_composites = ["chlo.*"]` (composite names inlined instead of
   kept; a trailing `*` matches a prefix), `views = true`. Resources: `threads = 8`, `memory_gb = 8`,
   `timeout_min = 30`.

6. **Verification** (`verify.rs`; it reads the written module and the export's files and never calls the
   builder). Per phase, `<model>:<phase>:<check>`:
   - `ssa`: every value is defined once, by the op or region its `def` names; every use is dominated by
     its definition (earlier in the region, or in an enclosing one); terminators are last; region owners
     and parents agree. The other checks are skipped on a malformed module.
   - `types`: every value agrees with its declared type in `graph.json` (rank, element type, every
     declared constant dim); every op's result element types follow its table rule; for every op whose
     shape does not depend on operand values, the shape rule re-derived from the stored operand types
     equals the stored result shape.
   - `sym_eval`: translation validation of the symbolic interpretation. At up to three sample assignments
     of the symbols satisfying the constraints (from `testing.tiny_dim`, repaired constraint by
     constraint), an independent concrete interpreter of the module's integer ops (StableHLO semantics,
     checked arithmetic; run-time integers taken at 0) recomputes every size value; at each point every
     op's stored result dims equal its shape rule on the concrete operands, every shape operand is
     computable, every shape assertion holds and every dimension argument of a composite is bound to its
     symbol's value.
   - `census`: the expansion of `graph.json` from the entry recounted here equals `census.json`
     `executed`; the module's op multiset (composites expanded at every call site) equals it minus what
     inlining removes (calls, inlined composites, returns of inlined bodies); the stored ledger equals the
     recount; the GEMM and convolution signatures with their declared shapes equal `census.json` `ops`.
   - `coverage`: every op has a table entry and fits its arity; no call survived; every decomposition has
     a composite.
   - `size_tag`: the tags and `dim_as_data` ports equal an independent recomputation (work lists and an
     iterated greatest fixpoint instead of the builder's ordered passes).
   - `const_tag`: the constant tags equal an independent recomputation: data dependence as the least
     fixpoint of repeated sweeps that pull from operands, region ops and call sites (instead of the
     tagger's work list), literals and the region closure as fixpoints of their own. A tagged op that a
     data argument reaches is reported as such (no tagged op has a path from a data argument), and every
     size-computation op must be tagged. Counts: tagged ops and `const_in` edges.
   - `summary`: the stored summary equals the one recomputed from the module, structural hash included.

   Once: `imported` (every exported phase has a module) and `inventory` (every op of every phase is listed
   for its model; none is unknown; entries agree with the table). Every check has a negative test in
   `tests/smoke/import.rs`, which also checks that the structural hash does not change when the ops of
   `graph.json` are reordered (any valid order) or its values renumbered.

7. **Resources.** Memory scales with the largest phases in flight: the parsed `graph.json` plus the module
   (peak 1.2 GB for qwen3vl alone, 4.4 GB for the whole set at `full` with 8 phases in parallel; 1.4 GB at
   `reduced`; 2026-11-01). Outputs: 263 MB for the model set at `full` (about 2 MB per 16 k ops).

8. **Limitations.**
   - The expression normal form is canonical but not a decision procedure: two equal dims with different
     forms are unified only when the bounds prove them equal, else the import fails at that op (none of
     the 78 phases does).
   - Comparisons are decided with intervals and single facts; the cases of a guard are pruned only when
     an interval or a fact contradicts them.
   - Run-time integers are taken at 0 in `sym_eval`; the builder already rejects dims that depend on them.
   - Constant computation is tagged, not evaluated: its values reach later passes as `const_in` inputs,
     not as literals. A scalar computed from constants (a quantization bound written as `convert` of an
     integer constant) is therefore an input of its kernel where a literal would have been folded into a
     port by `normalize`.
   - A decomposition shared by several calls is tagged once, for all of them: an op that is constant at
     one call and reached at another stays untagged.
   - Every composite with a `chlo.` name is inlined by default: JAX emits `chlo.top_k`, `chlo.erf` and
     `chlo.erfc` as composites, which are not module boundaries.
   - The table holds the ops the 25 models use plus the dynamic ops of the spec; another op needs an
     entry (`crates/cc-ir/src/ops/mod.rs`).
