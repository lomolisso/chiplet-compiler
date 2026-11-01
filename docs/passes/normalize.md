# `normalize`

1. **Purpose.** Removes incidental structure from every component with rewrites that keep its meaning
   exactly, so that `kernels` and `kernel_classes` see one structure for one computation. It also fixes
   the reduction order ω of every reduction, which pins what a float reduction computes. Each rewrite rule
   is proven once with SMT; each run writes a ledger of the rewrites it applied, which the verification
   replays. Position: `components → normalize → kernels`.

2. **Background.** Design §4.4, §5; the components in docs/passes/components.md; the SMT runner in
   docs/framework/smt.md.
   - Rewrite rules proven once for all inputs, and applications that carry only a match: Alive (Lopes,
     Menendez, Nagarakatte, Regehr, "Provably correct peephole optimizations with Alive", PLDI 2015,
     DOI 10.1145/2737924.2737965). The obligations are SMT-LIB 2 text (Barrett, Fontaine, Tinelli, *The
     SMT-LIB Standard 2.6*), checked by Bitwuzla (Niemetz, Preiner, CAV 2023, DOI
     10.1007/978-3-031-37703-7_1) and re-checked by Z3 (de Moura, Bjørner, TACAS 2008, DOI
     10.1007/978-3-540-78800-3_24). `unknown`, a timeout or an error is never a proof.
   - Translation validation (Pnueli, Siegel, Singerman 1998; Necula 2000): the ledger is the witness of a
     run, checked without the pass's rewriter.
   - Float addition is not associative (e.g. in binary32, `(1e8 + -1e8) + 1 = 1` but `1e8 + (-1e8 + 1) = 0`),
     so a float reduction has no single meaning until its order is fixed. StableHLO leaves it open:
     `reduce` computes "`exec(schedule)` for some binary tree `schedule`", an implementation-defined full
     binary tree whose in-order traversal is the inputs "in the ascending lexicographic order of `index`",
     "interspersed with an implementation-defined amount of `init_values`"; `dot_general` is defined as a
     `reduce` with `init_values=[constant(0, element_type(result))]`; `convolution` as a `dot_general`;
     `scatter`'s `schedule` is "an implementation-defined permutation of `index_space(updates[0])`"
     (spec `#reduce`, `#dot_general`, `#convolution`, `#scatter`). Error bounds per order: Higham,
     *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, ch. 4 (sequential summation
     `γ_{n-1} Σ|x_i|`, pairwise `γ_{⌈log2 n⌉} Σ|x_i|`).
   - No rule reassociates, contracts `a*b + c` into a fused multiply-add, or simplifies float algebra.

3. **Inputs and outputs.** Input: `build/.components/out/components/components/<id>.postcard` and its
   summary. Outputs (`build/.normalize/out/normalize/`; schemas `docs/schemas/normalize_*`):
   - `components/<id>.postcard` (`NormalizedComponent`): the normalized graph (`Graph<NNode>`), the input's
     height, interface and class counts. An `NNode` is the input node's label (`ANode`), its ω (on
     reductions), its folded operands (`splats`: port, the constant's and the broadcast's input node, the
     operand's type) and its `origin`, the node of the input component. Ids are the input's: one
     normalized graph per component, no new deduplication. Nodes keep their class ids, so every instance's
     bindings (docs/passes/components.md §3) apply through `origin`; a folded constant's value is the
     parameter of its `constant` node.
   - `components/<id>.postcard.summary.json` (`NormalizedSummary`): nodes before and after, rewrites per
     rule, reductions per order, instances per model.
   - `ledger/<id>.json` (`LedgerFile`): every rewrite in order: rule, matched nodes, edits (`unlink`,
     `link`, `relabel`, `remove`; node ids are the input's).
   - `rules/<rule>.smt2` (the obligation) and `rules/<rule>.result` (`RuleResult`: expected verdict, the
     query's blake3, solver, version, verdict with the model when `sat`).
   - `views/normalize.md`: rewrites per rule, and nodes and reductions weighted by instances per model.

4. **Algorithm.**
   1. *Proofs* (`proofs.rs`), before any rewrite: write every obligation, run the configured solver on all
      of them in one process (`cc_smt::run_batch`), write the results; the pass fails unless each rule is
      `unsat` and the controls answer as expected.
   2. *Rewrites* (`algo/rules.rs`), per component in parallel. The working graph keeps the input's node
      ids and a multiset of edge ends; rules never create nodes. Sweeps apply the rules in the order below,
      each over the nodes in ascending id, until no rule matches. Every step removes a node or an edge end
      or shortens a transpose chain, so the loop ends.
      Never rewritten: the interface (`input`, `output`, region arguments, `size_in`), composite calls
      (each is a child instance), `func.return`, region terminators and region owners, ops without
      results.

      | rule | match → rewrite | obligation (`rules/<rule>.smt2`) |
      |---|---|---|
      | `dce` | an op without uses → removed | the outputs `g(x)` do not depend on an unused `h(x)` (QF_UF); the op table has no op with an effect inside a graph (size assertions are not in graphs) |
      | `transpose_identity` | `transpose(x, [0…r-1])` → `x` | the forward index map `result_index[d] = operand_index[permutation[d]]` is the identity, ranks 1–8 |
      | `broadcast_identity` | `broadcast_in_dim` / `dynamic_broadcast_in_dim` with `broadcast_dimensions = [0…r-1]` and the operand's type (equal classes) → `x` | `operand_index[d] = 0 if dim(operand, d) = 1 else result_index[d]` reads `x` at the same in-bounds index, ranks 1–8 |
      | `convert_identity` | `convert` whose operand has the result's element class and dims → `x` | `to_fp` (RNE) to the same format is the identity for f64, f32, f16 (QF_FP); the encoding is kept for every type up to 64 bits (bf16 and FP8 included: Bitwuzla supports only the IEEE interchange formats) |
      | `transpose_collapse` | `transpose(transpose(x, p), q)` → `transpose(x, c)`, `c[d] = p[q[d]]` | both sides send every operand index to the same result index: `fwd(p, xi)[j] = xi[p[j]]` for symbolic `j` (each `q[d]` is one), `p` and `xi`, ranks 1–8; transposes are bijections (`#transpose` C2), so equal index maps give equal tensors |
      | `reshape_collapse` | `reshape(reshape(x))` (or `dynamic_reshape`) → the outer reshape reads `x` | reshape keeps the row-major order (`#reshape`): `reshape_s = unflatten_s ∘ flatten` with `flatten ∘ unflatten_s = id`, so `reshape_s2 ∘ reshape_s1 = reshape_s2` (QF_UF) |
      | `splat_fold` | an operand of an element-wise op that is a splat `constant`, or a `broadcast_in_dim` of a rank-0 `constant` → a port annotation | a rank-0 operand broadcast has an empty `operand_index`, so every element is the scalar: `f(…, bcast(c), …)[i] = f(…, c, …)` for every element-wise `f` (uninterpreted), arities 1–3, every port, ranks 0–8 |
      | `cse` | two ops with equal labels (op, attributes, result types with their class ids), the same incoming ends, no parameters, folded operands or regions, both outside every region → one | functional congruence (QF_UF), arities 1–3 |

      Tensors are uninterpreted functions of bit-vector indices, so a proof holds for every element type.
      Index entries that are only compared for equality use 4-bit vectors: at most 8 occur, so any
      counterexample renames into that width. Layout rules match only up to rank 8 (`MAX_RANK`).
      A node is inside a region when it is reachable from a region argument or reaches a region
      terminator along data edges; CSE leaves those nodes alone (merging across scopes would hoist code).
      Two reshapes of one value to different shapes have different (fresh) classes, so CSE never merges
      them; the result types must be equal in every instance.
   3. *ω* (`algo/omega.rs`) on every reduction: `order` from `[passes.normalize.order]` for the family of
      its results (a float or complex result decides), over `axes`:

      | op | axes (of the operand the definition reduces) |
      |---|---|
      | `reduce` | `dimensions`, ascending |
      | `reduce_window`, `dynamic_reduce_window` | every window axis (padding elements are init values, in place) |
      | `dot_general` | `lhs_contracting_dimensions`, in attribute order (the spec's reshaped slice) |
      | `convolution`, `dynamic_conv` | `kernel_spatial_dimensions`, then `kernel_input_feature_dimension` (the spec's `rhs_contracting_dimensions`) |
      | `scatter` | every axis of `updates[0]`; always `seq` (updates apply one at a time) |

      - `seq`: one accumulator; the init value, then the elements in ascending lexicographic order of
        `axes` (outermost first). It is the spec's left-deep schedule with one init value first.
      - `tree:<b>`: the elements in the same order, grouped by `b` (each group left to right), level by
        level, the init value combined at the root: also one of the spec's schedules.
      - Default `seq` for every family. Integer families may be left out (`seq`: every order gives the
        same bits for two's-complement arithmetic); `float` and `complex` must be set, or the pass refuses
        to run. A systolic chain and a single-accumulator MAC compute `seq`; the latency of a pipelined
        float adder is hidden by interleaving independent outputs, not by reordering one sum.
   4. *Write* the normalized graphs (nodes in ascending input id), summaries, ledgers and the view.

   Model set at `full` (712 components, 2026-11-01): 82 226 nodes before, 74 993 after; 5 804 splat folds,
   5 699 DCE (8 dead in the input, the rest folded constants and broadcasts), 1 392 CSE, 132 identity
   converts, 18 reshape collapses, 10 identity broadcasts, 1 transpose collapse.

5. **Config** (`[passes.normalize]`): `views = true`; `[passes.normalize.order]`: `bool`, `sint`, `uint`
   (default `"seq"`), `float`, `complex` (required): `"seq"` or `"tree:<fan-in>"` with fan-in ≥ 2.
   `[tools.smt]`: `solver` (`bitwuzla` proves, the other re-checks), `timeout_s` per query, `runner`
   (`docker`: the solvers of the toolchain image; `host`: the binaries `z3` and `bitwuzla`). The solver and
   the image tag enter the cache key. Resources: `threads = 8`, `memory_gb = 2`, `timeout_min = 30`.

6. **Verification** (`verify.rs`; it reads the input components and the written files and never calls the
   pass's rewriter; its rule predicates and applier are its own):
   - `rules:<rule>`: the obligation regenerates byte for byte; the stored result is the configured
     solver's answer to this query (its hash) and is `unsat`; the other solver also answers `unsat`.
   - `solver_sanity`: both solvers answer the controls: integer reassociation `unsat`, float reassociation
     in binary16 `sat` (an unsound rule fails its proof).
   - `complete`: every input component has a normalized graph and a ledger.
   - `<id>:replay`: from the input graph, every ledger step's match satisfies its rule, its edits are
     exactly the rule's edits for that match, and the replayed graph equals the stored one (ω aside).
   - `<id>:normal_form`: no rule matches the stored graph.
   - `<id>:omega`: every reduction carries the ω its family's config and the op's definition give; no
     other node has one.
   - `<id>:classes`: classes recomputed on the stored graph (`components::algo::classes::assign`) refine
     the stored class ids, so every instance binding stays valid; the interface is unchanged; every
     interface and call node is kept with its label.
   Every check has a negative test in `tests/smoke/normalize.rs` (a wrong proof result, an edited query, an
   edited ledger step, a dropped ω, a leftover pattern, a merged class, a missing ledger), next to a direct
   test that float reassociation is `sat` for both solvers and a property test that renumbering a
   component's nodes leaves its normalized graph unchanged up to numbering.

7. **Resources.** Small: the components are about 9 MB at `full`. Run and verification together at `full`:
   11 s, 165 MB peak RSS; the proofs take 2 s (Bitwuzla) and 3.4 s (Z3) in one container each
   (2026-11-01).

8. **Limitations.**
   - The rule set is small: identities only where the labelled graph proves them (a reshape back to its
     operand's shape has fresh classes and stays); no slice, pad or concatenate rules; no constant folding.
   - CSE skips ops with parameters (constants, size attributes): their values differ per instance, and
     merging them would need every instance to agree.
   - `fft` has no ω: the spec leaves its algorithm open, not only an order. `sort` is not a reduction.
   - ω fixes the order and the init position; mixed-precision products and accumulation types
     (`precision_config`, `DotAlgorithm`) are not part of it (`datapaths` pins the scalar arithmetic).
   - Identifying `dynamic_*` ops with their static forms is not a rule yet (components keeps them apart).
   - `dce` and `cse` obligations are congruence facts; their real premise is that every op in a graph is
     pure, which follows from the op table, not from the solver.
