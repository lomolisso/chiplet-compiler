# `datapaths`

1. **Purpose.** For every kernel class Z, the pass extracts the scalar datapath of one iteration (what one
   functional-unit invocation computes) and proves it equal to the StableHLO semantics of the kernel's
   ops, for every width tuple of the class's instance table T_z. A class routed to a template keeps only
   its per-element stages: the prologue of each core operand (once per element at the array's edge) and
   the epilogue (once per result element); the array computes the core. The pass also states, per
   accumulator, whether its combiner is commutative and associative (so any reduction order gives the
   same bits) or ω binds. `mine` and `merge` work on these graphs, and `merge` reuses the SMT harness
   (docs/framework/smt.md). Position: `iteration_space → datapaths → mine`.

2. **Background.**
   - Design §4.8, §5, §6; the op table (docs/framework/ir.md); kernels and their roles (core, body,
     prologue, epilogue, map, broadcast: docs/passes/kernels.md); classes, element classes and T_z
     (docs/passes/kernel_classes.md); skeletons, accesses and routing (docs/passes/iteration_space.md).
   - Translation validation (Pnueli, Siegel, Singerman, TACAS 1998; Necula, PLDI 2000): each run proves
     its output against its input; the extraction may be wrong, the proof catches it.
   - Combinational equivalence by a miter: two circuits are equal iff "some output differs" is
     unsatisfiable. Bit-vectors (QF_BV) are decided by word-level rewriting and bit-blasting; IEEE floats
     by the SMT-LIB FloatingPoint theory (Brain, Tinelli, Rümmer, Wahl, ARITH 2015), bit-blasted through
     SymFPU (Brain, Schanda, Sun, TACAS 2019). Solvers: Bitwuzla (Niemetz, Preiner, CAV 2023) and Z3
     (de Moura, Bjørner, TACAS 2008).
   - Float reductions are not associative (Higham, *Accuracy and Stability of Numerical Algorithms*,
     2nd ed., SIAM 2002, ch. 4); two's-complement sums are. The StableHLO spec leaves the order of
     `reduce` open; ω (docs/passes/normalize.md) fixes it.
   - Float formats: IEEE 754-2019; OCP 8-bit Floating Point (OFP8) for E4M3FN (no infinities, one NaN
     pattern, maximum 448); ml_dtypes' conversions (`float8.h`) for overflow to NaN.

3. **Inputs and outputs.**
   - Inputs: `kernel_classes` (classes, instance tables), `kernels` (the representative graphs),
     `iteration_space` (`routing.json`, `skeletons/<z>.json`).
   - `out/datapaths/datapaths/<z>.json` (`Datapath`, schema `datapaths_datapath`):
     - `route`, `class`, `core`; `widths`: width variable → family (`w<e>` is element class `e`; a
       folded constant's width is its value column);
     - `vertices`: `id`, `label` (`<op>@<family>`), `op`, `family`, `width`, `attrs` (`comparison_direction`
       with GT/GE written as LT/LE and the operands exchanged, `compare_type`, `reduce_precision`'s bit
       columns; a constant's `column` or the literal `value` 0), `stage` (`prologue {port}`, `core`,
       `epilogue`, `map`), `args` (`src`, `port`: `p<i>` or `*` for a commutative op), `from` (the kernel
       node and its access instance: the extraction certificate);
     - `src` is a vertex result, an input by name, accumulator `i`'s current value (`acc`), or core
       result `i`'s final value (`result`, what the epilogue reads);
     - `inputs`: `name` (the skeleton operand: `in<j>`, `in<j>.<u>` for a second access of the same
       value, `indices`; `core_result<i>` for a template), `kind` (`prologue`, `core_operand`,
       `epilogue_param`, `map`, `address`, `core_result`), the kernel node, `incidence` (the iteration dims
       that index it), `width`;
     - `outputs`: `out<j>` (kernel outputs) and, for a template, the core inputs by port name (`lhs`,
       `rhs`): the ends of the prologue stages;
     - `accumulator` (datapath classes with a reduction): one entry per result (`width`, `init`, `next`),
       the `combiner` vertices, ω, and per tuple `order` (`free`: every order gives the same bits);
     - `tuples`: `name` (`t<k>`), `elems` (per element class), `consts` (per constant column, its
       element type), `attrs` (structural attributes), `rows`.
   - `out/datapaths/proofs/<z>/<t>.smt2` and `.result` (`ProofResult`): the equivalence query for tuple
     `t` and its verdict (`expect` `unsat`); `<t>.order.smt2` and `.result`: the order query (`expect`
     `either`).
   - `out/datapaths/lemmas/<format>.<what>.smt2` and `.result`: the encoding lemmas the queries need
     (Z3, `unsat`).
   - `out/datapaths/summary.json` (`Summary`): counts by route, vertices and their labels (the census
     `mine` works on), tuples, proofs, order queries (free / bound), lemmas, allowed unknowns, the
     uninterpreted functions used.
   - `views/datapaths.md`; `cache/` (solver verdicts by query hash, outside `out/`).

4. **Algorithm.**
   1. *Load* every class, its T_z, its representative kernel graph, its skeleton and its route.
   2. *Extract* (`algo/extract.rs`), from the op table only:
      - The kernel outputs and, for a template, its core inputs are resolved backwards through the
        skeleton's accesses: a member read at access `a` is the vertex of its access instance with
        result `a`, created once per instance (a broadcast feeding both prologues gives two values); a
        broadcast is the identity on one element (spec `broadcast_in_dim`): its consumer reads its
        operand at the operand's access; a kernel input read at `a` is the skeleton operand with that
        access; a folded constant is a `const` vertex for its value column; a core result is
        `result {i}`.
      - An element-wise member becomes `<op>@<family>` with its result's element class as width,
        `convert` included; GT/GE become LT/LE with exchanged operands, as kernel class identity
        writes them.
      - A contraction core (`dot_general`, `convolution`, `dynamic_conv`): the operands converted to
        the result's element type where their element classes differ (the spec's `dot_general` is
        `reduce(multiply(lhs, rhs), 0, add)`, and `multiply` needs one element type), `multiply`, and the
        accumulator's `add` on `acc 0`, init `const 0`.
      - A reduction core (`reduce`, `reduce_window`, `dynamic_reduce_window`): its body on (accumulators,
        inputs), init the `init` operands; a scatter core: its body on (current results, updates), init
        the operands' elements, the indices an `address` input (not extracted); a scatter without a
        reduction (class `map`) applies its body once, with no accumulator.
      - A template class: no core vertices; every vertex is in a prologue or the epilogue.
   3. *Tuples* (`algo/tuples.rs`): the distinct (element types per class, element types of the
      folded constants, `reduce_precision` bits) over T_z's rows, sorted, `t0`, `t1`, …. Constant values
      are not part of a tuple: they are free variables of the proofs. `max_tuples` bounds the count.
   4. *Obligations* (`proofs.rs`), per tuple, in one term DAG:
      - the kernel's values (`spec.rs`): the representative's members evaluated with `cc_smt::ops` under
        the tuple's types, from the graph (original operand order and attributes), with the core's
        spec for one ω-`seq` step: `acc + convert(lhs) · convert(rhs)` and init 0 (contractions);
        `body(acc…, x…)` with the accumulators as the first arguments, the left-deep schedule with the
        init value first, and the `init` operands (reductions); `body(current…, update…)` and the
        operands' elements (scatter);
      - the datapath's values (`eval.rs`): its vertices in order;
      - the miter: some value differs (every kernel output, a template's core inputs, every
        accumulator's next and init value), with float equality bit-exact and all NaNs equal. `unsat`
        proves the datapath for all inputs and all constant values;
      - the order query: `F(a, b) ≠ F(b, a)` or `F(F(a, b), c) ≠ F(a, F(b, c))` for the combiner `F`
        (from the graph); `unsat` proves order freedom.
   5. *Lemmas*: the encodings the queries used (`Ctx::used`) give the lemma files
      (docs/framework/smt.md, "Floats").
   6. *Solve*: cached verdicts first (`cache/`), then one sharded batch of the configured solver over
      the proofs and order queries and one of Z3 over the lemmas (`threads` shards, each with a share
      of the memory cap). A proof that is not `unsat` fails the pass unless the class is in
      `allow_unknown` with an `unknown` or timeout; a lemma must be `unsat`.
   7. *Write* the results, the datapaths with their order flags, the summary and the view.

   Cost: extraction and obligations are linear in the kernel graph per tuple; solving dominates. The
   proofs are cheap because both sides are built in one hash-consed DAG with commutative operands in a
   canonical order: equal subterms are one node, and the solver works only where the two sides differ
   (an exchanged comparison, a broadcast, an extra convert).

5. **Config** (`[passes.datapaths]`): `allow_unknown = []` (`[{ class = "z…", reason = "…" }]`),
   `max_tuples = 64`, `lemma_timeout_s = 600`, `views = true`. `[tools.smt]`: `solver` (`bitwuzla` proves
   the obligations, the other re-checks), `timeout_s` per query, `runner`. The solver and the image tag
   enter the cache key. Resources: `threads = 16`, `memory_gb = 8`, `timeout_min = 120`.

6. **Verification** (`verify.rs`; it never calls the extraction or the tuple builder):
   - `complete`: every class has a datapath; its tuples are the distinct (element types, constant
     types, structural attributes) of its rows, recomputed; every tuple has its proof and result, every
     accumulator its order query and result.
   - `<z>:structure`: vertices in topological order with every source declared; each vertex is the op
     and family of the kernel node it names (core vertices: `multiply`, `add`, `convert` of a
     contraction), its width the node's element class; constants are folded operands of their node and
     value columns every row has; directions only as LT/LE for GT/GE; inputs are skeleton operands
     (or a template's core results); outputs are the kernel's (and a template's core inputs); stages
     agree with the node roles and, for a template, with the routing's stage lists; an accumulator
     exactly for datapath classes with a reduction, with the skeleton's ω; no dead vertex.
   - `<z>:proof`: every stored query equals the one regenerated from the stored datapath; its result is
     for that query and solver, `unsat` (or allowed with the configured reason); the second solver
     re-checks every query (a claimed order freedom must be re-proven, a counterexample must not be
     refuted); the order flags are the order verdicts.
   - `lemmas`: the lemma files are those the regenerated queries need, byte for byte; each stored and
     re-run result is `unsat` in Z3.
   - `solver_sanity`: the first non-commutative binary vertex (`subtract`, `divide`, …) with distinct
     operands gets its operands exchanged; its obligation must be `sat` in both solvers.
   - `summary`: the summary recounts from the datapaths.

   Negative tests (`tests/smoke/datapaths.rs`): a swapped port, a convert replaced by another op, an
   edited label or stage, an edited query or order flag, a missing result, an unknown verdict without
   (and with) `allow_unknown`, an edited summary. `tests/smoke/smt.rs` checks the op encodings on the
   spec's cases.

7. **Resources.** Model set at `full` (25 models, 830 classes, 2026-11-02, iacoma231, debug build,
   16 threads): 703 datapath and 127 template classes, 5 499 vertices with 55 labels, 920 tuples and
   proofs, 155 order queries (57 free, 98 bound), 8 encoding lemmas. The pass takes about 106 s and its
   verify about 120 s, both dominated by the two bf16 lemmas that bit-blast a multiplier and a divider
   (66 s and 98 s in Z3); every equivalence proof takes milliseconds. Peak RSS 0.32 GB; outputs 15 MB.

8. **Limitations.**
   - Ops whose float result the spec leaves to an approximate function are uninterpreted: a proof
     holds for any implementation of them, and `softfloat` must implement and prove them.
   - Padding and window bounds are address logic: a datapath input reads the operand element (or the
     padding value) the address unit selects; `reduce_window` padding elements are init values in
     place.
   - A scatter's index computation is address generation and is not extracted.
   - Carrier encodings exist for IEEE formats up to 8 exponent and 23 significand bits and for
     finite-NaN formats' conversions; arithmetic in an FN format needs a lemma that does not exist yet
     (none is in use), and fails the pass.
   - `compare_type = TOTALORDER` and integer `power` are not encoded (none is in use).
   - The order query proves commutativity and associativity of one combiner step; it does not model a
     mixed-precision accumulator (none: every accumulator has the core result's type).
