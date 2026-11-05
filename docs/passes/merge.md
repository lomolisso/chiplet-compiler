# `merge`

1. **Purpose.** The pass forms the set F of functional units that covers every kernel class, and gives
   every covered class a configuration of its unit, proven equal to the class's datapath. Datapath
   classes and the per-element stages of template classes are merged into configurable units: blocks
   (operators, constant registers, ports, accumulator registers) shared where the area saved exceeds the
   multiplexers sharing needs. Template classes share parameterized arrays. Position:
   `datapaths, mine, iteration_space, characterize → merge → allo_emit`.

2. **Background.**
   - Design §4.11 and §6; the datapaths and the SMT harness (docs/passes/datapaths.md,
     docs/framework/smt.md), the mined patterns (docs/passes/mine.md), the skeletons, embedding relation,
     routing and space-time mappings (docs/passes/iteration_space.md), the primitive library
     (docs/passes/characterize.md).
   - **Datapath merging** (Moreano, Borin, de Souza, Araujo, "Efficient datapath merging for partially
     reconfigurable architectures", IEEE TCAD 24(7):969-980, 2005). Given datapaths G₁, G₂, find a
     merged datapath and injective maps of both into it that realize every edge, minimizing operator
     plus multiplexer area. The problem is NP-complete (by reduction from subgraph isomorphism, Sec. III).
     Two graphs are merged exactly through their **compatibility graph** (Defs. 6-8): one node per
     *vertex mapping* (a ∈ G₁, b ∈ G₂ that one block can implement) and per *arc mapping* (an edge of
     each whose ends map and whose destination ports match, or are exchanged on a commutative operator);
     two mappings are adjacent when they can hold together (no vertex mapped twice, Def. 7). A maximum-
     weight clique, with weights the area each mapping saves, is the best merge of the pair (Sec. V-C).
     Unmapped blocks are copied; an operand port with several sources gets a multiplexer. More than two
     datapaths are merged pairwise, the merged unit with the next one (Sec. V-E).
   - **APEX** (Melchert et al., ASPLOS 2023) uses the same construction on frequent subgraphs ranked by
     their MIS, and adds a check against combinational cycles: two vertex mappings that order a pair of
     blocks one way in G₁ and the other way in G₂ would make a cycle of two blocks. A cycle can also run
     through four or more blocks with every pair consistent (b₁ → b₂ in G₁, b₂ → b₃ in G₂, b₃ → b₄ in G₁,
     b₄ → b₁ in G₂), so acyclicity is checked on the rebuilt unit.
   - **Maximum-weight clique** is NP-hard (Karp 1972). Östergård's algorithm ("A new algorithm for the
     maximum-weight clique problem", Nordic J. Computing 8(4):424-436, 2001) orders the vertices and
     computes, from the last, the best clique c(i) within {vᵢ, …, vₙ}, pruning a branch when its weight
     plus c(i) cannot beat the incumbent. As an integer program the clique is an assignment problem with
     side constraints: one constraint per vertex of G₁ and of G₂ over the mappings that use it.
   - **Configuration by construction.** A configuration is read off the clique: the block of every
     vertex, the source each multiplexer selects, the value of every constant register. Correctness is
     then a ∀-query per configuration (no ∃∀ synthesis): the unit with the configuration substituted
     equals the datapath on all inputs and constants (design §5, "Configuration by construction").
   - **Width.** A block serving integers of one signedness at several widths runs at the widest; a
     narrower configuration extends its operands by their signedness and truncates the result
     (two's-complement `add`, `multiply`, comparisons, … commute with that). Floats share only within one
     format: rounding a wider result to a narrower format is exact only under conditions on the
     precisions and ranges (Figueroa, "When is double rounding innocuous?", SIGNUM 30(3), 1995), and the
     finite-only formats differ in their special values.
   - **Precision decomposition** (design §6.2): a w-bit integer operand runs on an n-bit multiplier as
     pieces, x = Σₖ xₖ·2^{sₖ}, one product per pass; low pieces are unsigned (n − 1 bits on a signed
     element), the top piece keeps the sign (TPU v1: Jouppi et al., ISCA 2017, half rate for one 16-bit
     operand on its 8-bit array; Bit Fusion: Sharma et al., ISCA 2018).

3. **Inputs and outputs.**
   - Inputs: `datapaths` (`datapaths/<z>.json`), `mine` (`selection/<family>.json`,
     `patterns/<family>/<p>.json`), `iteration_space` (`routing.json`, `embedding.json`,
     `templates/<z>.json`, `skeletons/<z>.json`), `kernel_classes` (`instances/<z>.json`), `library`
     (`library.json`).
   - **Units** are those of `mine`: a datapath class, or one stage of a template class
     (`<z>/prologue<p>`, `<z>/epilogue`), in the families `map`, `reduce`, `contraction`, `stage`. A
     unit's graph has a node per datapath vertex (constants included), per input and template core result
     (`in:<name>`), per accumulator (`acc<i>`: a register with ports `next`, `init`; result 0 its current
     value, result 1 its final value) and per output (`out:<name>`). Every node carries its types at
     every width tuple.
   - Outputs (`out/merge/`; schemas `docs/schemas/merge_*`):
     - `fus/<F>.json` (`FuFile`): the blocks (kind, op, semantic attributes, the widest types served,
       per operand port the sorted sources, the mined-pattern group, the library price), the
       multiplexers, the host class, the covered units with their tuples, area, critical path, bound.
     - `configs/<F>/<unit>/<t>.json` (`ConfigFile`): per datapath node its block, its variable (input,
       constant column, accumulator, output name), its types at tuple `t`, its mode (`native`, `widen`)
       and, for an operator, the block port of each operand; the select of every multiplexer it uses.
     - `proofs/<F>/<unit>/<t>.smt2` and `.result`: the configured unit differs from the datapath.
     - `proofs/widen/<op>_<narrow>_<wide>.smt2` and `.result`: a widened block differs from the op at the
       narrow types.
     - `arrays/<A>.json` (`ArrayFile`): dataflow, host class, array dims as functions of the host's group
       extents (parameters), the processing element `(lhs, rhs, acc)` and its price, and per covered
       class its selected mapping and per tuple its mode (`native`, `widen`, `compose` with the pieces
       and the passes per multiply-accumulate).
     - `proofs/arrays/<A>/<z>/<t>.smt2` and `.result`: the element's step differs from the class's.
     - `trace/<family>.json` (`Trace`): every agglomeration step (the pair, the area saved, units and area
       after it, depth, the clique's size, algorithm, optimality, no-good rounds): the area-vs-units curve.
     - `cliques/<family>/<step>.json` (`CliqueCert`): the compatibility graph of a step with at most
       `certify_max_nodes` nodes: weights, conflicting pairs, arc-to-end implications, no-goods, the
       chosen set, its weight.
     - `requant.json` (`RequantFile`, `fixed` only): per class with rescaling chains, the chains (vertex
       ids in the datapath as `datapaths` wrote it, the integer's types, the scale), `fixed` or `kept` with
       the reason; per integer type its tolerance lemma and whether it is proven.
     - `proofs/lemmas/rescale_<type>.smt2` and `.result`: the tolerance lemma (step 11).
     - `coverage.json` (`CoverageFile`): per class its route, its units and their units of F, its array,
       its stages without vertices; `summary.json` (`Summary`); `views/merge.md`.

4. **Algorithm.**
   1. **Prices** (`algo/library.rs`). Rows of `kit` (default ASAP7): area at the `clock` target, delay
      at the `fast` target (the minimum delay). Operators by their catalog key (`add@i32`,
      `compare.LT@ui8`, `convert@i8>f32`; a complex operator as its real parts); constant and accumulator
      registers as `register@w<n>`; an n-input multiplexer as `mux<n>@w<w>` (more inputs than the
      library: its last per-input increment; wider: copies). A key without a row (the pending float
      ops, design §4.9) gets the largest operator row of its type and is flagged `estimated`.
   2. **Contraction** (`contract = true`). Each unit is covered by occurrences of its family's selected
      patterns: patterns in selection order, each pattern's independent occurrences in the unit (its
      MIS), an occurrence kept when it overlaps none kept before. The blocks of an occurrence form a
      group; a group maps only onto a group of the same pattern, member by member (a macro-vertex).
   3. **Compatibility graph** of units A and B (`algo/compat.rs`). Super-nodes are groups and the
      blocks outside groups, labelled by pattern or by op and semantic attributes. A vertex mapping pairs
      super-nodes with equal labels whose member types join (equal, or integers of one signedness on an
      operator that widens); its weight is Σ members price(a) + price(b) − price(join). An arc mapping
      pairs a wire of A with a wire of B whose ends are mapped (same member, same result) and whose
      ports are equal, or exchanged on a commutative two-operand block (`perm`); its weight is a 2-input
      multiplexer at the port's width. Conflicts: two vertex mappings sharing a super-node; two that
      order two super-nodes oppositely (reachability in A and in B, edges out of accumulator registers
      excluded); arcs whose ends conflict, two arcs of one wire, two orientations of one block.
      |nodes| ≤ |A||B| + |E_A||E_B|, restricted by labels.
   4. **Clique** (`algo/clique.rs`, `[passes.merge.clique] algorithm`):
      - `exact_ilp`: max Σ wᵢxᵢ subject to Σ ≤ 1 over the mappings of each super-node of A and of B,
        x_arc ≤ x_end for both ends, Σ ≤ 1 over the arcs of one wire, an orientation variable per block
        with both orientations, Σ ≤ |g| − 1 per no-good g, and the order conflicts added when a solution
        violates one (re-solved until none does). microlp's branch and bound, `node_budget` nodes; out of
        budget it reports its incumbent as not optimal.
      - `ostergard`: Östergård's search on the compatibility graph as adjacency bitsets (vertices by
        increasing weight), the same no-goods; `node_budget`.
      - `greedy`: by weight, a node (with its arc ends) kept when compatible with every kept node.
      A graph over `max_compat_nodes` nodes, an exact solve without an incumbent, an evaluation past
      `timeout_s` (one deadline for all its cycle rounds) or past 20 cycle rounds is answered by `greedy`,
      recorded as the step's `fallback`. Mappings that save nothing (ports, pairs of equal price) are kept
      only as ends of arc mappings. Both exact strategies maximize over the same set (an
      arc's ends are compatible with everything the arc is, and weights are non-negative, so a maximum
      clique closes under arc ends).
   5. **Rebuild** (`algo/rebuild.rs`). Each vertex mapping becomes one block per member with the joined
      types; every other block is copied. Both sides' configurations are carried over; B's commutative
      pairs flip where a chosen arc mapping says `perm`. A port's sources are the images of the edges
      into it over all configurations; two or more make a multiplexer. If the rebuilt unit has a cycle
      (DFS; edges out of accumulator registers do not count), the vertex mappings of its merged blocks
      on the cycle become a no-good and the clique is solved again.
   6. **Agglomeration** (`algo/agglomerate.rs`). Every unit starts alone. A pair is admissible when its
      hosts are related by ⊑ (`embedding.json`; the merged host is the larger) and no class is in both
      (the stages of one template class run at the same time). A lazy greedy picks the merge with the
      largest gain = area(A) + area(B) − area(merged): every pair is queued under an upper bound (per
      label, the best pairing of super-node prices, sorted, Σ min; a merge saves at most the cheaper of
      two shared blocks and never reduces existing multiplexers); the top entries' gains (clique,
      rebuild, area, critical path) replace their bounds, in parallel batches of a fixed size; the top
      entry is accepted when it is evaluated. With `rank = "greedy"` the candidates are evaluated with
      `greedy` cliques and only the accepted pair is solved by `algorithm` (its solution replaces the
      greedy one when it saves at least as much, else the step records `fallback = "rank"`): one exact
      solve per step instead of one per candidate. The clique objective is linear in the multiplexer
      inputs; the area of the rebuilt unit is not, so the exact clique can save less. A merge whose critical path exceeds the bound is not
      admissible. The family stops when the best exact gain is ≤ 0, or at `target_units` units.
   7. **Depth bound.** With `max_logic_depth = 0`, per family: the deepest single unit's critical path
      (operators and multiplexers at their `fast` delays; inputs, constants and accumulator outputs at 0)
      rounded up to whole clock periods of the kit. XLS pipelines each unit (design §4.14), so the bound
      keeps merging from adding pipeline stages beyond the family's deepest datapath; a non-zero key is
      an absolute bound in ps.
   8. **Arrays** (`algo/arrays.rs`). Per dataflow, the (class, tuple) pairs left are grouped: the element
      type is the tuple with the most work (class workload split by tuple rows); every pair it runs
      joins: `native` (equal), `widen` (narrower integer operands of one signedness, or unsigned into a
      wider signed operand), `compose` (one wider integer operand in pieces: n − 1 unsigned bits each on
      a signed n-bit element, the top piece signed; passes = pieces), with the accumulator type equal.
      Within a group, hosts are chosen greedily by the work that embeds into them (⊑). An array's dims
      are the host's selected mapping's array (parameters); its element price is the two converts, the
      multiply and the add at the accumulator type, and the accumulator register.
   9. **Configurations and proofs** (`proofs.rs`). Per covered unit and tuple: the bindings, the selects
      (the position of the configuration's own source in each multiplexer). The query evaluates the unit
      along its wiring only: each multiplexer takes its selected source; a block computes `ops::apply`
      at the configuration's types; the datapath side is `datapaths::eval::values`, with the same
      variables for inputs, constants and accumulators; outputs and accumulator next/init values are
      compared bit-exactly (all NaNs equal). A block in `widen` mode is justified by its **widening
      lemma**, one per distinct (op, narrow types, wide types): the op at the wide types on operands
      extended by their signedness, truncated, equals the op at the narrow types. Splitting the proof
      this way keeps the configuration query structurally equal to the datapath (an i8 `select` widened
      to i32 in front of two f32 multiplies took the solver over a minute as one query).
      Array queries compare the element's step with `acc + convert(lhs)·convert(rhs)` at the class's
      types. Queries run in the configured solver, cached by hash; composed steps (a multiplier against
      its decomposition, which bit-blasting handles badly: Kaufmann, Biere, Kauers, FMCAD 2019) run in
      Z3 with `compose_timeout_s`, as the datapaths' encoding lemmas do.
   10. **Coverage.** Every datapath class has a configuration of its unit per tuple; every template
       class has an array for every tuple and a configuration per tuple of every non-empty stage. The pass
       fails on a failed proof or an uncovered class.
   11. **Fixed requantization** (`requant_variant = "fixed"`, `requant.rs`; applied to the datapaths
       before the units are built). A **rescaling chain** is `round_nearest_even(multiply(convert(x), s))`
       where, at every tuple, `x` is a signed integer of at most 32 bits and everything else is f32, the
       convert and the multiply have no other reader and the three vertices share a stage outside any
       combiner (mobilenetv4's int8 epilogues: `acc → convert → × s → round → + zp → clip → convert i8`).
       It becomes one vertex `rescale(x, s)` (`cc_smt::ops::rescale`, design §6.3): the scale's
       significand `M` (24 bits) and shift `k = 150 − max(e, 1)` are read off its bits, so `|s| = M·2^-k`
       exactly; the exact product `P = ±x·M` (at most 56 bits) is shifted right with rounding half up,
       `(P + 2^(k−1)) >> k`, or left by at most 7 places (saturating beyond), saturated to i32 and
       converted to f32. The hardware is a 32×24 integer multiplier, a shifter and an adder in place of
       an f32 multiplier, an f32 round and an int-to-f32 convert, and the scale input stays f32 (no
       offline step). The chain's readers (the zero point, the clip) are unchanged. At ASAP7 (DC)
       `rescale@i32` is 636 µm² against 331 µm² for the three f32 blocks it replaces (the fp32 multiply
       works on the integer rounded to 24 bits; `rescale` keeps the exact 32×24 product): `fixed` is a
       verified alternative, not a saving, and `fp32` stays the default.
       - **Tolerance lemma**, one per integer type (`proofs/lemmas/rescale_<type>.smt2`). Part 1: the
         scale decode is exact (`±M·2^-k = s` for every finite s). Part 2: for every product P of the
         type, every shift k and every f32 `p` that IEEE 754 allows for the fp32 chain's product, with
         `Q = rescale` and `R = round_nearest_even(p)`: `|Q| < 2^22` implies `|Q − R| ≤ 1`; otherwise R has
         Q's sign and `|R| ≥ 2^21`, so a clip narrower than that saturates both alike. "Allowed" is the
         standard bound of the two fp32 roundings (convert and multiply, each correct to half an ulp, IEEE
         754-2019 §4.3; Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Thm. 2.2):
         `|p − x·s| ≤ (|p| + |x·s|)·2^-24 + 2^-150`, an overflow to infinity of the same sign past
         2^127, or a zero of either sign when `|x·s| < 2^-94`: the flush of design §6.4 (a product below
         2^-126) and a subnormal scale read as zero (`|x·s| < 2^31 · 2^-126`). P is a free variable (with the facts the decode gives: `|P| < 2^(n+23)`, and `P = 0` or
         `|P| ≥ 2^23` for a normal scale), so the query has no multiplier: stated against the f32 multiply
         itself, the solver must relate two multipliers, which bit-blasting cannot do in practice (the
         direct form ran past 10 minutes at 8 bits; Kaufmann, Biere, Kauers, FMCAD 2019). The
         lemma holds for any rounding within the bound, so for the fp32 chain as deployed.
       - The bound is not an absolute one on the class's outputs: the f32 operations after the round
         (adding a non-integer zero point) can turn a difference of 1 into 2 at an int8 output. The claim is
         the design's: the requantized value is within 1 of fp32's, and saturates alike.
       - A class takes the rewrite when every lemma it needs is proven (`requant_timeout_s`); otherwise it
         keeps fp32, with the lemma and its answer recorded. Its configurations are then proven against
         the rewritten datapath as usual. Mined occurrences are carried through the renumbering; one that
         holds a rewritten vertex is dropped.

   **Worked example** (the unit test `mac_and_add_mul_share_the_operators_with_muxes`). A: `out = a·b + c`
   (`m₁ = multiply(a, b)`, `s₁ = add(m₁, c)`); B: `out = (a + b)·c` (`s₂ = add(a, b)`,
   `m₂ = multiply(s₂, c)`), i32, multiplier 400, adder 40, a 32-bit 2-input multiplexer 4.
   - Vertex mappings: `m₁~m₂` (400), `s₁~s₂` (40), every input of A with every input of B and
     `out~out` (0). Arc mappings need both ends mapped: e.g. `c→m₂:1` with `b→m₁:1` (4), or with
     `a→m₁:0` (perm: the multiply is commutative, 4); `s₁→out` has no partner, since `m₂→out` starts at
     a multiply.
   - `m₁~m₂` and `s₁~s₂` conflict: m₁ → s₁ in A but s₂ → m₂ in B, a two-block cycle. The maximum clique
     takes `m₁~m₂` (400), the input mappings and the arcs they allow.
   - Rebuilt: one multiplier, two adders, the multiplier's ports fed by multiplexers (A: `a`, `b`; B: `s₂`,
     `c`). Gain 400 − the multiplexers. Configuration A selects `a`, `b`; B selects `s₂`, `c`. The proof of
     each is `unsat`; a wrong select gives `sat` (the smoke test `a_wrong_select_fails_the_proof`).

5. **Config keys** (`[passes.merge]`):
   - `requant_variant` (`"recipe"`): requantization hardware (design §6.3). Every recipe rescales in fp32
     today, so `recipe` and `fp32` give the datapaths as they are; `fixed` rewrites the rescaling chains
     (step 11).
   - `requant_timeout_s` (1800): the limit of one `rescale` tolerance lemma.
   - `target_units` (0): stop a family at this many units; 0 stops when no merge saves area.
   - `max_logic_depth` (0): the critical-path bound in ps; 0 derives it per family (step 7).
   - `kit` (`"asap7"`): the library kit that prices merges.
   - `contract` (true): cover units with the mined patterns first (step 2).
   - `max_compat_nodes` (4000): larger compatibility graphs go to `greedy`.
   - `certify_max_nodes` (60): steps with at most this many nodes store their graph for `verify`.
   - `compose_timeout_s` (600): Z3's limit per composed-step proof.
   - `views` (true): write `views/merge.md`.
   - `[passes.merge.clique]`: `algorithm` (`"exact_ilp"` | `"ostergard"` | `"greedy"`), `rank`
     (`"greedy"` | `"algorithm"`: how candidate pairs are evaluated), `node_budget` (2000), `timeout_s`
     (30, one deadline per evaluation).

6. **Verification** (`verify.rs`, `verify.json`; never runs the compatibility graph, the solvers of
   step 4, the rebuild or the agglomeration):
   - `files`: every unit, configuration, array, trace and result parses.
   - `<F>:configs`: each configuration binds every datapath node of its unit to one block (injective) of
     the same kind, op and semantic attributes, at the node's types for that tuple; `native` exactly when
     they are the block's, `widen` only for narrower integers of one signedness on an operator that
     widens; operands of a non-commutative operator keep their ports; every datapath edge is a source of
     the port it uses, at the stored select.
   - `<F>:wiring`: every source of every port is used by some configuration; a port has a multiplexer
     exactly when it has two or more sources, of that size.
   - `<F>:dag`: no cycle (Kahn's algorithm; edges out of accumulator registers excepted).
   - `<F>:area`: block and multiplexer prices recomputed from `library.json`; their sum is the area; the
     critical path recomputed is the stored one and within the bound. `bounds`: each family's derived
     bound recomputed from its units alone.
   - `proofs`: every query regenerated byte for byte from the stored unit, configuration and datapath
     (or array), and the widening lemmas from the widened bindings; every result is the configured solver's `unsat` for that text (Z3 for composed steps);
     the other solver agrees on every configuration query, Z3 again on composed steps.
   - `solver_sanity`: up to eight configurations with one select moved to another source; each solver
     refutes at least one (a move can pick a source with the same value in that configuration).
   - `arrays`: each covered class is a template class with the array's dataflow, the host or a guest of
     the host in `embedding.json`; composed pieces add up to the operand's width, one pass each.
   - `cliques`: each stored solution is conflict-free, closed under arc ends, avoids the no-goods and has
     the stored weight; when claimed optimal, an independent branch and bound over the stored graph finds
     nothing heavier.
   - `trace`: per family, the units and area after the last step are the stored units'; the initial
     count is the family's.
   - `coverage`: recomputed from the datapaths and the stored configurations and arrays.
   - `requant` (`fixed` only): the rescaling chains found again in every datapath as `datapaths` wrote it,
     each recorded; a class is `fixed` exactly when its lemmas are proven; a fixed class's datapath in the
     units is the original with the converts and multiplies removed and the rounds as `rescale` (the
     renumbering re-derived independently), a kept class's is unchanged; every lemma regenerated byte for
     byte, its result the configured solver's `unsat` when recorded as proven, and the other solver agrees.
   - Negative tests (`tests/smoke/merge.rs`): a wrong select, a rebound node, a dropped configuration, an
     edited area, a missing multiplexer, a cycle, an edited clique objective, an edited trace and array
     pieces each fail their check; an edited requantization decision, lemma or lemma result fails
     `requant`, and a lemma out of time keeps its class on fp32.

7. **Resource behaviour.** Memory is small (the units are a few thousand blocks in total). Time is the
   agglomeration: pair bounds are O(n²) per family at the start and O(n) per step; each exact gain is a
   compatibility graph (vertex mappings restricted by labels, arc mappings by the labels of both ends)
   and a clique solve, bounded by `node_budget`, `timeout_s` and `max_compat_nodes`. Exact gains run in
   parallel batches of 32 (fixed, so the result does not depend on the thread count); families run in parallel. Proofs are one query per
   (unit, configuration, tuple) and per (array, class, tuple), milliseconds each except composed steps.
   With `fixed`, one tolerance lemma per integer type rescaled (one CPU: Bitwuzla 23 s at i8, 57 s at
   i32; Z3, in `verify`, 91 s at i8, 11 minutes at i32), solved once per solver version thanks to the
   shared answer cache.
   Measured on the model set at `full` (2026-11-05, iacoma227): 572 s in a debug build (pass about 430 s,
   `verify` about 140 s), 165 s in a release build (`verify` 69 s). About 390 s of the debug pass are 38
   slow evaluations, most of them one pair with 13 254 vertex mappings evaluated again after every merge
   into its growing host (about 10 s each); a bound that survives the host's growth would avoid them.

8. **Known limitations.**
   - Float prices come from `characterize`'s provisional `apfloat` rows until the float IP cores are
     integrated (design §4.12); the proofs already state the numerics of design §6.4 (f32 and bf16 flush
     to zero, FP8 conversions saturate), the prices do not.
   - Merging is pairwise and greedy (Moreano Sec. VI-A); the result depends on the merge order, which the
     lazy greedy fixes by gain. A clique answered by `greedy` or out of budget is not optimal (recorded).
     With `rank = "greedy"` the order follows greedy gains; the exact solve improves only the accepted pair.
   - `timeout_s` is wall-clock: a solve that reaches it makes the output machine-dependent (recorded as
     `fallback = "timeout"`); `node_budget` is the deterministic limit.
   - The cost model is area only: operator and multiplexer areas from the library, wires free, no
     throughput term. Units serving many classes are time-multiplexed; whether they keep up is a
     scheduling question for later passes.
   - Contraction restricts sharing: a block inside a pattern occurrence merges only with the same pattern.
   - Floats are not widened across formats; integer widening covers the operators listed in step 3. A
     bf16 operation on an f32 block would not be exact under design §6.4: the two formats flush at
     different thresholds (a bf16 product just below 2^-126 rounds up to it at 8 bits and is flushed at
     24), so such a widening would fail its lemma.
   - Prices of pending float operators are estimates (`estimated`), and array element prices use the
     multiply at the accumulator type (what the semantics computes), not a narrower multiplier.
   - Composed array steps are proven by Z3 alone, as encoding lemmas are.
   - `fixed` costs more area than `fp32` at ASAP7 (one `rescale` block per family that has the chains:
     +576 µm², 0.3 % of the model set's units); a cheaper form would round the integer or the scale's
     significand first, which changes the tolerance lemma.
   - `fixed` matches the chain `convert → multiply → round` only; a scale applied by a divide, or a chain
     whose convert or product has other readers, stays fp32. The zero point and the clip after the round
     stay f32 (the models pass the zero point as an f32 input). The tolerance lemma rests on the IEEE 754
     rounding bound for the fp32 side, not on the solver's f32 multiply (step 11).
