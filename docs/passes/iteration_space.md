# `iteration_space`

1. **Purpose.** For every kernel class Z of `kernel_classes`, the pass derives the iteration domain of its
   representative kernel, the affine access function of every value in it, the class *map*, *reduce* or
   *contraction* (from the accesses, not from op names), the dependences under the reduction order ω,
   and the embedding relation ⊑ between skeletons that `merge` uses. It then decides by proof whether a
   class is a uniform recurrence with a valid space-time mapping, and routes it either to a generated
   systolic array template or to datapath mining (design §4.7, §7). Position: `kernel_classes →
   iteration_space → datapaths`.

2. **Background.** Design §4.4 (ω), §4.5 (K1-K4), §4.6 (classes), §5, §7; the op-semantics table in
   docs/framework/ir.md; SMT obligations through `cc-smt` (docs/framework/smt.md).
   - *Iteration domains and affine accesses.* An op with `n` loop dims has the box domain
     `D = [0, E_1) × … × [0, E_n)`. Each dim is parallel (it indexes the result) or a reduction. Operand
     X is read at `F_X i + c_X`: `F_X` is an integer matrix whose entries may be size parameters
     (strides, dilations), `c_X` holds the offsets (padding). X is *reused* along `u` when `F_X u = 0`;
     for `u = e_d` that means X is not indexed by dim `d`. The index formulas are those of the StableHLO
     specification (https://openxla.org/stablehlo/spec: `dot_general`, `convolution`, `reduce`,
     `reduce_window`, `scatter`, `broadcast_in_dim`, the element-wise ops).
   - *Uniform recurrences.* A system `X(z) = f(X(z − d_1), …)` over `D` with constant dependence vectors
     `d_j` is a uniform recurrence (Karp, Miller, Winograd, "The organization of computations for
     uniform recurrence equations", J. ACM 14(3):563-590, 1967, DOI 10.1145/321406.321418). A
     contraction such as `C[m, n] = Σ_k A[m, k] B[k, n]` reads A at every n and B at every m (broadcasts).
     Localization replaces each broadcast by propagation along a reuse direction (Rajopadhye and
     Fujimoto, "Synthesizing systolic arrays from recurrence equations", Parallel Computing
     14(2):163-189, 1990; Quinton, INRIA RR-0216, 1983):
     `a(m, n, k) = a(m, n − 1, k)`, `b(m, n, k) = b(m − 1, n, k)`, `c(m, n, k) = c(m, n, k − 1) + a·b`,
     with `d_a = e_n`, `d_b = e_m`, `d_c = e_k`. The accumulation along k is sequential, which is the
     order contract `seq` of ω.
   - *Space-time mapping.* A schedule `t(z) = λ·z` is valid when `λ·d ≥ 1` for every dependence; a
     projection direction u with `λ·u ≠ 0` gives a conflict-free allocation (one PE per line `z + t·u`;
     two iterations on one line never share a time step). With an allocation matrix S (`S u = 0`, rank
     n − 1), dependence `d` becomes a link `S d` with `λ·d` registers (Quinton, "Automatic synthesis of
     systolic arrays from uniform recurrent equations", ISCA 1984, DOI 10.1145/773453.808184, Theorems 1
     and 5; Moldovan, Proc. IEEE 71(1):113-120, 1983; Lamport, "The parallel execution of DO loops",
     CACM 17(2):83-93, 1974, for the hyperplane schedule). An operand whose dependence has `S d = 0`
     stays in its PE: it is *stationary* (Chen, Emer, Sze, "Eyeriss", ISCA 2016, DOI
     10.1109/ISCA.2016.40, for the weight-, output- and input-stationary taxonomy).
   - *Systolic arrays* use each fetched item many times with local communication only (Kung and
     Leiserson, "Systolic arrays (for VLSI)", CMU-CS-79-103, 1978; Kung, "Why systolic architectures?",
     IEEE Computer 15(1):37-46, 1982, DOI 10.1109/MC.1982.1653825).
   - **Worked example: matmul.** Over `(m, n, k)` with `d_a = e_n`, `d_b = e_m`, `d_c = e_k`, the only
     schedule in `{-1, 0, 1}^3` is `λ = (1, 1, 1)` (each unit vector needs a positive entry). Ten
     projections satisfy `λ·u ≠ 0` and keep every link at distance at most 1; the three axis projections
     are the classic arrays:

     | u | S (PE coordinates) | `S d_a`, `S d_b`, `S d_c` | stationary | array |
     |---|---|---|---|---|
     | `e_m = (1, 0, 0)` | `(n, k)` | `(1, 0)`, `(0, 0)`, `(0, 1)` | B | weight-stationary: an `N × K` array holds the weights; A enters skewed along n, partial sums move along k and leave after `K` PEs (the TPU's matrix unit, Jouppi et al., ISCA 2017, DOI 10.1145/3079856.3080246) |
     | `e_k = (0, 0, 1)` | `(m, n)` | `(0, 1)`, `(1, 0)`, `(0, 0)` | C | output-stationary: an `M × N` array keeps one accumulator per output; A moves along n, B along m, each PE accumulates its k chain in order (the tiled MAC arrays of tensor-core style units) |
     | `e_n = (0, 1, 0)` | `(m, k)` | `(0, 0)`, `(1, 0)`, `(0, 1)` | A | input-stationary: an `M × K` array holds the activations; B moves along m, partial sums along k |

     Iteration `(m, n, k)` runs at time `m + n + k`, so `A[m, k]` is first used at time `m + k` (at
     n = 0). The weight-stationary array's
     result for `(m, n)` leaves the last PE of column n at time `m + n + K − 1`. The diagonal projections
     (`u = (1, 1, 0)`, `(1, 1, 1)`, …) give hexagonal arrays in the style of Kung and Leiserson, with
     more PEs than the axis projections for the same work and lower utilization. A tensor-core
     instruction that sums its `k0` products in a tree computes another order. It corresponds to ω =
     `tree`, which this pass does not map: under `seq`, the output-stationary array accumulates each
     output sequentially.

3. **Inputs and outputs.** Inputs: `build/.kernel_classes/out/kernel_classes/{classes,instances}/<z>.json`
   (the representative, its local classes and group kinds, the value columns and the instance rows) and
   `build/.kernels/out/kernels/kernels/<component>.postcard` (the representative's kernel graph).
   Outputs (`build/.iteration_space/out/iteration_space/`; schemas `docs/schemas/iteration_space_*`):
   - `skeletons/<z>.json` (`Skeleton`):
     - `dims`: per iteration dim its tag (`par`/`red`), its extent (`class` = an axis class of the
       instance table; `const`; `column` = a value column entry such as a window size; `runtime` = a
       shape-operand entry; `quot` = a class divided by a group count), and the core's role for it;
     - `omega`: ω on the dims;
     - `operands`: the core's operands and results, then the kernel interface (one `in<node>` per
       distinct access of an input, one `out<node>` per output), each with its axis and element
       classes, its access and its reuse (the dims that do not index it);
     - `accesses`: per member (once per distinct access), the access of its results and operand ports;
     - `class`, `affine`, the prologue and epilogue `stages`, the path of the access proof.

     A row of an access is `{terms: [{k, factors, dim}], divisor, padded, data}`, i.e.
     `(Σ k · Π factors · i_dim) / divisor`. A factor is a value column entry (`n<node>.window_strides`[i],
     `n<node>.padding`[2i], …), a shape-operand entry, or a dim's extent. `padded` marks an index that
     reads padding out of range, `data` one that also depends on data (scatter).
   - `dependences/<z>.json` (`Dependences`): `uniform` and the reason, the groups (in linearization
     order), dims in no group, the `space`, and the vectors (operand, `propagate` or `accumulate`,
     coordinates over the space).
   - `templates/<z>.json` (`TemplateFile`, template classes only): the space and its groups, every valid
     candidate (λ, u, S, links with `S d` and delays, stationary operand, dataflow, array dims as
     expressions of the group extents, weighted utilization and link bits, rank), `selected`, the
     families of the core inputs and result, and the path of the mapping proof.
   - `embedding.json` (`Embedding`, compact): one `Pair` per `guest ⊑ host`: φ (`dims`), π
     (`operands`, host operand indices), the row maps, and the host factors' values (`zero`, `one`, or
     `guest(factor)`).
   - `routing.json` (`Routing`): per class its route, class, core, a readable reason, the reuse numbers
     (contractions), the selected dataflow and the stages (templates), and its weighted workload.
   - `smt/access.<z>.smt2`, `smt/mapping.<z>.smt2` with `.result` files (`ProofResult`: query hash,
     solver, version, verdict).
   - `summary.json` (`Summary`) and `views/routing.md`, `views/iteration_space.md`.

4. **Algorithm** (`algo/`, per class in parallel).
   1. *Iteration semantics* (`cc-ir`, `ops/iter.rs`): the op table's `iteration` hook gives, for an op,
      its dims and, per operand and result, the affine map from the op's iteration vector to the element
      it reads or writes, parametric in size attributes and shape operands. Conventions: `dot_general`
      iterates over its result axes and then one reduction dim per contracting pair; a convolution over
      its result axes (with the output feature split into a group dim and an in-group feature when
      `feature_group_count` is not 1, as the spec's split and concatenation define it), its kernel
      spatial axes and its kernel input feature; `reduce` over its input axes; a windowed reduction
      over its result axes and one window dim per axis; `scatter` over its update axes; element-wise
      ops and broadcasts over their result axes.
   2. *Skeleton* (`skeleton.rs`). The core's dims are the domain (a map kernel's domain is its widest
      element-wise member's result). Accesses are composed outward to a fixpoint: a member whose result
      access is known gets its operands' accesses through its hook (backward: prologues, broadcasts);
      an element-wise member whose operand access is known gets the same result access (forward:
      epilogues). A broadcast absorbed into the kernel may feed both operands' prologues, so a member
      can have several accesses. Size-attribute entries that are the literals 0 or 1 are folded, and
      the others become value columns. Every row is kept in a canonical form (like terms combined,
      sorted). Class: no reduction dim, map; otherwise contraction when some core input has a zero
      column on a parallel dim and every access is affine, else reduce (design §4.7). Stages:
      per core input port, the prologue and broadcast members on its path; the epilogue and the other
      broadcasts.
   3. *Dependences* (`deps.rs`). Contraction dims are grouped by the core operands that index them:
      batch (lhs, rhs, result), M (lhs, result), N (rhs, result), K (lhs, rhs). K is linearized in ω's
      order: `seq` visits the reduced elements in ascending lexicographic order of ω's axes, so the
      accumulator chain is the unit step of the linearized K. M and N are linearized in result order.
      Batch dims carry no dependence and are outer loops. A dim in no group, or ω = `tree`, makes the
      class non-uniform. Over the space (the nonempty groups among M, N, K) the vectors are lhs along N,
      rhs along M and the accumulation along K. A convolution becomes its im2col view: its input is
      passed along N but its sliding-window reuse along `e_y − e_r` is not used. Reductions are one
      chain along their reduced dims; scatter-reduce is data-dependent.
   4. *Mapping search* (`mapping.rs`). Every λ ∈ {-1, 0, 1}^n with `λ·d ≥ 1` for all d, and every u ∈
      {-1, 0, 1}^n with a positive first nonzero entry and `λ·u ≠ 0`. S has rows `u_p e_j − u_j e_p`
      for `j ≠ p`, where p is u's first nonzero entry: `S u = 0`, and `[λ; S]` is nonsingular
      iff `λ·u ≠ 0`. The links `S d` must be nearest-neighbour (entries in {-1, 0, 1}). With n ≤ 3 this
      is 27 × 27 pairs at most. Scores, at each instance row's extents (E_g = product of the group's
      dims' extents), weighted by the row's trip-count-weighted workload: utilization
      `|D| / (#PE · T)`, where `#PE = Π E_i − Π (E_i − |u_i|)` (lines through the box) and
      `T = Σ |λ_i| (E_i − 1) + 1`; then link bits per PE, the widths of the operands whose links move.
      Candidates are ranked by utilization (descending), link bits (ascending), then (λ, u); all are
      kept, the first is `selected`. Utilization favours projecting along the longest group (decode
      with M = 1 never selects weight-stationary). The dataflow is named after the stationary operand:
      the result (output-stationary), an input whose extents depend on no input symbol in any row
      (weight-stationary), any other input (input-stationary), none (systolic).
   5. *Routing* (`route.rs`). Map and reduce classes go to datapaths. A contraction goes to a template
      when it is uniform, its space has at most `max_dims` dims, it has a valid mapping, and the rows
      where both inputs are reused at least `min_reuse` times hold at least `min_share` of its weighted
      workload. An input's reuse is the number of iterations that pass it on in the recurrence: lhs
      along N (`E(n)`), rhs along M (`E(m)`), which is what the array realizes (Kung 1982). For a dense
      GEMM this equals MACs per element. For a convolution it leaves out the sliding-window reuse the
      grouped recurrence does not exploit, so a depthwise convolution (`E(n)` = channel multiplier)
      does not count as reusing its input.
   6. *Embedding* (`embed.rs`). For every ordered pair of distinct skeletons of one class:
      backtracking over φ (guest dims to host dims of the same tag, injective; host dims that share an
      extent class only receive guest dims that share one; a host dim fixed to 1 shares no extent with
      a mapped one), then π (guest operands to host operands of the same direction, injective; the
      core operands of contractions and reductions, the interface of map kernels), then a row map
      per operand. Each guest row must equal its host row after substitution: host dims outside φ are
      index 0 with extent 1; a host factor on a mapped dim becomes 1 or the guest's factor; a factor
      only in constant terms (an offset) becomes 0; a host divisor becomes 1 or the guest's. Host rows
      left over must vanish. The first witness found is stored. A search is bounded at 200 000 steps
      (`EMBED_BUDGET`); a search over it counts as no embedding and is reported.
   7. *Proofs* (`spec.rs`, `proofs.rs`). `spec.rs` transcribes the spec's index formulas a second time,
      independently of the hooks. For the core, it checks the iteration convention (dims, tags,
      extents) and gives every operand's and result's index over the iteration vector. For every other
      member, it gives each operand's index from the member's stored result access (element-wise:
      equal; broadcast: `operand_index[d] = 0` for an expanded axis, else
      `result_index[broadcast_dimensions[d]]`). Every stored row is claimed equal to its specified row
      (numerator and divisor). Both sides are polynomials of degree at most 2 in each variable; a
      polynomial of degree at most `t_i` in `x_i` that vanishes on `S_1 × … × S_n` with `|S_i| > t_i`
      is zero (Alon, "Combinatorial Nullstellensatz", Combinatorics, Probability and Computing
      8:7-29, 1999, DOI 10.1017/S0963548398003411, Lemma 2.1). So the access obligation asserts the
      disjunction of the negated claims with every variable in `{0, …, 7}`, in 32-bit vectors where
      nothing wraps; `unsat` proves the claims for all integers. The mapping obligation states, for every
      candidate, `λ·d ≥ 1` and that `x ≠ 0 ∧ S x = 0 ∧ λ·x = 0` has no solution with `|x_i| < 2^20`
      (x is the difference of two box points). Every term there is a sum of at most three such values
      times ±1, so 32-bit arithmetic is exact. One solver batch runs all obligations (the configured
      solver, `[tools.smt]`).

   Complexity: skeletons are linear in the kernel; mapping search is constant (n ≤ 3); scoring and
   routing are linear in the rows; embedding is quadratic in the classes of one kind with a bounded
   search per pair.

   Model set at `full` (25 models, 2026-11-02): 831 classes. Every kernel_classes kind keeps its
   class (157 contractions, 105 reductions, 557 maps), except the scatter-reduce kind: 11 are reduce
   (data-dependent, not uniform) and 1 is a map (a scatter without scatter dims). 820 classes are
   uniform. Every contraction's space is `(m, n, k)`, and each has 10 valid mappings with λ = (1, 1, 1).
   **127 classes are routed to templates**: dot_general/float 79, dot_general/sint 27, convolution/sint
   15, convolution/float 6. They hold 99.96 % of the contraction workload. Selected dataflows:
   weight-stationary 58, input-stationary 39, output-stationary 30. The 30 contractions on the datapath
   route all fail the reuse threshold: decode GEMVs (M = 1) of the language models and of whisper,
   rnnt and mamba2, depthwise and short filter convolutions (evo2, mamba2, mobilenetv4), small heads.
   99 959 embedding pairs (map 86 606, contraction 7 840, reduce 5 513), none over budget; 958 proof
   files, all `unsat` by Bitwuzla and re-checked by Z3.

5. **Config** (`[passes.iteration_space]`): `views = true`. `[passes.iteration_space.template]`:
   `enabled = true` (false: every class goes to datapaths), `min_reuse = 8` (reuse of each input in a
   row that counts), `min_share = 0.5` (share of the weighted workload in such rows), `max_dims = 7`
   (bound on the mapping search's space; after grouping it has at most 3 dims). Resources: `threads = 8`,
   `memory_gb = 2`, `timeout_min = 30`. The solver and its runner are `[tools.smt]`.

6. **Verification** (`verify.rs`). The checks read the inputs and the written files and never call the
   hooks, the skeleton builder, the grouping, the search, the router or the embedding search:
   - `complete`: every class has a skeleton, dependences and exactly one routing entry; a template file
     exists exactly for the template classes.
   - `<z>:skeleton`: reuse lists, `affine` and the class recompute from the stored accesses; the core
     operands' accesses are the core's.
   - `<z>:access`: the obligation regenerates byte for byte from the stored skeleton and the
     representative kernel (`spec.rs`); its stored result is `unsat` from the configured solver for this
     query hash; the other solver also answers `unsat`.
   - `<z>:dependence`: the groups recompute from incidence and K respects ω's order; each propagation
     vector satisfies `F_X u = 0` (the operand is indexed by no dim of the group it moves along); the
     accumulation vector is K's unit vector; `uniform` follows from the groups and ω.
   - `<z>:template`: per candidate: entries in {-1, 0, 1}, `λ·d ≥ 1`, `λ·u ≠ 0`, `S u = 0`, `[λ; S]`
     nonsingular (its determinant), links `S d` on nearest neighbours with delay `λ·d`, the stationary
     operand and the dataflow name. Utilization and link bits recompute from the rows, the ranking and
     `selected` hold, no candidate repeats, and the mapping obligation regenerates and is `unsat` for
     both solvers.
   - `<z>:routing`: the reuse share recomputes from the rows and the groups. The route follows from the
     class, uniformity, `max_dims`, the share and the config: a uniform contraction always has the valid
     mapping λ = (1, …, 1), u = e_k. Template classes carry the selected dataflow and the skeleton's
     stages.
   - `embedding`: every certificate (φ, the extent rule, π, row maps, every guest row equal to its host
     row under the stored factor values, other host rows vanishing). The check has its own substitution.

   Every check has a negative test in `tests/smoke/iteration_space.rs` (a removed skeleton, edited
   reuse, an edited access row and an edited obligation, an edited dependence vector, an edited
   schedule and an edited utilization, an edited route, an edited certificate). Unit tests on
   hand-built kernels (`algo/tests.rs`) pin the decisions: matmul ⊑ convolution but not back, depthwise ⋢
   dense convolution, `x·W` and `x·Wᵀ` embed both ways, the three axis projections of a GEMM are output-,
   weight- and input-stationary with λ = (1, 1, 1), a decode GEMM fails the threshold, a block-scaled
   dot is a GEMM with the block index as batch, a tree ω is not uniform, shared tables give two dims of
   one extent (which embed in a general GEMM but not back), softmax's reduction and map are not
   templates (the reduction is uniform but nothing is reused), a scatter-add with data-dependent
   indices is never a template, and the hooks agree with `spec.rs` (a mutated stride does not).

7. **Resources.** Linear in classes and rows, plus the embedding search (quadratic in the classes of
   one kind). At `full` (2026-11-02, debug build, 8 threads, iacoma227): 37 s for pass and verification
   together, solver batches included, 0.5 GB peak RSS; outputs 30 MB (skeletons 15 MB, `embedding.json`
   12 MB, obligations 8 MB).

8. **Limitations.**
   - Only uniform recurrences are mapped. Diagonal sliding-window reuse (a convolution's input along
     `e_y − e_r`) is not used, so convolutions are mapped as im2col and depthwise convolutions are not
     templates. Row-stationary dataflows (Eyeriss) are not a single linear projection and are not
     generated.
   - The reuse threshold and the scores use the untiled array (array dims as functions of the extents).
     Tiling to a fixed physical array is left to the later passes.
   - Block-scaled GEMMs (FP8 blocks, int4 per 32): the per-block GEMM is a template with the block index
     as an outer batch dim, the scale multiply is an epilogue stage, and the sum over blocks is its own
     reduction class. Accumulating across blocks inside the array, with the scale between the partial
     and the final accumulation, would need a cross-kernel fusion.
   - `dynamic_reduce_window` and `dynamic_conv` read windows or padding from shape operands: their maps
     are parametric in these run-time values (`runtime` factors and extents). Their workload in the
     instance table counts the result only.
   - ⊑ is a relation on skeletons. It does not compare datapaths (that is `merge`'s compatibility
     graph), and ω tree-vs-seq compatibility is not part of it.
   - `batch_group_count` other than 1 is not supported (no class at `full` has one).
