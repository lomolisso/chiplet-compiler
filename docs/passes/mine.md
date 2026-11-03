# `mine`

1. **Purpose.** The pass finds the datapath fragments that recur across kernel classes, weighted by the
   work they cover, as candidate macro-blocks for the functional units `merge` builds. Per family of
   datapaths it mines frequent connected patterns with GraMi, re-finds every occurrence itself, keeps the
   convex ones, counts how many non-overlapping occurrences each class has (a maximum independent set)
   and ranks the patterns by s(p) = Σ_z workload_z · MIS(p, z). Position: `datapaths → mine → merge`.

2. **Background.**
   - Design §4.10; the datapaths (docs/passes/datapaths.md), the skeletons and routing
     (docs/passes/iteration_space.md), the instance tables and workloads (docs/passes/kernel_classes.md);
     GraMi's formats and invocation (docs/tools/grami.md).
   - Frequent subgraph mining in a single graph: a pattern is frequent when its support reaches τ. Support
     must be anti-monotone (a larger pattern never has more) so that infrequent patterns need not be
     extended. Counting occurrences is not anti-monotone: one `multiply` feeding three `add`s is one
     occurrence of `multiply` but three of `multiply -> add`.
   - **MNI** (minimum image based support; Bringmann and Nijssen, "What is frequent in a single graph?",
     PAKDD 2008, LNCS 5012:858-863): the minimum over the pattern's vertices of the number of distinct
     graph vertices it is mapped to by some embedding. Anti-monotone and computable without enumerating
     all embeddings; GraMi (Elseidy, Abdelhamid, Skiadopoulos, Kalnis, PVLDB 7(7):517-528, 2014) decides
     MNI ≥ τ as a constraint satisfaction problem. MNI counts overlapping occurrences.
   - **MIS support** (Vanetik, Gudes, Shimony, ICDM 2002): the size of a maximum independent set of the
     overlap graph (one vertex per occurrence, an edge when two occurrences share a graph vertex), i.e. the
     number of occurrences that can run at the same time on separate hardware. MIS ≤ MNI. APEX (Melchert
     et al., ASPLOS 2023) mines CGRA datapaths with GraMi and ranks by this count.
   - **Example (APEX's 4-tap chain).** `y = (((x0·w0 + x1·w1) + x2·w2) + x3·w3) + c`, products
     `p0..p3`, sums `s1 = p0 + p1`, `s2 = s1 + p2`, `s3 = s2 + p3`, `y = s3 + c`. The pattern
     `multiply -> add -> add` has four occurrences `{p0, s1, s2}`, `{p1, s1, s2}`, `{p2, s2, s3}`,
     `{p3, s3, y}`. The second `add` maps to `s2`, `s3`, `y`, so MNI = 3. The occurrences overlap
     (the first three share `s2`, the last two `s3`); at most two are disjoint (`{p0, s1, s2}` and
     `{p3, s3, y}`), so MIS = 2: two units shaped like the pattern can be busy at once, never three.
     The smoke test `mni_and_mis_differ_on_the_chain` checks exactly this.
   - Convexity (Atasu, Pozzi, Ienne, DAC 2003; Pozzi, Atasu, Ienne, IEEE TCAD 25(7), 2006): a vertex set
     is convex when no path leaves it and comes back; only a convex set can run as one unit, because an
     outside value on such a path would be needed in the middle of it.
   - Maximum independent set is NP-hard (Karp 1972, as the complement of clique) and not approximable
     within n^(1-ε) (Håstad 1999; Zuckerman 2007); the overlap graphs here are per class and small.

3. **Inputs and outputs.**
   - Inputs: `datapaths` (`datapaths/<z>.json`), `iteration_space` (`skeletons/<z>.json` of template
     classes), `kernel_classes` (`instances/<z>.json`).
   - **Families and units.** A datapath class (route `datapath`) is one unit, in the family of its
     skeleton class: `map`, `reduce` or `contraction`. A template class gives one unit per stage
     (`<z>/prologue<port>`, `<z>/epilogue`) in the family `stage`. A unit's graph has one node per
     non-constant vertex and one edge per pair of vertices with dataflow between them; constants, inputs,
     accumulators and core results are not nodes, their uses are pattern inputs.
   - **Workload** of a unit: the class's weighted workload (Σ over rows of |iteration domain| × trip
     count); for a stage, Σ over rows of trip count × the elements of the core operand it feeds
     (prologue) or of the core result (epilogue), from the skeleton operand's axis classes.
   - Outputs (`out/mine/`; schemas `docs/schemas/mine_*`):
     - `labels.json` (`Labels`): node label `i` is `nodes[i]`, the vertex's `op@family` with the
       attributes that change what it computes in brackets (`compare@sint[LT,SIGNED]`); edge label `j` is
       `edges[j]`, the merged ends (result index, port; `*` for a commutative op) of the parallel edges
       between two vertices (`x * x` is one edge with two ends). Both lists are sorted.
     - `lg/<family>.lg`: GraMi's input, the disjoint union of the family's units (vertices numbered unit
       by unit); `lg/<family>.map.json` (`LgMap`): per unit its id, class, stage, workload, first vertex
       and datapath vertices.
     - `grami/<family>.json` (`GramiFile`): the final τ, GraMi's pattern count, the patterns over
       `max_vertices` (counted only; none with GraMi's node bound) and the others by canonical id.
     - `patterns/<family>/<p>.json` (`PatternFile`): the canonical code and id, vertex labels and edges,
       τ, MNI, the number of embeddings and of non-convex images, the convex occurrences (unit, the
       datapath vertex of each pattern vertex, and every operand fed from outside: `const`, `input`,
       `acc`, `result` or another `vertex`), per unit the MIS (its occurrences, the chosen ones, size,
       whether optimal, an upper bound, the algorithm, the unit's workload), and the score.
     - `selection/<family>.json` (`Selection`): the top-k patterns by score.
     - `sweep/<family>.json` (`Sweep`): every τ tried (finished or not, GraMi's count, the kept count,
       the selection's hash) and why the sweep stopped.
     - `overlap.json` (`Overlap`): kept patterns found in several families, and where they are selected.
     - `summary.json` (`Summary`); `views/mine.md`.
   - `cache/grami/<key>/tau<τ>/` (outside `out/`): GraMi's job directory (`graph.lg`, `Output.txt`, a
     `done` marker with the wall time), keyed by the graph, the GraMi commit, the image and the label
     node bound. GraMi's timing never reaches `out/`, so outputs are byte-identical across runs.

4. **Algorithm.**
   1. *Units and labels* (`algo/family.rs`): units, their graphs and workloads, the label tables, one
      `.lg` per family.
   2. *Sweep* (`algo/sweep.rs`), per family, families in parallel as memory admits containers: GraMi at
      τ = `tau_start`, then τ halved (rounded down, at least `tau_min`) until `tau_min`.
      - *GraMi* (`algo/grami.rs`): the directed subgraph build of the image, `limitedtime=false` (its
        default `true` adds a candidate to a domain when a 5 s timer fires, so it can report infrequent
        patterns), `maxNodes = max_vertices` (our patch `third_party/patches/grami/0002`: a larger pattern
        counts as infrequent, so it is neither reported nor extended; the node count only grows along an
        extension, so every pattern within the bound is found with its support), one CPU, `heap_gb` of
        heap, in a container of `heap_gb + 2` GB, within `run_timeout_s` and what is left of `budget_s`.
      - *Parse and filter* (`algo/lg.rs`): `Output.txt` strictly (its count must match); a pattern over
        `max_vertices` (none with the bound) would be counted as dropped; the others are canonicalized
        (`algo/canon.rs`: among the vertex orders that sort the labels, the one with the smallest sorted
        edge list; id `p` + 16 hex digits of the code's hash).
      - *Re-match* (`algo/rematch.rs`), each new pattern once (results carry over to lower τ): every
        embedding in every unit by VF2 in its non-induced mode (`cc_ir::absgraph::monomorphisms`: labels
        and edge ends equal, extra edges between the images allowed, which is GraMi's semantics), MNI over
        the family from all embeddings, one occurrence per image (the first map VF2 finds), kept if convex
        (no vertex outside the image is both a descendant and an ancestor of it).
      - *MIS* (`algo/mis.rs`), per unit with occurrences: the overlap graph, split into connected
        components (isolated occurrences are all taken), solved by the configured strategy:
        `exact_ilp`: max Σ x_o s.t. Σ_{o ∋ h} x_o ≤ 1 for every datapath vertex h (one clique constraint
        per vertex), binary x, by microlp through good_lp; `branch_and_bound`: branch and reduce (take
        vertices of degree 0 or 1, then branch on a vertex of maximum degree; Tarjan and Trojanowski,
        SIAM J. Comput. 6(3):537-546, 1977), with `node_budget` per component (over it, the incumbent
        is reported with bound = component size); `greedy_min_degree`: repeatedly take a vertex of
        minimum degree and delete its neighbourhood, with the ratio (Δ+2)/3 of Halldórsson and
        Radhakrishnan (Algorithmica 18(1):145-163, 1997) reported as the bound (optimal when it equals
        the size).
      - *Score and select* (`algo/sweep.rs`): s(p) = Σ_units workload · MIS size; the top `top_k` with a
        positive score, ties by more vertices, then id.
      - *Stop* when the selection is non-empty and equal to the previous τ's (`stable`), at `tau_min`
        (`tau_min`), or when a GraMi run does not finish in time (`budget`: the last finished τ is kept);
        a family without edges is `empty`.
   3. *Write* the patterns of the final τ, GraMi's list, the selection, the sweep, the overlap, the
      summary and the view.

   Cost: GraMi's search is exponential in the pattern size and grows quickly as τ falls (§7); VF2 per
   (pattern, unit) is small (patterns ≤ `max_vertices`, units of tens of vertices); the MIS components
   are small.

5. **Config** (`[passes.mine]`): `tau_start = 40`, `tau_min = 2`, `max_vertices = 6`, `top_k = 32`,
   `budget_s = 3600` (GraMi wall clock per family), `run_timeout_s = 600`, `heap_gb = 6`,
   `vf2_budget = 100000000` (candidate pairs of one re-match; over it the pass fails), `views = true`;
   `[passes.mine.mis]`: `algorithm = "exact_ilp" | "branch_and_bound" | "greedy_min_degree"`,
   `node_budget = 1000000`. The image tag and the GraMi commit enter the cache key, and Java's version
   through `tools()`. Resources: `threads = 8`, `memory_gb = 8` (one GraMi container per `heap_gb + 2`
   GB runs at a time), `timeout_min = 120`.

6. **Verification** (`verify.rs`; it never calls the miner's code: units, the `.lg` reading, canonical
   codes, matching, convexity and the independent-set search are written again there):
   - `labels`: the label tables are exactly the units' node keys and merged edge ends.
   - `<f>:lg`: the `.lg` file and its map translate the units: every vertex's label and every edge's
     label equal those recomputed from the datapaths, nothing more; units, first vertices and workloads
     recompute.
   - `<f>:grami`: GraMi's list has the sweep's τ, its count is listed plus dropped, every listed pattern
     has a pattern file and the sizes are within bounds; when GraMi's `Output.txt` is in the cache it is
     re-read and canonicalized independently and must give the same list.
   - `<f>:patterns`: per pattern, the code is canonical (search over every label-sorting order) and the
     id its hash; a backtracking matcher enumerates every embedding in every unit and must give the same
     embedding count, the same MNI (≥ τ), the same number of non-convex images and exactly the stored
     occurrences (by a forward and backward reachability test); each stored occurrence is an embedding
     and its operand feeds recompute; every MIS is independent (a mark array over datapath vertices),
     its size and bound are consistent, an exact search gives α with `size = α` when optimal and
     `size ≤ α ≤ bound` otherwise; the score recomputes.
   - `<f>:selection`: the top-k recomputes from the pattern files.
   - `<f>:sweep`: the τ sequence is the configured one, the final τ is the last finished step, the stop
     reason holds and no earlier stable step was missed.
   - `overlap`, `summary`: recount.

   Tests: `tests/smoke/mine.rs` runs the pass on synthetic datapaths (multiply-add chains, a reduction,
   a non-convex diamond, compares, a template class) with a brute-force miner (`tests/src/brute.rs`) in
   GraMi's place, checks the APEX example (MNI 3, MIS 2), reproducibility, every MIS strategy, the
   re-matcher against the brute force on random graphs, and a negative test per check (an edited
   occurrence, a dependent MIS, a wrong non-convex count or MNI, an edited score, selection, `.lg`,
   label, sweep, GraMi list, summary). Unit tests cover the `.lg` and `Output.txt` formats, the
   canonical code under vertex renaming, the MIS strategies on fixture graphs (path, clique, star,
   cycle, two components, an interval graph, an edgeless graph), and non-induced matching. The deep
   tests run GraMi in the image against the brute force on random graphs (equal frequent sets at τ = 2,
   3) and the whole pass on `testing.deep_models` at `full` with a golden digest and a byte-identical
   forced rerun.

7. **Resources.** Model set at `full` (25 models, 830 classes; 2026-11-02, iacoma231, debug build, GraMi
   on one CPU per run). Units and graphs: map 557 units, 2 954 vertices, 2 851 edges; stage, reduce and
   contraction a few hundred vertices each. No family's selection was stable before `tau_min`, so every
   sweep ran to τ = 2.

   | `max_vertices` | pass + verify | GraMi (map) | patterns kept (map) | top-32 picks at the bound | peak RSS | outputs |
   |---|---|---|---|---|---|---|
   | 5 | 4.6 min | 225 s | 2 932 (2 674) | map 3, contraction 6 | 0.29 GB | 26 MB |
   | 6 (default) | 12.9 min | 610 s | 7 077 (6 788) | map 2, contraction 5 | 0.31 GB | 58 MB |
   | 8 | estimated 2 h or more | 13 + 468 + 628 s to τ = 10; τ = 5 over 16 min | 13 855 at τ = 10 | | | |

   The other families take under 4 s per GraMi run at any bound. Without the node bound (upstream
   GraMi) the map family did not finish τ = 20 in 16 minutes; with `maxLabelAppearance` alone neither.
   Most selected patterns have 2 to 4 vertices; the contraction family is the one that uses the 6th.

8. **Limitations.**
   - The sweep stops at the first τ whose selection equals the previous one. A pattern whose MNI is
     below that τ is not mined even if its score (workload-weighted MIS) would place it in the top-k:
     GraMi's output is monotone in τ, so only a lower τ could reveal it.
   - A sweep stopped by the budget depends on the machine's speed; the sweep file records it.
   - Upstream GraMi has no size bound; the pass relies on the image's patched build (`maxNodes`). Without
     it the `map` family does not finish below τ = 40 (§7).
   - Convexity is checked per unit: a template's stages are separate units, so a path through the array
     between a prologue and the epilogue is not seen (it cannot exist: the core sits between them).
   - Constants are inputs: `x · c` and `x · y` match the same pattern; the occurrence records which
     operands are constants, and `merge` decides about constant registers.
