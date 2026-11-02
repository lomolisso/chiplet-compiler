# `kernel_classes`

1. **Purpose.** Groups the kernels of every component of every model into canonical *kernel classes* Z.
   Two kernels are one class when they are the same computation up to the order of each value's axes,
   the extents of the axes, the widths and formats of the element types and the values of constants:
   identity is structure plus dtype family (design §4.6, §6.1). Per class the pass writes an instance
   table T_z (one row per kernel instance: model, phase, component instance, extents, element types,
   values, trip-count-weighted workload), a certificate per member kernel, and the sharing report that
   compares sharing at component and at kernel level. Position: `kernels → kernel_classes →
   iteration_space`.

2. **Background.** Design §4.6, §5, §6.1; the kernel graphs in docs/passes/kernels.md; the classes of
   components in docs/passes/components.md §4.3; the op-semantics table in docs/framework/ir.md.
   - Isomorphism of labelled directed graphs, decided by VF2 (Cordella, Foggia, Sansone, Vento, "A
     (sub)graph isomorphism algorithm for matching large graphs", IEEE TPAMI 26(10):1367-1372, 2004, DOI
     10.1109/TPAMI.2004.75) after bucketing by 1-WL colour refinement (Weisfeiler and Leman 1968;
     Shervashidze et al., "Weisfeiler-Lehman graph kernels", JMLR 12:2539-2561, 2011). Equal hashes prove
     nothing (Cai, Fürer, Immerman, Combinatorica 12(4):389-410, 1992), so every member is confirmed by an
     isomorphism, which is stored as its certificate. The WL and VF2 code is `cc_ir::absgraph`, shared
     with `components`.
   - Graphs with extra structure (here: which axes a value has, which a dot contracts) are reduced to
     plain labelled graphs by vertex colouring and edge subdivision, the encoding nauty and Traces use for
     coloured and directed graphs (McKay and Piperno, "Practical graph isomorphism, II", J. Symbolic
     Computation 60:94-112, 2014, DOI 10.1016/j.jsc.2013.09.003).
   - Translation validation (Pnueli, Siegel, Singerman 1998; Necula 2000): the certificates and rows are
     re-checked without the pass's encoder or classifier.

3. **Inputs and outputs.** Inputs: `build/.kernels/out/kernels/` (partitions and kernel graphs),
   `build/.normalize/out/normalize/` (normalized graphs, to see which inputs are composite calls; ledgers),
   `build/.components/out/components/` (instance records with their bindings), and per phase
   `build/.export/out/export/<model>/<phase>/{symbols.json, census_concrete.json}` (symbol sizes at the
   config, trip count, the concrete census). Outputs (`build/.kernel_classes/out/kernel_classes/`; schemas
   `docs/schemas/kernel_classes_*`):
   - `classes/<z>.json` (`ClassFile`): id (`z` + 16 hex digits of the WL hash, `-k` for the `k`-th class
     of a bucket), kind (`contraction`, `reduction`, `scatter_reduce`, `map`), core key (`<op>/<family>`),
     tags, the representative (component, kernel, group kinds) with its roles and local classes, the
     parameters (axis classes and whether each is an expanded unit axis; element classes with their
     family; value columns `n<node>.<attribute>` and `n<node>.splat<j>`), the iteration axes, the identity
     graph, the members, the models using the class, its instances and total weighted workload.
   - `instances/<z>.json` (`InstancesTable`, compact JSON): ω's order and the rows. A row: model, phase,
     component instance, component, kernel, member index; per axis class its extent at the config and its
     symbolic size; per element class its type; per value column its value (`null` where the member has
     none); workload, trip count, `weighted = workload × trip_count`.
   - `certificates/<z>.json` (`CertificatesFile`): per member kernel, its node map, axis-class map and
     element-class map from the representative, and its own local classes. One file per class rather than
     one per instance: every instance of a member kernel has the same map (the rows point to their member).
   - `sharing.json` (`Sharing`) and `views/sharing.md`; `summary.json` (`ClassesSummary`) and
     `views/classes.md` (classes per core, per number of models, the 40 with the most workload).

4. **Algorithm** (`algo/`).
   1. *Kernels* (`algo/mod.rs`, per component in parallel). Each kernel graph gets its local classes (step
      2), the first slot of every class (to read extents and types), its convolution nodes, its kind, core
      key and ω, its iteration axes (step 5), and its tags.
   2. *Local classes* (`algo/local.rs`). A kernel's nodes keep their component's class ids, but ops outside
      the kernel can join two component classes, so those ids are not a function of the kernel. The
      classes are recomputed on the kernel graph alone with the op table's rules, the way `components`
      computes a component's classes, with one difference: the literal 1 is an extent like any other (a
      unit axis is a parameter), except for an operand axis that a `broadcast_in_dim` (static or dynamic)
      expands, which the rules keep apart. Inputs relate nothing; an output has no value of its own.
   3. *Rows* (`algo/mod.rs`, per phase in parallel). For every component instance with bindings and every
      kernel of its component: the group kinds of its convolutions (`feature_group_count` 1, equal to the
      input's feature extent (depthwise) or anything else (grouped); `batch_group_count` 1 or not), every
      class's extent (its first slot's component dim through the bindings, evaluated at the phase's symbol
      sizes with `SymExpr::eval`), every element class's type, the values (every entry of every size
      attribute through `instantiate`, `exponent_bits`/`mantissa_bits`, every folded constant's value) and
      the workload. A kernel is a *member* once per group-kind signature its instances show.
   4. *Identity graph* (`algo/encode.rs`). The member's kernel nodes, then one `axis` node per local dim
      class, one `elem` node per element class (labelled by family), then role and window nodes. An op node
      is labelled by its op (or region argument), its identity attributes (table below), result families,
      ω's order, folded operands (port and family) and group kinds; inputs by their family; outputs plain.
      Kernel edges keep their labels. Every statement about an axis is a role node between the op and the
      class: `axis` (result `r` has it), `elem`, `batch`, `contract`, `reduced`, `expand`, the convolution's
      `in_batch`/`in_feature`/`kernel_in`/`kernel_out`/`out_batch`/`out_feature`, `omega{k}` (the `k`-th
      axis of ω), the scatter lists, and `window` nodes that tie the input, kernel and output axes of one
      spatial dimension (convolution) or the operand and result axes of one window axis (`reduce_window`).
      Axis positions are gone, so dimension numbers and layouts canonicalize by construction and an
      isomorphism maps classes to classes. When two axes of one value share a class (e.g. `Q·Kᵀ` with
      rotary tables shared by Q and K), positions cannot be recovered from classes, and the kernel's role
      nodes carry their axis position (`positional`, a stricter identity).

      | Attribute | Identity or parameter | Why |
      |---|---|---|
      | op name and target (`round_nearest_even` vs `_afz`, custom-call target) | identity | another function |
      | element family | identity | design §6.1 |
      | width, float format | parameter (element classes) | cross-width reuse (design §6.2) |
      | dot dimension numbers, convolution layout, reduce `dimensions`, `broadcast_dimensions` | roles on classes | positions are layout |
      | `comparison_direction` | identity, `GT`/`GE` written as `LT`/`LE` with the operands exchanged | `a > b ⇔ b < a`, NaN included (IEEE 754-2019 §5.11) |
      | `compare_type` and other literal attributes | identity | another comparator or function |
      | ω | identity (order and axes) | tree vs chain; float results depend on it |
      | `feature_group_count`, `batch_group_count` | identity as {1, depthwise, grouped}, per instance | the reuse structure |
      | strides, dilations, window sizes (1 included), padding | parameters (value columns) | affine access, same datapath |
      | unit axes, except a broadcast's expanded operand axis | parameters (extent 1) | an extent; decode (sequence 1) and prefill share classes |
      | folded constants, `exponent_bits`, `mantissa_bits` | parameters | configuration values |
      | `precision_config`, `indices_are_sorted`, `unique_indices` | dropped | hints |
      | member roles | not in the identity, stored with the class | they follow from the structure |

   5. *Iteration axes and workload*. Workload = |iteration domain| of one execution of the phase: the
      product of the extents of the iteration classes (with multiplicity), times the window sizes of a
      `reduce_window`. Contractions: the result's axes and the contracting axes (`dot_general`), or the
      result's axes, the kernel's spatial axes and its input-feature axis (convolution, grouped included):
      the number of MACs, which is half the census's FLOPs. Reductions: the operand's axes (combiner
      applications). Scatter-reduce: the updates' axes. Map kernels: the domain (points, not ops).
      `dynamic_reduce_window` reads its window from operands, so its workload counts the result only.
   6. *Classification* (`algo/classify.rs`). 1-WL on every identity graph (the initial colour is the
      postcard bytes of the node label; rounds until stable, at most `wl_max_rounds`), buckets by hash,
      and in each bucket every member, in member-key order, is matched with VF2 against the bucket's
      representatives (node predicate: equal colour and equal label; edges: equal labels; `vf2_budget`
      candidate pairs). The first match gives its class and map, else it is a new representative.
   7. *Tags*. `size_independent`: no row's extent depends on an input symbol (constant work per
      invocation: parameter arithmetic such as scale products, foldable at deployment); `low_rank`: a map
      kernel whose domain has rank at most 1; `call_fed`: members reading a composite call's result (an
      activation in the parent module of its GEMM); `positional`.
   8. *Sharing* (`algo/sharing.rs`). Per model, the set of components with at least one kernel instance
      and the set of kernel classes. For each level: items, items used by at least two models, the
      pairwise Jaccard index matrix and its mean over all model pairs, and the share of each model's
      weighted workload that lies in items used by at least two models (`coverage`), over all models
      together (`coverage_total`, dominated by the heaviest models) and as the mean of the per-model shares
      (`coverage_mean`). A component's workload is the sum of its kernels'. Two variants: `all`, and
      `activation` (rows of `size_independent` classes left out). The answer to "is kernel-level sharing
      greater than component-level sharing?" is `kernel_sharing_exceeds_component_sharing`: on
      `activation`, both the mean Jaccard and `coverage_total` are higher at kernel level.

   Complexity: local classes and encoding are linear in the kernel; WL is O(rounds · (V + E) log E);
   VF2 is exponential in the worst case but close to linear with stable colours as labels on these graphs.
   The rows are O(instances × kernel nodes).

   Model set at `full` (25 models, 78 phases, 2026-11-02): 4 956 kernels (no kernel's instances differ in
   group kind) give **831 classes**: 157 contraction, 105 reduction, 12 scatter-reduce, 557 map; 136 665
   rows. 127 classes are size-independent, 118 low-rank, 108 have call-fed members, 14 are positional; no
   WL bucket held two classes. 188 classes are used by two models or more (643 by one; the most shared, an
   element-wise map, by 19).

   | core | classes | kernels |
   |---|---:|---:|
   | `dot_general/float` | 94 | 232 |
   | `dot_general/sint` | 33 | 292 |
   | `convolution/sint` | 21 | 43 |
   | `convolution/float` | 9 | 18 |
   | `reduce/float` | 73 | 308 |
   | `reduce/sint` | 10 | 23 |
   | `reduce/bool` | 9 | 148 |
   | `reduce_window/float`, `/sint` | 4, 4 | 4, 9 |
   | `dynamic_reduce_window/float`, `/sint` | 1, 4 | 1, 5 |
   | `scatter/float`, `/sint` | 8, 4 | 28, 178 |
   | map | 557 | 3 667 |

   Sharing (`views/sharing.md`):

   | variant | level | items | used by ≥ 2 models | mean Jaccard | workload in shared items | per-model mean |
   |---|---|---:|---:|---:|---:|---:|
   | activation | components | 548 | 60 | 0.010 | 18.8 % | 36.9 % |
   | activation | kernel classes | 704 | 151 | 0.041 | 27.6 % | 53.1 % |
   | all | components | 569 | 64 | 0.009 | 18.8 % | 37.3 % |
   | all | kernel classes | 831 | 188 | 0.057 | 27.6 % | 53.4 % |

   Kernel-level sharing exceeds component-level sharing: about four times the mean Jaccard index, and a
   larger share of the workload in shared items, overall and per model. Examples: one int8 convolution
   class with its dequantization epilogue (convert to f32, scale, bias) serves resnet50, yolo11, rtdetr_v2
   and pointpainting (398 instances), whose components differ; ptv3's components share nothing with other models, but 92 % of its
   workload is in classes another model uses.

5. **Config** (`[passes.kernel_classes]`): `wl_max_rounds = 64` (upper bound on refinement rounds),
   `vf2_budget = 200000000` (candidate pairs one VF2 search may test; a search over it fails the pass with
   the bucket named), `views = true`. Resources: `threads = 16`, `memory_gb = 4`, `timeout_min = 30`.

6. **Verification** (`verify.rs`; it reads the inputs and the written files and never calls the encoder
   or the classifier: local classes are recomputed with its own union-find over the op table's rules, and
   members are compared through a position-free description of each node written there):
   - `complete`: every kernel of every partition is a member of a class, no member key is in two classes,
     every component instance has one row per kernel of its component, and no row is repeated.
   - `ids`: class ids are unique, name their files and carry their hash.
   - `<z>:certificate`: the stored local classes of the representative and of every member are those the
     rules give; the axis and element maps are bijections; and `check_isomorphism` (O(V + E)) accepts the
     node map with the predicate "equal description under the class maps" and equal edge labels (with the
     operands of `GT`/`GE` comparisons exchanged). A description is the op and its identity attributes,
     result families, ω's order and the classes of ω's axes in order, the classes of every result, of the
     batch, contracted, reduced and expanded axes, of the convolution's roles, of every window, and of the
     scatter lists, folded operands and group kinds. A wrong node map, axis map, element map or local
     class fails it.
   - `<z>:rows`: every row's extents and element types recompute from its instance's bindings through
     the certificate and the phase's symbol sizes; the workload recomputes from the extents and the window
     columns; trip count and weighted workload match the phase.
   - `workload:<model>/<phase>`: 2 × (Σ workload of the contraction rows + the MACs of GEMMs and
     convolutions that `normalize` removed, read from its ledger) equals `census_concrete.json`'s FLOPs
     exactly. At `full` all 78 phases match, with nothing removed.
   - `sharing`: the per-model sets, items, shared items, mean Jaccard, `coverage_total` and
     `coverage_mean` of both variants and levels recompute from the rows; the views equal the ones
     rendered from the written report and summary.
   Every check has a negative test in `tests/smoke/kernel_classes.rs` (a dropped row, a misfiled class,
   an edited node map, axis map and local class, an edited extent and workload, an edited census, an
   edited report and view). Unit tests on hand-built kernels (`algo/tests.rs`) pin the identity
   decisions: `x·W` and `x·Wᵀ`, NHWC and NCHW convolutions, 1×1 and 3×3 kernels, decode and prefill are
   one class each; integer and float GEMMs, depthwise and dense convolutions, an expanding and a plain
   broadcast, `GT(a, b)` and `LT(a, b)` differ; `GT(a, b)` is `LT(b, a)`; and every axis permutation of a
   GEMM's input, every node order and both operand orders of its residual add (432 kernels) are one class.

7. **Resources.** Linear in kernels and rows. At `full` (2026-11-02, debug build, 16 threads): the pass
   in 6 s and verification in 6 s, 1.1 GB peak RSS for both together; outputs 91 MB (instance tables 56
   MB compact, classes 27 MB with their identity graphs, certificates 8 MB).

8. **Limitations.**
   - Identity is exact structure: two kernels that differ by one epilogue op are two classes, and so
     are a GEMM and a convolution, or GEMMs with one and with two free axes on a side. Relating them is the
     embedding relation of `iteration_space` and the merging of `merge`.
   - Positional classes (14 at `full`) keep axis positions in their identity, which can split kernels
     that are permutations of each other.
   - Value columns hold size attributes in the member's own axis order; the certificate's maps give the
     permutation to the representative's classes.
   - `dynamic_reduce_window`'s workload counts its result only (its window sizes are operands).
   - The workload unit differs by kind (MACs, combiner applications, points), so workload coverage is
     dominated by contractions.
   - A kernel that no component instance executes is classified with no group kinds and has no rows.
