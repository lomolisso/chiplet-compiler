# `components`

1. **Purpose.** Each executed call of a `cc.*` composite, and each maximal region of the entry's own
   ops, is an *instance*. Each instance is abstracted into a graph without sizes or widths, and equal
   graphs across all models are one *component*. Components are the component level of the sharing
   analysis and the input of `normalize`. Every instance stores the values that turn its component back
   into it, and a vertex map that proves it. Position: `import → components → normalize`.

2. **Background.** Design §4.3, §5, §6.1; the IR in docs/framework/ir.md.
   - A `stablehlo.composite` "can be replaced with its decomposition without changing program semantics"
     (spec `#composite`). The export wraps every module level in one (design §11), import keeps `cc.*`
     composites as ops with their decomposition functions (docs/passes/import.md), so the nesting of
     modules is the nesting of composites. JAX may give several calls one decomposition function
     (alphafold2, gemma4): an instance is a call *path* (the composite-call ops from the entry), not a
     function.
   - Isomorphism of labelled directed graphs; subgraph monomorphism (non-induced) and induced subgraph
     isomorphism (Cordella, Foggia, Sansone, Vento, "A (sub)graph isomorphism algorithm for matching
     large graphs", IEEE TPAMI 26(10):1367-1372, 2004, DOI 10.1109/TPAMI.2004.75). For two graphs with
     the same number of nodes, an injective label-preserving map is a bijection; when the edge counts
     are equal too, a monomorphism, an induced subgraph isomorphism and an isomorphism are the same map.
     The pass compares whole graphs, checks both counts first and runs VF2 in its isomorphism mode (the
     `vf2` crate, docs/tools/rust_stack.md).
   - 1-dimensional Weisfeiler-Leman colour refinement as an isomorphism-invariant hash (Weisfeiler and
     Leman 1968; Shervashidze, Schweitzer, van Leeuwen, Mehlhorn, Borgwardt, "Weisfeiler-Lehman graph
     kernels", JMLR 12:2539-2561, 2011). Different hashes prove two graphs non-isomorphic; equal hashes
     prove nothing (Cai, Fürer, Immerman, Combinatorica 12(4):389-410, 1992), so every equality is
     confirmed by VF2 and certified. The paper's algorithm is for undirected, node-labelled graphs; the
     directed, edge-labelled form used here (in- and out-neighbour multisets with their edge labels) is
     this project's extension.
   - Translation validation (Pnueli, Siegel, Singerman 1998; Necula 2000): the pass stores certificates
     (vertex maps, bindings) and its verification re-checks them without the pass's algorithm.

3. **Inputs and outputs.** Input: `build/.import/out/import/<model>/<phase>/ir.postcard`. Outputs
   (`build/.components/out/components/`; schemas `docs/schemas/components_*`):
   - `instances/<model>/<phase>.json` (`InstancesFile`): the module's op count, the entry's terminator
     (the one op in no instance), and per instance its id, kind (`composite`, `anon`), composite name,
     `composite_attributes` and `version`, position (call path, `cc_path`, parent and children), the
     decomposition function, the owned op ids as ranges, counts (owned ops, of which left out of the
     graph, graph nodes), its component and its bindings: one `Dim` per dim class (a constant or a
     symbolic expression), one element type (MLIR spelling) per element class, and per component node
     its parameters (size-attribute entries, a constant's value, a size value).
   - `components/<id>.postcard` (`StoredComponent`): the abstract graph, its height (0 without
     composite calls), its interface classes and class counts; `<id>.postcard.summary.json`
     (`ComponentSummary`): node and edge counts, ops per name, child components, instance names, models
     and phases, the instance list. `id` = `c` + the first 16 hex digits of the WL hash, with `-k` for
     the `k`-th isomorphism class of a hash bucket.
   - `certificates/<model>/<phase>.json` (`CertificatesFile`): per instance, the image of every
     component node (an op, a value, a `size_in` port, an exported value). One file per phase rather
     than per instance: the model set has about 25 k instances, and the calls of a shared decomposition
     have equal maps.
   - `views/components.md`: components × models with instance counts, sorted by the number of models.

4. **Algorithm.** Per phase in parallel (largest modules first), then once across phases.
   1. *Instances* (`algo/mod.rs`). A depth-first walk from the entry over composite calls gives every
      composite instance; its id is `<model>/<phase>/<o1>/<o2>/…` (the call ops). An instance owns its
      call op (at the parent's path) and its decomposition's ops (nested regions included) except the
      nested calls. The entry's other ops, without the composite calls and the terminator, are grouped
      by union-find: compute ops joined by def-use at non-shape ports, ops of nested regions with their
      owner. A group of size computation joins the region of its first compute consumer; a group with
      none is an instance without a graph. These regions are not split further: layout and data
      movement boundaries belong to `kernels` (design §4.5). Anon ids are `anon:<hash of the op ids>`.
   2. *Graphs* (`algo/graph.rs`), once per decomposition (children first) and per anon region. Nodes:
      `input k` per argument that is not a dimension variable (anon: one unordered `input` per external
      value and one `output` per value used outside), one node per op except size computation and ops
      whose every use is a shape operand (constant shape vectors), one per region argument, and one
      `size_in` per `dim_as_data` operand (the size value enters compute there; its symbolic value,
      recomputed by import's size-value interpreter, is a parameter). Shape operands and composite
      dimension arguments are not edges: import validated that the result types determine them. Edges
      carry (result index, port): `*` for a commutative op's operands, the index among non-shape
      operands for a composite call, `region r.i` from an owner to its region's arguments and `yield r`
      from a region's terminator to its owner; parallel edges merge into one edge with the sorted list.
      Labels: op name and target; attributes, where a composite call keeps none (its child component
      is its label) and a constant keeps only `splat`; every entry of a size attribute (the op table's
      `size_attrs`: slice bounds and strides, padding, convolution and window strides, dilations, group
      counts, window and slice sizes, FFT lengths) is a parameter unless it is 0 or 1; per result, the
      element family (`bool`, `sint`, `uint`, `float`, `complex`) and which axes are the literal 1.
   3. *Classes* (`algo/classes.rs`). Union-find over slots (one per axis of every result, one per result
      for its element type): the op table's `dims` rules (docs/framework/ir.md, `ops/dims.rs`) relate
      operand and result axes the spec forces equal, its element rules relate element types, and a
      composite call relates its operands and results as its child component's interface does. Classes
      are numbered by first occurrence; the literal 1 is never a variable. A class is a function of the
      labelled graph, so any isomorphism maps classes to classes. *Bindings*: every slot of a class must
      carry one dim (equal, or provably equal under the scope's bounds) and one element type, else the
      pass fails (a wrong rule cannot pass silently).
   4. *WL hash*: initial colour = the node's label key and, for a call node, the child's hash; rounds
      until the number of colours stops growing (at most `wl_max_rounds`); the hash covers the sorted
      colours and the node and edge counts. O(rounds · (V + E) log E).
   5. *Deduplication* (`algo/dedup.rs`), by height so that children have their component first: graphs
      are bucketed by WL hash; each graph is matched with VF2 (isomorphism mode, after the count check)
      against the bucket's representatives in order, with node predicate = equal stable colours, equal
      label keys and equal child components, and edge predicate = equal labels. The first match assigns
      the component and its node map; no match makes a new representative. A member's bindings are
      renumbered into the component's classes through the map. VF2 is exponential in the worst case;
      with stable WL colours as labels it is close to linear on these graphs; the crate scans
      candidates linearly, so a search costs O(V²) node scans (graphs here have at most 2.3 k nodes).
   6. *Write* instances, certificates, components with summaries, and the view.

   Model set at `full` (78 phases, 2026-11-01): 22 557 instances (339 anon, 141 without compute ops)
   give 712 components; 73 are used by two or more models; heights 0 to 8; the largest component has
   2.3 k nodes (`cc.flax.Decoder`); no WL bucket held two isomorphism classes.

5. **Config** (`[passes.components]`): `wl_max_rounds = 64` (upper bound on refinement rounds),
   `vf2_budget = 200000000` (candidate pairs one VF2 search may test; a search over it fails the pass
   with the graph named), `views = true`. Resources: `threads = 8`, `memory_gb = 4`, `timeout_min = 30`.

6. **Verification** (`verify.rs`; it reads the modules and the written files; it never calls the
   graph builder, the class computation for instances, WL or VF2). Per phase, `<model>:<phase>:<check>`:
   - `partition`: an independent walk of the composite expansion visits every op at every executed
     call path; each occurrence must be owned exactly once (an instance's ops at its path, its call op
     at its parent's path, the entry's terminator), no owned occurrence may be outside the expansion,
     and the number of occurrences equals `Module::expanded_multiset`.
   - `recoverability`: the instance's concrete graph is rebuilt from the module by this check's own
     code (nodes keyed by their module identity, edges with their labels). The component instantiated
     with the instance's bindings (`algo::graph::instantiate`) must equal it under the stored
     certificate: the certificate is a bijection onto the concrete nodes, every node's op, attributes,
     result types (dims compared exactly or provably equal), child component and size value are
     equal, every component edge maps to an edge with an equal label, and the edge counts are equal.
     O(V + E) per instance, no search.
   Once:
   - `complete`: every imported phase has its instances and certificates files.
   - `classes`: every component's classes and interface recompute from its graph and its children's
     interfaces.
   - `wl_hash`: every component's WL hash recomputes from its graph and its children's hashes, and its
     id carries that hash.
   - `dedup`: every instance's component exists and lists it, and every listed instance points to it.
     Since every member is checked against the one stored graph, the map between two members (one
     certificate composed with the inverse of the other) is an isomorphism.
   - `sharing_table`: `views/components.md` equals the table recomputed from the summaries.
   Every check has a negative test in `tests/smoke/components.rs` (a dropped op, a swapped
   non-commutative port, a corrupted binding, an edited component, a wrong component, an edited view),
   which also checks that component ids do not change when the ops are reordered, the values renumbered
   or commutative operands swapped, that five Dense calls of different widths are one component, and that
   a shared decomposition gives one instance per call.

7. **Resources.** Memory is the modules in flight plus every phase's abstract graphs, which are kept
   for deduplication. At `full` with 8 phases in parallel, run and verification together peak at
   3.3 GB, verification alone at 2.2 GB, mobilenetv4_hybrid_m alone at 1.0 GB; 34 s for run and
   verification (2026-11-01). Outputs: 178 MB of instance records, 40 MB of certificates, 9 MB of
   components for the model set at `full`.

8. **Limitations.**
   - Classes come from the op table's rules; an op without a rule relates nothing (reshape, slice, pad,
     gather and others give their results fresh classes), so two dims that are equal only through such
     an op are separate variables. This is sound and costs only sharing.
   - Dims that are equal by coincidence do not split components (a square and a non-square Dense are
     one component), but a difference in which axes are 1 does (a 1×1 and a 3×3 convolution are
     different components), and so does a difference in which size-attribute entries are 0 or 1
     (stride 1 and stride 2). A decode phase (sequence length 1) and its prefill phase therefore give
     separate components for the same layer.
   - `dynamic_*` ops and their static forms (`dynamic_broadcast_in_dim` and `broadcast_in_dim`) are
     different labels, so a layer exported with a constant batch and the same layer with a symbolic
     batch are different components; identifying them is a rewrite for `normalize`.
   - A custom call's `called_computations` stays a literal name in the label.
   - Equal WL hashes of non-isomorphic graphs only cost VF2 searches (each is a new representative);
     a VF2 search over `vf2_budget` fails the pass instead of guessing.
   - Size values of `size_in` nodes come from import's interpreter (`size_value::interpret`), which
     both the pass and its verification use; import's `sym_eval` validates that interpreter.
