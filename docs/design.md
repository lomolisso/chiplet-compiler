# chiplet-compiler: system specification

**Status:** v1, 2026-10-30. Every rule below is stated so that it can be challenged individually. A change to a rule is made together with the code that implements it and recorded in the decision log (section 11).

## 1. Goal and scope

Given a set M of quantized DNN models written in JAX, chiplet-compiler produces, automatically and without hand-written mappings:

1. **C**: the abstract *components* of M (layers and functional blocks).
2. **Z**: the canonical *kernel classes*. A kernel is a convex unit of computation with at most one contraction or reduction core. Every component is a composition of kernels plus data movement.
3. **F**: a set of *functional units* (FUs). Each FU F covers a subset cover(F) ⊆ Z, and together the FUs cover every compute kernel class in Z.
4. **For every FU:** an Allo description, XLS-generated Verilog, verification evidence and synthesis reports on FreePDK45 and ASAP7.

**In scope:** compute only, meaning datapaths plus the minimal per-FU loop controller.

**Out of scope:** memory hierarchy, interconnect, global scheduling, place and route.

**Ground rules.**
- **No hard-coded op-to-hardware mappings.** Every decision follows from StableHLO semantics, iteration-space analysis, graph theory or formal arithmetic identities. NP-hard steps use published algorithms, selectable in `config.toml`.
- **Reuse existing tools**: JAX, StableHLO, GraMi, Allo, XLS, Synopsys and Cadence tools, Verilator / iverilog, Yosys, z3 / bitwuzla. We write the glue and the analyses that no tool provides.
- **Fully automated.** `chipc run` executes the whole flow; no learned heuristics and no manual steps.
- **Memory-safe on a shared machine.** No pass computes kernels at model scale (section 8).

## 2. Inputs

`models/<key>/` holds 25 JAX ports, each rewritten to its standard deployed quantized recipe. The catalog of model keys and upstream checkpoints is in `models/catalog.py`.

- **The datatype contract per model** is `models/<key>/recipe.py`, collected by `models/quant_recipes.py`. It declares the weight, activation and accumulator dtypes, scale granularity, float carve-outs, and an `audit` of the GEMM/conv signatures every lowered phase must have. The framework reads recipes as **data**; it never special-cases a model.
- **Interface.** Every port exposes `CONFIGS` (`full`, `reduced`) and `build(cfg) -> ModelSpec`, defined in `models/spec.py`. A `ModelSpec` is a list of `Phase(name, fn, inputs, trip_count, …)` plus `param_specs` (abstract `ShapeDtypeStruct`s). No weights exist anywhere.

Facts about the lowered graphs that the design must handle. All were measured on 2026-10-30 with `python -m verify.dtype_audit` and op censuses.

| Fact | Consequence |
|---|---|
| Quantized GEMMs/convs in explicit quantize/dequantize form: `i8×i8→i32`, `f8E4M3FN×f8E4M3FN→f32`, int4 weights unpacked to `i8` with per-block scales | Datatypes are explicit operand/result element types; nothing is inferred. |
| Requantization is an **fp32** chain (`convert → ×scale → +bias → [act] → ÷scale → round/clip → convert`) | Requantization is a kernel class with two hardware variants (section 6.3). |
| Float carve-outs everywhere (softmax, norms, fp16/bf16 attention, fp32 SSM scans) | Float kernels are first-class, not an afterthought. |
| `func.call` to private functions in every model; `fft`, `real`, `imag`, `complex` (evo2, rnnt); `reduce_precision` (gemma4); CHLO ops in non-exported lowering | The op table is built from measurement (`import` pass) and must cover every op present. |
| Symbolic export adds `get_dimension_size`, `dynamic_broadcast_in_dim` / `dynamic_iota` / `dynamic_reshape`, `custom_call @shape_assertion` | These are **size computation**: tagged, not rejected. |
| Gather/scatter with arithmetic update (segment sums, embedding bags, MoE combine, sparse conv) | Scatter-reduce is a kernel class, not a data-movement boundary. |

## 3. Architecture

```
            config.toml (+ config.local.toml)
                         │
   chipc (Rust CLI) ── PipelineManager ── PassRegistry ── Pass impls (crates/cc-passes)
                         │                                   │
                  ArtifactStore (build/.<pass>/)       ToolRunner (Docker | Eda | Host)
                                                             │
                     python/cc_export (JAX)   GraMi   Allo   XLS   Yosys/Verilator   DC/Genus/FM/LEC
```

### 3.1 Pass contract

Each pass is one type implementing one trait. Sketch:

```rust
pub trait Pass: Send + Sync {
    fn name(&self) -> &'static str;               // == config section == build dir
    fn version(&self) -> u32;                      // bump when outputs change meaning
    fn requires(&self) -> &[ArtifactKind];        // inputs (produced by other passes)
    fn provides(&self) -> &[ArtifactKind];        // outputs
    fn resources(&self, cfg: &PassConfig) -> Resources;   // threads, memory_gb, timeout
    fn run(&self, ctx: &mut PassContext) -> Result<()>;
    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport>;  // pass-specific checks
}
```

- **As built.** The trait in `crates/cc-pipeline/src/pass.rs` adds three methods to this sketch: `tools()` (the tools whose versions enter the cache key), `model_dependent()` (false for passes that `--models` must not invalidate) and `check_config()` (strict parse of the section before anything runs). See `docs/framework/pipeline.md`.
- **The rule.** A pass reads only its declared inputs, writes only into its own `build/.<name>/`, and is a pure function of (inputs, config section, tool versions, pass version). That is what makes caching and parallelism safe.
- **Verification is part of a pass.** `run` is always followed by `verify`, and a pass whose `verify` fails is *failed*. `chipc verify <pass>` re-runs only the checks on existing artifacts.
- **Patterns:**
  - **Strategy** for selectable algorithms (MIS, clique, partition tie-breaks, …), one trait per decision, chosen by config.
  - **Adapter** for external tools (`ToolRunner`).
  - **Registry** for passes.
  - **Repository** for artifacts.
  - **Builder** for configs and test fixtures.
  - Errors are typed per crate (`thiserror`), with `anyhow` only at the CLI boundary.

### 3.2 Registry

`crates/cc-passes/src/registry.rs` lists every pass in one place. Adding a pass takes:
1. one module under `crates/cc-passes/src/<name>/` documented in `docs/passes/<name>.md`;
2. one registry line;
3. one `[passes.<name>]` section in `config.toml`;
4. smoke and deep tests.

Nothing else changes; the pipeline manager derives the order from `requires`/`provides`.

### 3.3 Pipeline manager

- **DAG.** The pass graph is built from `requires`/`provides` (`petgraph`). Cycles or a missing producer are a configuration error.
- **Parallel passes.** Supported from day one: the DAG already has independent branches (`characterize` vs the analysis chain; the soft-float library). Execution is Kahn-style. A ready pass is *admitted* when:
  - fewer than `pipeline.max_parallel_passes` passes are running, **and**
  - Σ `memory_gb` of the running passes, including this one, is ≤ `resources.memory_budget_gb`, **and**
  - Σ `threads` of the running passes, including this one, is ≤ `resources.threads_total`.

  **Oversize rule.** A pass whose own declaration exceeds a limit can never satisfy it, so it is admitted only when nothing else is running, and then runs alone.

  Each admitted pass runs on its own OS thread with its own `rayon` thread pool of `passes.<name>.resources.threads` threads.
- **Parallelism inside a pass.** Through that pool, with work stealing (`par_iter`, chunking with `with_min_len`). This is the dynamic scheduling of OpenMP's `schedule(dynamic)`. Passes iterate over independent units (models, phases, components, kernel classes, FUs) so that this is natural.
- **Resume and caching.** Before running a pass, its *cache key* is computed: blake3 over its input artifacts' content hashes, its config section (without `resources`, which never change outputs), the model set (for model-dependent passes), the versions of the tools it calls, and the pass version. If `build/.<name>/manifest.json` has the same key with `status = ok` and its outputs still hash to the recorded values, the pass is skipped.
  - `chipc run --from <pass>` forces a pass and its dependents.
  - `--only <pass>` runs one pass whose inputs exist.
  - `--models a,b` restricts the model set.
- **Failure.** A failed pass stops its dependents; independent branches finish. Logs and partial outputs stay in its build dir.
  - With `pipeline.stop_on_failure = true` the first failure stops admission and cancels running passes. Cancellation is cooperative (a `CancelToken` checked between work units), and tool runners kill their child processes and containers.
  - A pass exceeding `resources.timeout_min` is cancelled the same way and fails.
- **Observability.** `tracing` spans per pass and per work item. A run summary (time, peak RSS, cache hits) is written to `build/.pipeline/run-<timestamp>.json`.

### 3.4 Artifacts (`build/`)

```
build/
  .pipeline/                     run summaries
  .<pass>/
    manifest.json                cache key, inputs (kind, hash), config section, tool versions, status, timings, peak RSS
    out/<kind>/                  machine-readable outputs, one directory per artifact kind: JSON for small/human data, postcard for large binary data
    views/                       human views: DOT, Markdown tables, MLIR excerpts
    verify.json                  every check: name, status, evidence (certificate paths, solver logs, counts)
    logs/                        tool stdout/stderr
```

Each artifact kind has a versioned schema (a Rust type with `serde`, plus a JSON Schema in `docs/schemas/` generated from the Rust types). Notebooks and later passes read the same files; nothing is passed in memory between passes.

### 3.5 Configuration

- **Files.** One `config.toml` (tracked, commented defaults) plus an optional `config.local.toml` (untracked, machine paths and overrides) are merged, local winning. Unknown keys are an error (`deny_unknown_fields`).
- **Sections:**

| Section | Holds |
|---|---|
| `[pipeline]` | `max_parallel_passes`, `targets` (passes to reach), `stop_on_failure` |
| `[resources]` | `memory_budget_gb`, `threads_total` |
| `[paths]` | `build_dir`, `models_dir`, `third_party_dir`, `eda_toggle`, `pdk_*` (local file) |
| `[models]` | `keys` (`"all"` or a list), `config` (`full` / `reduced`) |
| `[testing]` | `deep_models`, `tiny_dim` (the size used for tiny instantiations) |
| `[tools.<tool>]` | docker image and caps, tool binaries, versions |
| `[passes.<pass>]` | pass options; `[passes.<pass>.resources]` holds `threads`, `memory_gb`, `timeout_min` |

### 3.6 Tool runners

One trait, three implementations. Every call is logged with its command line, caps, duration and exit code.

| Runner | Used for | Isolation and limits |
|---|---|---|
| `DockerRunner` | JAX export, Allo, XLS, `xlscc`, GraMi, Yosys / Verilator / iverilog, z3 / bitwuzla | Our image (`docker/Dockerfile`), `--memory=M --memory-swap=M --cpus=C --pids-limit`, `--user $(id -u):$(id -g)`, repo and `build/` bind-mounted. An OOM kills only the container. |
| `EdaRunner` | dc_shell, lc_shell, fm_shell, pt_shell, vcs, spyglass, genus, lec, xrun | Host, via `bash --norc -i -c '. "$EDA_TOGGLE"; eda-run "$@"' eda-run <tool> … </dev/null`. `eda-run` relies on an interactive shell: plain `bash -c` loads no modules. It inherits the session's Lmod variables and filters bash's no-TTY warnings (`docs/tools/eda_environment.md` §2). The driver's own environment is never touched. Tool-native CPU/memory caps, `ulimit`, an RSS watchdog that kills the process tree over `memory_gb`, timeouts; temp dirs on scratch disk, not the 8 GB `/tmp`. |
| `HostRunner` | uv-managed Python on the host (fallback), small utilities | `ulimit` plus watchdog. |

Why this split:
- The host has glibc 2.28 (XLS binaries need newer) and cgroups v1 (only Docker `--memory` reliably caps memory).
- The licensed EDA tools are installed only on the host.

Details are in `docs/tools/`.

## 4. The pass pipeline

Names are config sections and build directories. Arrows show data dependencies.

```
export → import → components → normalize → kernels → kernel_classes → iteration_space → datapaths → mine → merge → allo_emit → xls → rtl_verify → synth → report
                                                                                                         ↑               ↑
                                                                                    characterize ────────┘   softfloat ──┘
```

- `characterize` (primitive area/delay library) and `softfloat` (the Allo soft-float library) depend only on the environment, so they run in parallel with the analysis chain.
- `synth` and `rtl_verify` both consume `xls` and run in parallel.

For each pass: purpose, method, reused tools, outputs and **verification**. The verification is formal or structural and never needs model-scale compute. Implementation details are in `docs/passes/<pass>.md`.

### 4.1 `export` (Python, DockerRunner)

- **Purpose.** Each port phase → StableHLO, with component boundaries.
- **Method.**
  - A module interceptor wraps every leaf module call in `jax.lax.composite` with a namespaced name (`cc.<Class>`) and static attributes.
  - Flax linen: pass `module.variables` explicitly, re-apply an unbound clone, re-entrancy guard (verified 2026-10-30).
  - Haiku and Keras ports need their own adapters.
  - Input-dependent dimensions (batch, sequence, points, tokens) are symbolic (`jax.export.symbolic_shape` with constraints). **Architecture dimensions stay concrete**: symbolic layer widths fail at `init`/`apply` (verified).
  - Output: the serialized `Exported`, the `.mlir` text, and a JSON graph dumped through JAX's bundled MLIR Python bindings.
- **Verification.**
  - MLIR verifier passes.
  - The JSON→graph and MLIR→graph op multisets are equal (round trip).
  - The op census meets `recipe.py`'s audit.
  - Every composite's decomposition is present.
  - Nothing is executed.

### 4.2 `import`

- **Purpose.** JSON → the Rust IR (`docs/framework/ir.md`).
- **Method.**
  - Build `Module/Func/Op/Value` (an arena with typed indices) with typed attributes and `Dim = Const | Sym(expr)`, `expr` in JAX's dimension algebra.
  - Inline `func.call` and JAX's own composites (`chlo.*`), keeping provenance; `cc.*` composites stay component boundaries.
  - Symbolic shape inference: the size computation is interpreted over symbolic expressions (case splits on undecided signs and comparisons, the floor-division identities, run-time integers opaque), and every `?` takes the dim the table's shape rule derives.
  - Tag size computation: ops that compute small integer values from a dimension and feed only shape operands or other size ops (`get_dimension_size` and shape assertions by definition). A dimension value that enters compute is allowed and recorded (`dim_as_data`): size independence constrains the graph's structure, not the use of a size as a run-time scalar.
  - Classify every op with the **op-semantics table** (`crates/cc-ir/src/ops/`), built from the measured inventory. It is data, not code paths.
- **Verification.**
  - SSA and type invariants; declared types agree.
  - Symbolic dims re-evaluated by a concrete interpreter at sample sizes (translation validation).
  - Op census equal to `export`'s minus what inlining removed; GEMM/conv signatures equal.
  - Size tags recomputed independently.
  - Every op is covered by the table: an unknown op is a hard error that names the model and phase.

### 4.3 `components`

- **Purpose.** Composites → component instances → abstract components C.
- **Method.**
  - Each executed composite call is an instance (a call path: one decomposition may serve several calls).
  - The entry's other ops form maximal weakly connected regions (`anon:<hash>`), with composite calls as
    cut points and size computation joined to its first consumer; they are not split further.
  - Abstract component = instance without sizes or widths: every dim except the literal 1 is a class
    variable (classes are the axes the op table's StableHLO rules force equal), element types are
    (family, class), size-attribute entries other than 0 and 1 and constant values are parameters. Size
    computation and shape operands are left out; a dimension value entering compute is a `size_in` input.
  - Deduplicate by canonical form (WL bucket + VF2 isomorphism confirmation), by nesting height; a
    nested call is labelled by its child component.
- **Verification.**
  - Partition property (every op at every executed call path in exactly one instance).
  - Recoverability: substituting an instance's dims, widths and parameters back into its abstract
    component gives a graph isomorphic to the instance. The VF2 mapping is stored as a certificate and
    re-checked edge by edge against a graph rebuilt from the module.

### 4.4 `normalize`

- **Purpose.** Exactness-preserving cleanup and a fixed reduction order.
- **Method.**
  - Rewrites to a fixpoint: DCE, CSE, collapse of transpose and reshape chains, removal of identity transposes, broadcasts and conversions (only where the labelled graph proves them), and folding of splat constants (directly or broadcast from a scalar) into the ports of element-wise ops. Nodes keep their input ids and class ids, so instance bindings apply unchanged; one normalized graph per component.
  - Annotate ω (reduction order) on every `reduce`, `reduce_window`, `dot_general`, `convolution` and `scatter`: `seq` or `tree(b)` over the axes the op's spec definition reduces, per dtype family (config; floats must be set, default `seq`).
  - No reassociation of floating point, no FMA contraction, no float algebra.
- **Verification.**
  - Every rewrite **rule** is proven once with SMT (QF_BV / QF_FP / uninterpreted functions over symbolic scalars and indices), Alive-style, by one solver and re-checked by the other; float reassociation must fail as a control.
  - Per run: a ledger of every rewrite, replayed by an independent checker from the input graph to exactly the output; no rule matches the output; ω on every reduction; recomputed classes refine the stored ones.

### 4.5 `kernels`

- **Purpose.** Partition each component into kernels.
- **Method.** Op classes come from the op-semantics table: core (contraction/reduction), elementwise, layout, data movement, size compute, irregular. A kernel is a set K with:
  - **K1:** at most one core;
  - **K2:** convex (no path leaves K and re-enters);
  - **K3:** no escaping intermediates;
  - **K4:** domain alignment of prologue/epilogue members (shape equality and broadcast structure).

  Deterministic algorithm: seed cores, absorb epilogues to a fixpoint, then prologues, then group the remaining elementwise ops into map kernels, with ties broken by SSA order. **Scatter-reduce** (scatter whose update computation is arithmetic) is a core. Pure data movement and size computation are boundaries.
- **Verification.** An independent checker re-verifies K1–K4, the partition property, **acyclicity of the quotient graph** (convex parts can still form a cyclic quotient), and that the union of kernels plus boundary ops equals the component.

### 4.6 `kernel_classes`

- **Purpose.** Canonical kernel classes Z, with instance tables.
- **Method.**
  - Canonicalize dimension numbers and constants (value → instance table).
  - **Identity** = op structure + dtype *family* (signed int, unsigned int, fixed, binary float with exponent/mantissa format). **Width and extents are parameters**, never identity.
  - WL hashing, then VF2.
  - Instance table T_z per class: model, component, extents, widths, constants, ω, trip-count-weighted workload.
  - Sharing report: per-model sets, Jaccard, coverage at component vs kernel level.
- **Verification.**
  - Every class member is re-checked against the representative with its stored VF2 certificate.
  - Property tests: renaming and dimension permutation do not change the class.

### 4.7 `iteration_space`

- **Purpose.** Iteration domain, access functions and dependence structure per class; template eligibility.
- **Method.**
  - From the core's StableHLO semantics (dot_general, convolution, reduce, reduce_window, scatter-reduce), derive:
    - iteration dims tagged parallel or reduction;
    - affine access functions per operand;
    - the map / reduce / contraction class (contraction ⇔ some operand is not indexed by some dim, i.e. reuse exists).
  - Embedding relation ⊑ between skeletons: injective, tag- and incidence-preserving (e.g. matmul ⊑ conv). Used by `merge`.
  - **Template eligibility (section 7).**
- **Verification.**
  - Access functions are checked symbolically against the StableHLO spec's index formulas (SMT over integer index variables).
  - Every claimed space-time mapping is checked by SMT: the schedule respects every dependence and the allocation is conflict-free.

### 4.8 `datapaths`

- **Purpose.** A scalar datapath per non-template class.
- **Method.**
  - One iteration of the class:
    - a contraction becomes multiply + accumulate, with an accumulator vertex labelled (combiner, ω);
    - a reduction body becomes its scalar ops;
    - elementwise ops map one to one, and `convert` stays explicit.
  - Labels: `op@family`; edges carry ports (`*` for commutative ops).
  - This pass also creates the **SMT harness** (a SMT-LIB generator + `easy-smt` runner) used by the later passes.
- **Verification.** For each class, an SMT proof that the scalar datapath equals the StableHLO semantics of the kernel's ops on symbolic scalars, per width tuple in T_z (QF_BV for integers, QF_FP for floats).

### 4.9 `characterize` (parallel branch)

- **Purpose.** An area/delay library A of primitives: op × dtype family × width, plus mux and register.
- **Method.**
  - Generate each primitive once (DSLX → XLS → Verilog, or a small Verilog template).
  - Synthesize with DC (Nangate45 `.db` built by `lc_shell`) and/or Genus on FreePDK45 and ASAP7.
  - Record area, delay and leakage.
  - XLS's area model is a fallback.
- **Verification.**
  - Synthesis logs are free of errors and latches.
  - Formality/Conformal LEC of netlist vs RTL for every primitive.
  - Sanity: area is monotone in width.

### 4.10 `mine`

- **Purpose.** Frequent datapath patterns per skeleton class.
- **Method.**
  - Datapaths of one class family → one GraMi `.lg` graph (disjoint union).
  - Run the **directed** GraMi build at support τ.
  - Re-match each pattern with `vf2` (non-induced, same semantics as GraMi) and keep only convex occurrences.
  - Count non-overlapping occurrences with a **maximum independent set** on the overlap graph. The algorithm is config-selectable: exact ILP, branch-and-bound, or greedy.
  - Score s(p) = Σ_z workload_z · MIS(p, z).
- **Verification.**
  - Re-matching reproduces GraMi's set (every pattern has MNI ≥ τ).
  - Every kept occurrence is convex.
  - The MIS is independent: a cheap check on the stored solution.

### 4.11 `merge`

- **Purpose.** Functional units F with configurations.
- **Method.**
  - Cover datapaths with the top patterns; one FU per class to start.
  - Merge admissible pairs (same class, skeleton embedding ⊑) with Moreano's compatibility graph + maximum-weight clique (algorithm selectable). Weights come from A.
  - Agglomerate greedily by area gain under a logic-depth bound. Stop at gain ≤ 0 or a target count.
  - **Precision decomposition (section 6.2).**
  - The clique solution *is* each covered class's configuration (mux selects, constant registers): **configuration by construction**.
- **Verification.**
  - For every F and z ∈ cover(F), per width tuple: SMT proof of FU(cfg_z) ≡ datapath_z (∀ inputs).
  - Coverage: Z_compute ⊆ ⋃ cover(F).

### 4.12 `softfloat` (parallel branch)

- **Purpose.** An Allo library of float formats implemented with integer bit operations:
  - the formats the recipes use: FP8 E4M3FN, bf16, fp16, fp32 (E5M2 is optional; no port uses it today);
  - add, mul, FMA-or-not per contract, compare, convert, round-to-nearest-even;
  - FTZ / DAZ as configured.

  It makes every FU integer to Allo, so the Allo `llvm` simulator and the Allo XLS backend accept it.
- **Verification.**
  - Exhaustive tests for 8-bit formats (every operand pair).
  - Randomized and edge-case tests for wider formats against a bit-exact reference (`ml_dtypes`/NumPy with explicit rounding).
  - SMT QF_FP equivalence of the integer circuit vs the FP theory for small formats.

### 4.13 `allo_emit`

- **Purpose.** Each FU and each template instance → an Allo kernel plus its configuration interface.
- **Method.**
  - Emit Allo Python from the merged datapath (and template parameters), using `softfloat` for float families.
  - Schedule customizations: pipelining and unrolling follow from the skeleton.
- **Verification (functional).**
  - The Allo `llvm` target runs each FU × configuration × width on generated vectors at **tiny sizes** (symbolic dims set to `testing.tiny_dim`).
  - It is compared against our reference interpreter of the kernel class (Rust, bit-exact for integers, order contract for floats).
  - `allo.verify` checks schedule transformations against the unscheduled kernel.

### 4.14 `xls`

- **Purpose.** RTL.
- **Method.** Allo `xls` target → C++ for `xlscc` → XLS IR → `opt_main` → `codegen_main` (pipeline generator, delay model `asap7`, clock and stages from config).
- **Verification.**
  - The same golden vectors run in the XLS IR interpreter/JIT.
  - `check_ir_equivalence_main` between unoptimized and optimized IR (bounded for procs).
  - Codegen signature (latency, II) recorded.

### 4.15 `rtl_verify`

- **Purpose.** Traditional RTL verification.
- **Method.**
  - Simulate each generated module (Verilator or iverilog in Docker; VCS/Xcelium optional via `EdaRunner`) with the golden vectors, through a generated testbench.
  - Lint (Verilator `--lint-only`; SpyGlass optional).
- **Verification.** Pass/fail per FU × configuration × width; lint clean.

### 4.16 `synth`

- **Purpose.** Area, timing and power.
- **Method.**
  - DC (or Genus) on Nangate45 and ASAP7 at the configured clock.
  - Formality (or Conformal) LEC between RTL and netlist.
  - PrimeTime STA.
- **Verification.** LEC equivalent; timing met or slack reported; reports parsed into JSON.

### 4.17 `report`

- **Purpose.** The coverage proof Z_compute ⊆ ⋃ cover(F), area and timing per FU, sharing statistics, and a pass-by-pass verification summary.

## 5. Verification philosophy

1. **Translation validation, not verified passes.** Each pass checks its *output* against its input (Pnueli et al. 1998; Necula 2000). Passes emit **certificates** (isomorphism mappings, partitions, clique solutions, configurations), so the checks are cheap and independent of the pass's algorithm.
2. **Formal first.**
   - Graph properties are checked structurally.
   - Arithmetic equivalences are checked with SMT (section 5).
   - Rewrite rules are proven once, not per instance.
   - Hardware is checked with LEC.
3. **Numeric evaluation only at tiny, generated sizes.**
   - Simulation (Allo `llvm`, XLS IR interpreter, RTL simulators) uses symbolic dims instantiated to small values and synthetic data.
   - Never real weights, never model-scale tensors, never a full model forward pass.
4. **Exactness contract.**
   - Integer kernels: bit-exact everywhere.
   - Float kernels: bit-exact against the reference interpreter that implements the **same order contract** (ω, FMA or not, FTZ). Versus JAX itself only a tolerance makes sense: XLA's reduction order is unspecified and it contracts `a*b+c` into FMA.
5. **Every check is recorded** in `build/.<pass>/verify.json` with its evidence.

## 6. Datatypes and precision reuse

### 6.1 Dtype families and widths

A dtype is (family, width) for integers and fixed point, and (family = binary float, exponent bits, mantissa bits, special-value convention) for floats, e.g. E4M3FN has no infinities.

- **Kernel class identity** uses the family.
- **Widths** are instance parameters.
- Units are generated per family and parameterized by width.

### 6.2 Precision decomposition (cross-width reuse)

A w-bit integer multiply can be computed by ⌈w/n⌉² n-bit multiplies plus shifts and adds (schoolbook partial products; Karatsuba as an alternative). `merge` may therefore map a wider instance onto narrower units at lower throughput. Examples: TPUv1's 8-bit MAC array runs at half speed when one operand is 16-bit and at quarter speed when both are (Jouppi et al. 2017), and Bit Fusion composes 2-bit "BitBricks" (Sharma et al. 2018).

This is an **arithmetic identity**, proven in SMT for each width pair used. It is not a lookup table. The throughput and area trade-off comes from A and is reported per FU.

### 6.3 Requantization

Requantization (int32 accumulator → next tensor's storage type) is a kernel class with two hardware **variants**, both generated and both verified:

| Variant | Arithmetic | Exactness |
|---|---|---|
| `fp32` | fp32 scale multiply, as deployed by the ports (TensorRT / XNNPACK) | Bit-exact vs the ports' semantics under the order contract |
| `fixed` | int multiplier + rounding shift (gemmlowp / TFLite style; 64-bit intermediate) | Checked within a tolerance vs `fp32`; never claimed bit-exact |

The variant per model follows its recipe by default (`passes.merge.requant_variant = "recipe" | "fp32" | "fixed"`).

### 6.4 Float formats

Every float FU uses the `softfloat` library: FP8 E4M3FN, bf16, fp16 and fp32 (E5M2 optional), with fp32 accumulation where the recipe says so. FP8 conversions follow the OCP OFP8 spec: round-to-nearest-even, with saturating and non-saturating modes and the overflow-boundary rules of Rev 1.1 (`docs/context/literature/fp8_formats.md`).

## 7. "Solved" kernels without hard-coded mappings

Dense GEMMs are, in practice, computed on systolic arrays or tensor cores. chiplet-compiler reaches that conclusion **by proof, not by op name**:

1. **Detect uniform recurrences.** `iteration_space` checks whether a class's core is a *uniform recurrence*: an affine iteration domain with constant dependence vectors after fixing the reduction order (Karp, Miller & Winograd 1967).
2. **Search for a space-time mapping.** For uniform recurrences it searches for a linear schedule λ (λ·d ≥ 1 for every dependence d) and an allocation (projection) giving a systolic array (Kung & Leiserson 1979; Quinton 1984). The search space is small: λ and projections with entries in {-1, 0, 1} over at most 7 dims. A satisfying assignment is checked by SMT.
3. **Generate a template.** If a mapping exists and the kernel's reuse makes it worthwhile (config thresholds on extents and reuse factor), the class is routed to a **parameterized array template**: dims, dataflow (weight-, output- or input-stationary per the projection), dtype family, width and precision decomposition. The template is generated from the mapping, not chosen from a list.
   - Output-stationary tiled MAC arrays (tensor-core style) and weight-stationary arrays (TPU style) are two projections of the same recurrence.
4. **Fall back to mining.** Classes without a valid mapping (elementwise chains, softmax, norms, scatter-reduce, …) go to `datapaths → mine → merge`.

**Evidence recorded:** the mapping, the SMT proof, and the eligibility reason.

## 8. Resource policy (shared machine)

- **Abstract shapes only.** No pass materializes a tensor of model scale. Model-derived data is abstract (shapes, dtypes, graphs). Numeric data exists only at `testing.tiny_dim` scale.
- **Memory declared and enforced.**
  - Every pass declares `memory_gb` and `threads`.
  - The pipeline manager admits passes within `resources.memory_budget_gb`.
  - Docker calls get hard caps; EDA calls get a watchdog.
  - Defaults are conservative: 8 GB per Docker call, 32 GB total.
- **Bounded builds.** Builds of third-party tools (LLVM for Allo, Bazel for xlscc) use capped `-j` and memory flags in the Dockerfile.
- **Tests:**
  - **Smoke** tests touch all 25 models at the `reduced` config; abstract lowering is cheap.
  - **Deep** tests run the full pass on `testing.deep_models` with the same caps.

## 9. Testing and CI

| Layer | Where | What | Runs |
|---|---|---|---|
| Unit | in each crate (`#[cfg(test)]`) | data structures, algorithms on toy graphs | CI |
| Smoke | `tests/smoke/` | every pass on all 25 models at `reduced`: preconditions, schemas, invariants; full runs only where they are seconds | CI |
| Deep | `tests/deep/` | the full pass on `testing.deep_models`, including the tool calls | locally: `chipc test --deep` |
| Python | `python/tests/`, `verify/test_quant.py` | export front end, quantization library | CI |

`cargo nextest` profiles separate smoke from deep, and deep tests are in a `heavy` test group with `max-threads = 1`. CI (`.github/workflows/ci.yml`) runs fmt, clippy, unit, smoke and the Python tests on every pull request and every push to `main`; a pull request can be merged only when its `lint` and `test` jobs pass. Deep tests run on the build host before a change that touches tool calls is merged.

## 10. Repository layout

```
README.md, CHANGELOG.md, CONTRIBUTING.md, config.toml, config.local.toml.example
crates/        cc-ir, cc-core (artifacts, config, runners), cc-pipeline, cc-passes, cc-smt, cc-cli (binary `chipc`)
python/        cc_export (JAX front end, uv project), tests
models/        the 25 quantized JAX ports (input)
verify/        pre-framework Python checks of the ports (dtype audit, census, unit tests)
docker/        Dockerfile, build and self-test scripts of the toolchain image
scripts/       on-build-host, build-area (build host helpers)
third_party/   git submodules: grami, allo, xls (shallow)
tests/         smoke/, deep/, fixtures/
docs/          design.md, development.md, testing.md, framework/, passes/, tools/, schemas/, toolchain-image.md
build/         (ignored) build/.<pass>/ artifacts
```

## 11. Decision log

| Decision | Alternative rejected | Reason |
|---|---|---|
| Rust passes; Python only for JAX export | Python throughout | Graph-heavy analyses, speed, typed artifacts; JAX needs Python only at export |
| JSON from JAX's MLIR bindings into Rust | Rust MLIR bindings (melior) | melior needs a pinned system LLVM/MLIR and has no StableHLO dialect |
| Kernels defined structurally (K1–K4) | "Frequent subgraphs = kernels" | Frequency gives no coverage guarantee |
| Width = parameter, family = identity | Width in identity | Cross-width reuse (section 6.2) |
| Template eligibility by space-time mapping proof | Name-based mapping of GEMM/conv | No hard-coded mappings |
| Configuration by construction + ∀-only SMT | ∃∀ synthesis per (FU, kernel) | The clique already gives the mapping |
| Soft-float in plain Allo | Fork Allo / DSLX for floats | One RTL route; FP8 bit-exact; no fork to maintain |
| Host driver + capped Docker + `eda-run` | Single container | glibc 2.28 host, cgroups v1, host-only licensed EDA |
| Parallel pass execution with memory admission | Sequential only | Independent branches exist; about 100 lines; shared machine |
| Requantization has two verified variants | Integer only | The ports rescale in fp32 as deployed; fixed point is the cheaper hardware |
| Oversize passes run alone; `threads_total` is admitted too | Reject passes over the budget | A pass may legitimately need more than the default budget; running it alone keeps the sum bounded |
| `stop_on_failure` = stop admission + cooperative cancel; runners kill children | Let running passes finish | A long branch should not keep the shared machine busy after a failure; the same token enforces `timeout_min` |
| JSON by default, postcard supported from v1 (always with a JSON summary) | JSON only | Later graph artifacts need it; keeping it in `cc-core` from the start avoids a store change |
| One directory per artifact kind (`out/<kind>/`) | Passes list their files | Every written file is accounted for and hashed; nothing to forget |
| Rust 1.99.1 pinned exactly (`rust-toolchain.toml`, CI) | Floating `stable` | New clippy lints cannot break CI without a code change |
| Build LLVM for Allo from source in a capped image stage | Base on `chhzh123/allo` | Pinned and reproducible with our Python 3.12 + uv; that image (2025-12-21, 14.5 GB) is 10 months older than the Allo pin and brings its own OS and conda |
| Legacy `docker build` with `--memory`/`--memory-swap`/`--cpu-quota` | A buildx builder | buildx is not installed here; the legacy builder caps every RUN step hard |
| apt frozen to one `snapshot.ubuntu.com` date, PyPI by `uv --exclude-newer` | Per-package version pins | The archive drops old versions; one date freezes the whole package set |
| XLS Bazel tools built on Ubuntu 22.04, final image Ubuntu 24.04 | One base for every stage | 22.04 is XLS's build platform; 24.04 has Python 3.12 and the glibc 2.39 the z3 release needs |
| ASAP7 `.db` for RVT TT only (`[tools.pdk]`) | All VTs or corners | Spec default; more sets need only a config change |
| CI neither builds nor pulls the image | Pull from a registry | The image is host-only and ~15 GB; CI lints the Dockerfile and runs `doctor --dry` |
| Host-specific EDA fixes live in the EDA environment (`~/pdks/setup_env.sh`, wrappers in `~/pdks/extra_bin/`) | A per-tool env map in chipc's config | One place for the environment: `eda-on` users and `eda-run` get the same fix; chipc only calls `eda-run` (e.g. LC's bundled `libk5crypto`) |
| Every module level is a nested composite (phase root excluded) | Leaf modules only | Attention BMMs, MoE dots and other function-level ops sit in parent modules: leaf-only leaves them `anon`; the `components` pass keeps the hierarchy |
| Each port declares its symbolic input sizes (`ModelSpec.symbols`, `Phase.shapes`) | Discover axes by perturbation probes | Explicit and reviewable; constraints (divisibility, bounds) cannot be discovered |
| Every input-dependent axis is symbolic; ports are fixed until it is; no concrete fallback | Record a concrete axis as a limitation | Functional units must be valid for every input size; one symbolic module per phase proves the graph does not depend on sizes |
| Invariance certificate: re-export with every symbol's config field perturbed, require the identical graph | Trust symbolic tracing alone | Python branches on config ints are invisible to tracing (it caught llama's `query.shape[1] > 1` kernel choice) |
| `lax.composite` + closure conversion, batching and DCE rules registered by the exporter | Restrict ports (no closures, no vmap over modules) | jax 0.12 lacks the three rules; with them every port wraps without model edits, and dead code inside components is removed |
| Haiku parameters reach composites by closure conversion | `hk.custom_getter` substitution by name | Exact under any custom getter the port installs; no name matching |
| `Pass::fingerprints` (source trees in the cache key) and `models.config` in the cache key | Tool version probes only | `export` reads `models/` and `python/cc_export`, which are not artifacts; `full`/`reduced` changes outputs |
| `exported.bin` kept; evaluation rewrites composites into calls | Drop it | Tiny-size reference evaluation later; XLA's shape refinement does not refine composite decompositions |
| alphafold2 template slots stay concrete | Symbolic template count | AlphaFold feeds a fixed number of padded slots with a mask (a capacity, like multi-hot sizes); a symbolic count turns the stack into a `while` whose census differs from the recipe's counts |
| Sizes that the network needs as multiples are written `32*H` (yolo11, unet3d, rtdetr_v2; sdxl UNet `4*HU`) | A bare symbol with a `mod` constraint | JAX cannot prove the skip-connection concat shapes equal for a bare symbol; the multiple is the true contract (upstream fails otherwise) |
| Arena IR with typed `u32` indices; graph views built on demand | `petgraph::StableGraph` as the IR | Regions and multi-result ops map directly; plain vectors serialize deterministically |
| Symbolic dims in JAX's dimension algebra (polynomials over symbols with `floordiv`, `mod`, `max`, `min`) | Polynomials only | The ports' sizes halve and pad (`floordiv(H + 1, 2)`, `-L mod 256`); it is the algebra the exporter used |
| JAX's `chlo.*` composites are inlined | Keep them as components | They are lowering helpers (`top_k`, `erf`), not module boundaries (`passes.import.inline_composites`) |
| A dimension value entering compute is allowed and recorded (`dim_as_data`) | Reject it | rtdetr_v2 anchors, llama positions, means over a symbolic axis use sizes as run-time scalars; the graph stays size-independent |
| Components are compared by graph isomorphism (VF2 isomorphism mode after equal node and edge counts) | Non-induced subgraph matching | With equal node and edge counts a monomorphism, an induced match and an isomorphism coincide; the isomorphism mode states the intent |
| Anon regions are not split further | Split at layout ops | Layout and data movement are kernel boundaries (`kernels`, K1-K4); a second rule here would duplicate it |
| Size computation is in the partition but not in the abstract graph; `dim_as_data` operands are `size_in` inputs | Size ops as graph nodes | Import validated that result types determine shape operands; lowering differences in size code would split equal components |
| Dims abstracted by spec-derived classes (every dim but 1 is a variable, symbolic or constant) | One variable per distinct value; symbols kept in the graph | Classes follow from the labelled graph, so certificates map them; coincidental equalities (square Dense) do not split components |
| Component identity is the graph, not the composite name | Name + graph | Equal graphs from different modules or frameworks are one component; names are listed per component |
| Size-attribute entries are parameters except 0 and 1; floats are (family `float`, format parameter) | All literal; convention in the family | Sizes must not enter identity; 0 and 1 mark degenerate structure; design §6.1 makes the format a parameter |
| Reduction order ω defaults to `seq` for every family; `float` and `complex` must be set | `tree(b)` for floats | Systolic chains and single-accumulator MACs compute `seq`; adder latency is hidden by independent outputs, not by reordering one sum |
| Normalized components keep the input ids (no new deduplication) and the input's node and class ids | Re-deduplicate; rewrite the instance records | Parents' call labels and every instance certificate stay valid; sharing across components is found at kernel level |
| Rule proofs run when the pass runs (the image's solvers) and in CI (the same pinned releases on the runner); verify re-checks with the second solver | Proofs in deep tests only | The negative control and every obligation are checked on every change; a solver bug needs both solvers to agree |
| Rules v1: DCE, CSE, transpose/reshape collapse, splat folding, and identities only where the graph proves them | Identities checked against every instance | A rewrite valid for the component must not depend on the instance set |
| Index entries compared only for equality are 4-bit vectors in the obligations | 32-bit indices | At most 8 entries occur, so a counterexample renames into 4 bits; Z3 timed out on the 32-bit form |
