# Adapters: external resources and PDK manifests

Design §3.7: every external resource family has one trait, implementations chosen by name in
`config.toml`, a declaration of what each implementation supports, and one conformance suite shared by
all implementations. The traits and the data they exchange are in `cc_core::adapters`; the
implementations stay next to the code that uses them, and `cc_passes::adapters` lists every family's
registry in one place. PDKs are data: one manifest per cell library (`cc_core::pdk`, `pdks/<kit>.toml`).

## Traits

Each trait has `name()` (from `Named`, the name its config key holds) and `capabilities()` (a plain
struct). Traits stop at the resource boundary; the passes keep their own data types (development.md:
plain functions and data types before trait hierarchies), so the selection traits are generic over the
caller's problem type.

| Trait | Capabilities | Implementations (registry) | Config key |
|---|---|---|---|
| `SynthFlow` | `licensed`, `library` (`.db` or Liberty), `equivalence` (its checker), `hierarchy`, `ip_blocks`, `flop_only_paths`, `leakage` | `dc`, `yosys` (`cc_passes::eda::FLOWS`) | `[passes.characterize] tool` |
| `SmtSolver` | `bit_vectors`, `uninterpreted_functions`, `interchange_floats`, `any_float_sort`, `lemmas` | `bitwuzla`, `z3` (`cc_smt::SOLVERS`) | `[tools.smt] solver` |
| `CliqueSolver<G, S>` | `exact`, `budgeted`, `timed` | `exact_ilp`, `ostergard`, `greedy` (`merge::algo::clique::SOLVERS`) | `[passes.merge.clique] algorithm` |
| `MisSolver<G, S>` | `exact`, `budgeted`, `timed` | `exact_ilp`, `branch_and_bound`, `greedy_min_degree` (`mine::algo::mis::SOLVERS`) | `[passes.mine.mis] algorithm` |
| `Miner` | `directed`, `vertex_bound`, `mni_support` | `grami` (`mine::algo::grami::MINERS`) | `[passes.mine] miner` |
| `HlsFrontend` | `implemented`, `external_modules`, `targets` | `allo` (a stub: `implemented = false`) | `[passes.allo_emit] frontend` (planned pass) |
| `PnrFlow` | `licensed`, `synth_flow` | none yet (Innovus, OpenROAD in later versions) | none |
| `IpProvider` | `flows`, `licence`, `ops` (core operation families) | `designware`, `hardfloat`, `flopoco`, `xls_math` (`cc_passes::ip::PROVIDERS`) | `[passes.characterize] providers` (per flow, in order) |

- `Registry<T>` holds a family's implementations in a fixed order; `by_name` returns the one a config
  names, or `CoreError::UnknownAdapter` with the available names (`synthesis flow `genus` is not
  available (available: dc, yosys)`). The passes check their keys in `check_config`, before anything
  runs.
- **`SynthFlow`.** A job is a batch of designs on one kit at one period (`SynthJob`: the kit, the designs,
  the period, the session caps, a directory for library analysis shared by every job, the equivalence
  options or none). The flow writes its synthesis scripts (their text is part of the job's key, so they
  are pure functions of the job), its separate equivalence scripts (none for Yosys, which checks inside
  its script), runs them through the runners it needs (`FlowRunners`: `eda` for licensed tools through
  `eda-run`, `docker` for the toolchain image), and parses a design's reports (`ReportTexts`) into
  `Measured` and its equivalence report into `Lec`. The kit (`Kit`) is built by
  `cc_passes::eda::kit` from the Liberty text.
- **`SmtSolver`.** A solver is called as `<program> <file>`; `cc_smt::Solver` stays the identifier stored
  in certificates. The lemmas (the float encodings against native sorts, the composed multiplier steps)
  go to `cc_smt::lemma_solver()`: the first solver whose capabilities declare `lemmas` (Z3; Bitwuzla's
  release build has only the interchange float sorts).
- **`CliqueSolver`, `MisSolver`.** `solve(graph, …, &Limits { node_budget, timeout })` returns the
  solution or why there is none (`timeout`, `budget`); `merge` falls back to `greedy` then, as before.
  An exact solver over `max_compat_nodes` is replaced by `greedy` (recorded as `size`).
- **`Miner`.** `mine(runner, &MineJob)`: the data graph in the `.lg` text format, the support, the
  pattern bound, the miner's memory and time limit, in its own job directory, where a finished result is
  cached; returns the patterns in the same format, or `Timeout`.
- **`HlsFrontend`, `PnrFlow`.** Declared for the passes that will use them; the Allo stub answers with an
  error that says it is not implemented yet.
- **`IpProvider`** (design §4.12; below).

## Conformance suites

The same input to every registered implementation; the same verdict, or for a heuristic the same
answer class.

| Family | Test | What every implementation must do |
|---|---|---|
| `SynthFlow` | deep `characterize_reduced_catalog_on_every_flow`; smoke `adapters::every_synthesis_flow_writes_what_the_pass_reads`, `characterize` (stand-in tools) | the reduced catalog on both kits and targets with every check of `verify` green (ports and widths, outputs driven, flops, library cells, area, delay, trends) |
| `SmtSolver` | smoke `adapters::smt_solvers_conform` (host or image solvers) | the same verdicts on bit-vector, float and uninterpreted-function queries; a non-interchange float sort answered exactly by the solvers that declare `any_float_sort` |
| `CliqueSolver` | unit `merge::algo::tests::every_registered_clique_solver_conforms` | feasible cliques (closed under arc ends, no-goods respected); every exact one at the same optimum, a heuristic at most it |
| `MisSolver` | unit `mine::algo::mis::tests::every_registered_solver_conforms` | independent sets; every exact one of maximum size, claimed optimal, a heuristic no larger, its bound no smaller |
| `Miner` | deep `adapters::miners_conform` | GraMi's two-path fixture at support 2: exactly the three frequent patterns; a second call reuses the run |
| `HlsFrontend`, `PnrFlow` | smoke `adapters::declared_families_refuse_what_they_do_not_implement` | names unique, capabilities declared; the stub refuses to generate; no place-and-route flow is registered |
| `IpProvider` | smoke `adapters::ip_providers_conform`; deep `characterize::ip_cores_generate_and_synthesize_in_their_flows` | every probed op at every probed width (the catalog's formats, wide inner formats, the edges of each documented range): an offer exactly within the documented range, from a declared op family; the requested widths; each operand and result connected once, nothing but operands, results, constants and an iterative core's handshake; a source the provider's flows can use; the contract of the documentation (exact ops correctly rounded). Deep: the generated cores (HardFloat, FloPoCo, XLS) come out of the RTL step, and wrappers around them and around DesignWare synthesize in their flows with every cheap check green |

## Float IP cores (`IpProvider`)

`cc_core::adapters::ip` and `contract`; implementations in `cc_passes::ip` (data: tables of components,
parameters and contracts from the vendors' documentation, listed in each module).

- **Lookup.** `core(op, exp_width, sig_width) -> Option<CoreOffer>`: by core operation and operand widths
  (stored significand bits, without the hidden bit), never by format name. `None` outside the provider's
  documented range. Core operations (`CoreOp`, written as strings):

  | Op | Operands → results |
  |---|---|
  | `add`, `sub`, `mul`, `div`, `pow` | `a0`, `a1` → `z` |
  | `sqrt`, `rsqrt`, `exp`, `exp2`, `ln`, `log2`, `sin`, `cos` | `a0` → `z` |
  | `compare` | `a0`, `a1` → `lt`, `eq`, `gt` (DesignWare also `unordered`); quiet |
  | `to_int.<rne\|rtz\|floor\|ceil>.<s\|u><w>` | `a0` → `z`, a `w`-bit integer, saturating |
  | `from_int.<s\|u><w>` | `a0` (integer) → `z`, nearest even; the widths are the result's |
  | `convert.e<E>m<M>` | `a0` → `z` in the (E, M) format, nearest even |
  | `from_ieee`, `to_ieee` | between IEEE bits and a core's own encoding (FloPoCo) |

- **Offer.** `CoreOffer`: the `CoreInstance` (module, parameters as Verilog literals, port → `a0`/`a1`, a
  result, a constant such as `3'b000`, or empty for an open output; an iterative core adds `clk`, `rst`,
  `start`, `done`), the `Contract` (accuracy: correctly rounded, faithful, n ulps, an absolute bound or
  unbounded; IEEE special values; subnormals IEEE or flushed), a delay hint (none yet: `characterize`
  measures), the bit-level Python model (`ieee.<op>`: the IEEE operation correctly rounded at the instance's
  widths under its subnormal mode; `reference.<op>`: the real function correctly rounded, within every
  contract here; `flopoco.<op>` for FloPoCo's encoding changes), the source (`Library { name }`: a vendor
  library the flow links; `Generated { generator }`: a generator run in the RTL step), the encoding (IEEE or
  FloPoCo's) and the timing (combinational, or iterative with a cycle bound). A generated core's module is
  `<provider>_e<E>m<M>_<op>`; the primitive that holds it carries that id.
  `cc_passes::ip::verilog_instance` writes the instantiation.
- **Implementations.**

  | Provider | Flows | Ops | Contract | Source |
  |---|---|---|---|---|
  | `designware` | dc | add, sub, mul, div, sqrt, rsqrt (correctly rounded); exp, exp2 (2 ulps); ln (1 ulp), log2 (2 ulps), with `extra_prec = sig_width`, so `sig_width` ≤ 29; sin, cos (radians, unbounded relative error); compare; to_int (signed), from_int | IEEE subnormals (`ieee_compliance = 1`) | DesignWare (`dw_foundation.sldb`) |
  | `hardfloat` | dc, yosys | add, sub, mul, div and sqrt (iterative, at most p + 2 cycles), compare, to_int, from_int, convert; correctly rounded | IEEE subnormals | `hardfloat-gen` (Verilog) |
  | `flopoco` | yosys | exp, ln, pow (faithful), from_ieee, to_ieee | subnormals flushed | `flopoco` (VHDL) |
  | `xls_math` | yosys | f32 sin, cos: \|error\| ≤ 2^-22 | subnormals flushed | `fp_sin_32`, `fp_cos_32` of `fp_sincos_32.x` |

  Ranges: DesignWare `exp_width` 3-31, `sig_width` 2-253 (exp 57, exp2 58, ln and log2 29, sin and cos 32;
  integer sizes 3-512); HardFloat w ≥ 3, p = sig_width + 1 ≥ 3, p ≤ 2^(w-2) + 3; FloPoCo the widths its unit
  tests run (`wE` 5-11, `wF` 5-52; exp and pow also `wF + 4 ≤ 2^(wE-1) - 1`); XLS f32 only. Round to integral
  is not a core anywhere: `to_int` with the mode and `from_int` with glue (`routes`).
- **Selection and fit checks.** `[passes.characterize] providers` lists the providers of each flow in order
  (default `dc = ["designware", "hardfloat"]`, `yosys = ["hardfloat", "flopoco", "xls_math"]`); the first
  that answers is used. A provider listed for a flow it does not serve, or an unknown one, is a configuration
  error. `cc_passes::ip::routes` maps every float op of the catalog to a core (native, or through a wider
  format within the `[numerics]` budget), a composition, glue, an exclusion or a gap
  (docs/passes/characterize.md §4.1).

## PDK manifests (`cc_core::pdk`)

One TOML file per cell library in `paths.pdk_manifests` (default `pdks/`), named after the kit:

| Key | Meaning |
|---|---|
| `name`, `description` | the kit's name (equals the file stem) and what it is |
| `root` | the kit's directory under `paths.pdk_root`; `paths.pdks.<kit>` overrides it with a full path |
| `liberty` | the Liberty file(s), relative to the root; `{group}`, `{vt}`, `{corner}` substituted, `*` matching within the file name; exactly one match per group (or one file without groups) |
| `groups`, `vt`, `corner` | the cell groups shipped as separate libraries, and the VT and corner of the files |
| `merged` | with groups: the one library `chipc doctor` merges them into (`{vt}`, `{corner}` substituted) |
| `lef` | LEF files, as `liberty` (doctor checks they exist; place and route will read them) |
| `driving_cell`, `clock_ps` | the cell that drives every input of a characterized design, and the default clock target |
| `xls_delay_model` | XLS's delay model for the library, when XLS has one |

`chipc doctor` checks every manifest (`pdk.<kit>`, `pdk.<kit>_merged`), compiles every Liberty file to
`.db` and synthesizes a tiny adder on each kit at its clock (`dc.<kit>`); `characterize` links the merged
library (or the single one) and takes the driving cell and clock from the manifest unless its section
overrides them. The repository ships `nangate45.toml` and `asap7.toml` (docs/tools/pdks.md). The former
keys `paths.nangate45_lib`, `nangate45_lef`, `asap7_root`, `asap7_nldm_dir`, `asap7_lef_dir` and
`[tools.pdk]` are rejected with a message that names the manifests.
