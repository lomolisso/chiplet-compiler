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

## Conformance suites

The same input to every registered implementation; the same verdict, or for a heuristic the same
answer class.

| Family | Test | What every implementation must do |
|---|---|---|
| `SynthFlow` | deep `characterize_reduced_catalog_on_every_flow`; smoke `characterize` (stand-in tools) | the reduced catalog on both kits and targets with every check of `verify` green (ports and widths, outputs driven, flops, library cells, area, delay, trends) |
| `SmtSolver` | smoke `adapters::smt_solvers_conform` (host or image solvers) | the same verdicts on bit-vector, float and uninterpreted-function queries; a non-interchange float sort answered exactly by the solvers that declare `any_float_sort` |
| `CliqueSolver` | unit `merge::algo::tests::every_registered_clique_solver_conforms` | feasible cliques (closed under arc ends, no-goods respected); every exact one at the same optimum, a heuristic at most it |
| `MisSolver` | unit `mine::algo::mis::tests::every_registered_solver_conforms` | independent sets; every exact one of maximum size, claimed optimal, a heuristic no larger, its bound no smaller |
| `Miner` | deep `adapters::miners_conform` | GraMi's two-path fixture: the frequent patterns and their support |
| `HlsFrontend`, `PnrFlow` | smoke `adapters::declared_families` | names unique, capabilities declared; the stub refuses to generate; no place-and-route flow is registered |

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
