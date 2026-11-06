# `env`: `chipc doctor` and the liberty `.db` files

## 1. Purpose and position

`env` is not a pass in the DAG. It is the environment every pass relies on. `chipc doctor` proves that each external tool works on this machine by making it do one small real job. It compiles the Liberty files that Design Compiler needs into `.db` files, and it records the evidence in `build/.env/`, the only directory it writes.

Later passes read from it:
- `characterize` and `synth` use `build/.env/out/pdk_db/*.db` and the merged libraries of kits shipped as cell groups (ASAP7) in `build/.env/out/pdk_lib/`;
- every Docker pass uses the image whose pins doctor checked.

Run doctor after cloning, after an image rebuild, and whenever a tool or licence looks broken:

```bash
chipc doctor                 # all checks; exit 0 = green
chipc doctor --dry           # print every command, run and write nothing
chipc doctor --skip-licenses # no DC run
chipc doctor --build-args    # the docker/build.sh inputs, from config.toml
```

## 2. Background

- Tool runners: design §3.6.
  - Licensed tools run on the host through `eda-run`, using the verified `bash --norc -i -c …` form (`docs/tools/eda_environment.md` §2).
  - Open-source tools run in the capped toolchain image (`docs/tools/docker_resources.md`, `docs/toolchain-image.md`).
- Liberty and `.db` files (`docs/tools/pdks.md` §4).
  - `write_lib` takes the Liberty `library (NAME)`, not the file name.
  - Design Compiler links only `.db` files.
  - Formality and PrimeTime can read `.lib` directly.
- The kits are data: one PDK manifest per cell library (`pdks/<kit>.toml`, `docs/framework/adapters.md`, `docs/tools/pdks.md`).
- Licences and the tool-native caps used here: `docs/tools/synopsys.md`.

## 3. Inputs and outputs

**Inputs:**
- `config.toml` `[tools.*]` holds the pins and `[tools.eda]` the licence settings; `paths.pdk_manifests` (`pdks/`) holds one manifest per kit, which selects its Liberty files (VT, corner, groups) and LEF.
- `config.local.toml` `[paths]` holds the toggle script, the scratch directory and the kits' root (`pdk_root`, or `pdks.<kit>` per kit).
- The submodules, the image, the host EDA installation and the PDK files.

`build/.env/` contents:

| Path | Content |
|---|---|
| `verify.json` | `VerifyReport` (pass `env`): one check per item in §6, with evidence paths, counts and notes |
| `manifest.json` | `Manifest` (pass `env`), with the fields below |
| `out/pdk_db/<library>.db` | `NangateOpenCellLibrary.db`, `asap7sc7p5t_<G>_<VT>_<C>_nldm_<date>.db` for the five ASAP7 groups, and `asap7sc7p5t_<VT>_<C>.db` for their merge |
| `out/pdk_lib/asap7sc7p5t_<VT>_<C>.lib` | the five ASAP7 group libraries merged into one (`liberty.rs`): the first file's header and groups, every cell of the others, their other top-level groups the first lacks |
| `out/selftest/` | inputs and outputs of `cc-selftest`: `add8.cc`, the IR, `add8.v`, `q.smt2`, GraMi `two_paths.lg` and `Output.txt`, `allo_add.py` |
| `out/dc_tiny/` | `<kit>_area.rpt` and `<kit>_netlist.v` of the tiny DC runs |
| `eda/<job>/` | the working directory of each EDA call, with its scripts and the tool logs |
| `logs/` | `tools.jsonl` and one stdout/stderr pair per call |

`manifest.json` fields:
- `inputs`: one `liberty:<library>` record per Liberty file, with its blake3 hash;
- `outputs`: `pdk_db`, `pdk_lib`, `selftest` and `dc_tiny`, with every file hashed;
- `tool_versions`: every probed version, the submodule HEADs and the image ID;
- `status`: `ok` or `failed`.

## 4. Algorithm

1. **Host.**
   - `MemAvailable` compared with `resources.memory_budget_gb`; below it, doctor only warns.
   - `paths.scratch_dir` must be set and writable; its free space is reported by `df -Pk`.
2. **Submodules.**
   - Each HEAD (`git rev-parse`) must equal its `config.toml` pin, and its `.gitmodules` entry must say `shallow = true`.
   - Allo's LLVM gitlink must equal `tools.allo.llvm_commit`.
   - Every patch set `third_party/patches/<submodule>/` must apply to its submodule as the image build applies
     it: in file-name order, each patch on top of the ones before. The series is applied with `git apply` to a
     copy of the files it touches (`patches/<submodule>/` in doctor's directory); the submodule is never written.
3. **Image.**
   - The daemon answers.
   - `<image>:<tag>` exists.
   - The pins in `/opt/cc/versions.json` equal the config.
   - `cc-probe`, run again now, reports the same version for every tool as at build time.
   - `cc-selftest /work/out` runs with `out/selftest` bind-mounted, and each `cc-selftest: pass <step>` line becomes one check. The steps are:
     - `xls`: `xlscc` → `opt_main` → `codegen_main` on an 8-bit `XlsInt` adder;
     - `xls_equiv`: `check_ir_equivalence_main`, unoptimised against optimised IR;
     - `rtl_tools`: Yosys `synth`, iverilog and Verilator lint of the generated Verilog;
     - `smt`: z3 and bitwuzla prove `x+x = x<<1` (QF_BV, `unsat`);
     - `grami`: directed GraMi on a 6-node toy graph, `freq 2`, `limitedtime=false`;
     - `allo`: a 4-element Allo kernel through the `llvm` target, compared with numpy;
     - `xls_extern`: an Allo kernel calling an external f32 multiplier (`allo.extern_verilog`, NumPy model) is
       simulated through `llvm`, then goes through `xlscc`, `opt_main` and pipelined `codegen_main`, whose Verilog
       must instantiate `DW_fp_mult` (the patched toolchain, `docs/third-party.md`);
     - `xls_float_lib`: XLS's `fp_sincos_32` (DSLX) to combinational Verilog, read by Yosys (`hierarchy -check`);
     - `flopoco_ghdl`: FloPoCo's `FPExp` (wE=5, wF=10) read by Yosys through the GHDL plugin.
4. **EDA versions.**
   - `dc_shell`, `lc_shell`, `fm_shell` and `pt_shell` with `-version`, `vcs -full64 -ID`, and `xrun` with `-version`.
   - The version is the first line naming one: `… version …` or `Tool: <name> <ver>`. The exit code is ignored, because `dc_shell -version` exits 1.
5. **PDKs and `.db`.** For every manifest in `paths.pdk_manifests` (sorted by name):
   - Its Liberty files and LEF files are found under the kit's root: Nangate45's `stdcells.lib`/`.lef`; ASAP7's five group files of the manifest's VT/corner (each group's date stamp is matched, not assumed; exactly one file per group, otherwise doctor fails) and its 1x LEF (`pdk.<kit>`).
   - `library (NAME)` is read from the first 256 KiB of each file.
   - A kit with cell groups is merged into its `merged` library, for ASAP7 `out/pdk_lib/asap7sc7p5t_<VT>_<C>.lib` (`pdk.<kit>_merged`). Units and the nominal operating point must agree, a cell may appear only once, and a shared group with different text (the driver waveforms) keeps the first file's (noted). Design Compiler's library analysis needs one library with inverters and flops: on the group files alone it leaves placeholders and repeats the analysis (about 100 s) on every compile. Yosys' ABC reads one Liberty file. OpenROAD's `mergeLib.pl` merges ASAP7 the same way.
   - `make_db.tcl` (`read_lib` + `write_lib NAME -format db` per file, the merged library included) runs in `lc_shell`, and the `.db` files are copied to `out/pdk_db/`.
   - LC is skipped when the previous manifest has the same `.lib` hashes and `lc_shell` version and every `.db` exists.
6. **Synopsys licence and `.db` link.**
   - An 8-bit adder goes through `compile_ultra` with `set_host_options -max_cores 2` on each kit's linked library (the merged one for a kit with groups) at the kit's clock in the library's time unit, read from its Liberty header: Nangate45 2 ns, ASAP7 1000 ps.
   - Pass: no `Error` lines, `Total cell area` > 0, and the netlist was written.
7. Write `verify.json` and `manifest.json`. The exit status is 0 only if no check failed.

Each step is a handful of process launches. The cost is the one-time LC compilation of about 120 MB of Liberty text plus two tiny DC runs.

## 5. Config keys

| Key | Default | Meaning |
|---|---|---|
| `tools.docker.{image,tag}` | `chiplet-compiler/toolchain`, `0.5.0` | the image doctor checks |
| `tools.docker.build.*` | see `config.toml` | image-only pins and build caps (`docker/build.sh`) |
| `tools.{xls,allo,grami}.*` | see `config.toml` | submodule and image pins |
| `paths.pdk_manifests` | `pdks` | the PDK manifests; each selects its kit's Liberty set (ASAP7: `RVT`, `TT`, all five groups) |
| `tools.eda.max_licenses` | 2 | concurrent EDA calls (one semaphore) |
| `tools.eda.license_retries`, `license_backoff_s` | 3, 30 | retry after a licence failure; the wait doubles each time |
| `tools.eda.license_error_patterns` | Synopsys `SEC-12/21/50/85/87`, the observed Cadence periodic-check message | output substrings that mean "no licence" |
| `paths.{eda_toggle,scratch_dir,pdk_root}`, `paths.pdks.<kit>` | none | machine paths (`config.local.toml`); the former `nangate45_lib`, `nangate45_lef`, `asap7_root`, `asap7_nldm_dir`, `asap7_lef_dir` and `[tools.pdk]` are rejected with a pointer to the manifests |

## 6. Verification

Every check is computed independently of the code under test: it reads tool output and files, never doctor's own state.

| Check | Proves | Evidence |
|---|---|---|
| `host.memory`, `host.scratch_dir` | room to run; temp space off the 8 GB `/tmp` | counts `mem_available_gb`, `free_gb` |
| `submodule.{xls,allo,grami}`, `submodule.allo.llvm` | source = pins | HEAD and pin in notes |
| `submodule.<name>.patches` | each pinned patch still applies to its submodule, in series | note per patch |
| `docker.daemon`, `docker.image` | image present | image ID |
| `image.pins` | the image was built from these pins | mismatch count |
| `image.tools` | every tool still starts, at its recorded version | tool and mismatch counts |
| `image.selftest.*` | each open-source tool does real work | `out/selftest/*`; GraMi pattern count |
| `eda.version.*` | every licensed tool starts through `eda-run` | version line |
| `pdk.manifests` | (only on failure) the manifests are readable | the error |
| `pdk.<kit>` (`pdk.asap7`, `pdk.nangate45`) | the manifest's Liberty and LEF files exist | manifest, file paths, library count |
| `pdk.<kit>_merged` (`pdk.asap7_merged`) | a kit's groups merge into one library | `out/pdk_lib/*.lib`, cell count, differing groups |
| `pdk.db` | `.db` files exist for every kit | `out/pdk_db/*.db`, LC error/warning counts |
| `dc.<kit>` (`dc.nangate45`, `dc.asap7`) | Synopsys licence; the `.db` files link and map | `out/dc_tiny/*` (area report, netlist) |

## 7. Resource behaviour

- Doctor's caps are constants in `doctor.rs`:
  - host commands: 1 GB, 120 s;
  - containers: 4 GB, 2 CPUs, 900 s;
  - EDA probes: 4 GB, 600 s;
  - LC: 8 GB, 1 CPU, 1 h;
  - DC: 8 GB, 2 cores, 30 min.
- All EDA calls hold one of the `max_licenses` slots, and each gets its `TMPDIR` under `<scratch_dir>/tmp/eda/` (removed after a successful call).

## 8. Known limitations

- Licence checks show that a licence was available at that moment, so passes still rely on `EdaRunner`'s retry.
- `image.tools` compares version strings only. A tool that starts but is functionally broken is caught by `image.selftest`, and only for the features the self-test exercises.
- The `.db` cache keys on `.lib` hashes and the `lc_shell` version. A changed LC licence or setup that alters output without changing the version is not detected; delete `build/.env/out/pdk_db/` to force a rebuild.
- Doctor relies on the EDA environment for host fixes: `lc_shell` starts only through the `~/pdks/extra_bin/lc_shell` wrapper that `setup_env.sh` puts on `PATH`. Without it, LC W-2024.09's bundled `libk5crypto` clashes with this host's `krb5-libs` 1.18.2-34. On another host, `eda.version.lc_shell` fails if such a fix is missing.
