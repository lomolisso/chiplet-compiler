# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `import` pass and `cc-ir`: typed IR with symbolic dimensions, inlined calls, size tagging and the op-semantics table.
- `components` pass: composite calls and entry regions as instances, deduplicated into size- and width-free components with certificates.
- `normalize` pass and `cc-smt`: exactness-preserving rewrites proven once with SMT, a replayable ledger, and a reduction order on every reduction.
- `kernels` pass: every component partitioned into convex single-core kernels and map kernels, with an independently checked partition and kernel graphs.
- `kernel_classes` pass: kernels grouped into position-, width- and size-free classes with certificates, instance tables with workloads, and the component vs kernel sharing report.
- `datapaths` pass and the SMT harness: every class's scalar datapath, proven equal to the kernel's StableHLO semantics for every width tuple.
- `characterize` pass: area, delay and leakage of every element-wise op at every type the SMT semantics defines, multiplexers, registers, accumulators and split multipliers, on Nangate45 and ASAP7 at two timing points, each netlist checked by LEC and sanity checks.
- `mine` pass: frequent datapath patterns per family by GraMi, re-matched with VF2, kept when convex, counted by a maximum independent set per class (exact ILP, branch and bound or greedy) and ranked by covered workload.
- `merge` pass: configurable functional units by datapath merging (compatibility graph and maximum-weight clique, agglomerated under a depth bound) and parameterized arrays for template classes, every configuration proven equal to its datapath.
- `merge` fixed requantization (`requant_variant = "fixed"`): rescaling chains become one integer `rescale` block (exact multiply by the scale's significand, rounding shift), proven within 1 of fp32, with its RTL in the `characterize` library.
- `iteration_space` pass and the op table's iteration semantics: per class its affine accesses proven against the spec, dependences, the embedding relation, and routing of uniform contractions to systolic templates by a proven space-time mapping.
- `import` tags constant computation (ops no data argument reaches: position and rotary tables, masks from `iota`, sizes as floats); `components` leaves it out of the graphs and reads its values as `const_in` inputs, so it never becomes a kernel.
- `datapaths` records the hardware's order of every reduction whose order binds, and its `verify.json` states the error bound against any other order.
- Adapters for external resources (`cc_core::adapters`): synthesis flows, SMT solvers, clique and MIS solvers, the miner, the HLS front end (a stub) and place and route (declared), each selected by name in `config.toml` with declared capabilities and a shared conformance test.
- PDK manifests (`pdks/<kit>.toml`): each kit's Liberty files, LEF, VT, corner, driving cell, default clock and XLS delay model, under `paths.pdk_root`.
- `characterize` cheap checks on every netlist: every output driven, and an area below half of a narrower width's fails.
- `[passes.mine] miner` selects the frequent subgraph miner (`grami`).
- `cc-float`: glue wrappers around float IP cores (subnormal flush with one more exponent bit, the E4M3FN decoder and round-to-nearest encoder, FP8 saturation, NaN and signed-zero handling, NaN to 0 for float to integer), each proven bit-exact on its Verilog through Yosys `write_smt2` with the cores bound by their contracts; compositions (power, tanh, log1p, rsqrt) sized to an error budget from the cores' accuracy, their special values proven the same way.
- Composition-error sampler (`python -m cc_float.sampler`): cores modelled at their contracts, exhaustive over the 16-bit and 8-bit formats, stratified over f32; and an exhaustive check of bf16 and f16 sine and cosine through XLS's `fp_sincos_32` IR (`python -m cc_float.sincos`).
- Toolchain image 0.5.0: external Verilog modules in Allo (`allo.extern_verilog`) and `xlscc` (`hls_extern_verilog` attribute), Allo's XLS emitter fixed for calls and following loop and partition schedules (pinned patches), FloPoCo, GHDL with its Yosys plugin, and XLS's DSLX sine/cosine and exponential libraries.
- Float IP providers (`IpProvider`): DesignWare for Design Compiler; Berkeley HardFloat, FloPoCo and XLS's float library for the open flow; each core with its accuracy contract, selected per flow by `[passes.characterize] providers`.
- `[numerics]`: the error budget of every float op and format in ulps, by default NVIDIA's CUDA table for f32 and 1 ulp for f16, bf16 and FP8.
- `characterize` maps every float op to the flow's IP cores (`routes.json`) and lists the ops without one; it reads VHDL and generated cores (HardFloat, FloPoCo, XLS DSLX libraries) and links DesignWare's synthetic library for the jobs that instantiate it.
- Toolchain image 0.6.0: Berkeley HardFloat (Chisel, pinned submodule) with our generator of IEEE-interface cores, firtool and Chisel pinned.

### Changed

- `characterize` runs 12 Design Compiler sessions at once by default (`tools.eda.max_licenses` 12, `threads` 12, `memory_gb` 96).
- `datapaths` and `merge` prove the design's numerics: f32 and bf16 flush subnormals to zero, conversions into FP8 saturate (NaN kept).
- `characterize` checks netlist against RTL only with `lec = true` (off by default); `library.json` and `summary.json` schemas 2 (`lec` and `lec_tool` optional, the kit's clock recorded).
- `characterize` Design Compiler job directories no longer depend on the toolchain image tag (the RTL is in their key by its text), so a new image reuses them.
- `paths.nangate45_lib`, `nangate45_lef`, `asap7_root`, `asap7_nldm_dir`, `asap7_lef_dir` and `[tools.pdk]` are replaced by the PDK manifests and `paths.pdk_root` / `paths.pdks.<kit>`; the old keys are rejected with a message.
- Solver answers are cached once for every pass and for `verify` (`paths.smt_cache`, shared by every build directory of a host): constant lemmas and rules are solved once per solver version, `verify` still regenerates and compares every query.
- `mine` runs GraMi once, at the last threshold, and derives every step's pattern set by MNI (`grami_runs = "once"`, outputs unchanged; `"sweep"` restores one run per step).
- Pass manifests and run summaries record the time spent in `verify`; `chipc run` prints it; `CC_PROFILE=release` runs the optimized `chipc` through `scripts/on-build-host`.
- A pass's containers share its declared memory and threads (each at most the `[tools.docker]` ceiling, now 16 GB and 8 CPUs) instead of one fixed 8 GB cap; run budget 128 GB and 32 threads, and pass resources raised to their measured needs (export, datapaths, mine, merge, solver passes).
- `export` records each symbol's integer size at the exported config and a census of the phase lowered on concrete shapes (`census_concrete.json`, integer FLOPs).
- `chipc doctor` also merges the ASAP7 group libraries into one Liberty file and `.db`.
- `chipc doctor` checks every patch set under `third_party/patches/` as a series, and its image self-test covers external modules, the XLS float library and FloPoCo through GHDL.
- Toolchain image 0.4.0: GraMi's directed build takes a `maxNodes` bound on the pattern size (`third_party/patches/grami/0002`).
- `scripts/on-build-host --pick` chooses among iacoma227, iacoma231 and iacoma232 by their scarcest resource (idle cores, RAM, disk); both scripts run directly on the local host.

### Removed

- The Genus and Conformal synthesis path (`tool = "genus"`) and their `chipc doctor` license checks.

### Fixed

- `merge` resolves gain bounds in batches of a fixed size: the merge order no longer depends on the thread count (outputs at the default 16 threads unchanged).
- `config.local.toml.example` no longer lowers the run memory budget to 32 GB, which held back the raised pass resources.
- Containers get `--cpus` rounded down to hundredths: a fractional share such as 16 CPUs over 6 solver jobs was rejected by Docker.
- JSON floats are parsed exactly (`serde_json` `float_roundtrip`); `import` read some float attributes one ulp off.
- `components` inlines a custom call's reducer computation as its region, so `dynamic_reduce_window` kernels carry their combiner and classes tell reducers apart.
- `scripts/on-build-host` sources the host's `/etc/profile`, so `eda-run` finds Lmod's modules: the licensed tools did not start through the wrapper.

## [0.3.0] - 2026-10-31

### Added

- `export` pass: every phase of every model to StableHLO with component composites and symbolic sizes.
- Invariance certificate and recipe audit in the export's verification.
- `scripts/on-build-host` and `scripts/build-area` to build and run on the build host.
- Version, changelog and contribution guide; CI gates merges on `lint` and `test`.

### Changed

- Toolchain image `0.3.0` adds the locked JAX stack for the export.
- Pass cache keys include source fingerprints and the model configuration.
- Documentation reorganized under `docs/`.

## [0.2.0] - 2026-10-31

### Added

- Toolchain image with GraMi, Allo, XLS, Verilator, Yosys and SMT solvers, built under memory caps.
- `EdaRunner` for licensed Synopsys and Cadence tools through `eda-run`.
- `chipc doctor`: environment checks, licenses and Liberty `.db` compilation.
- GraMi, Allo and XLS as shallow submodules.

## [0.1.0] - 2026-10-31

### Added

- Quantized JAX ports of the 25-model workload set and their Python checks.
- Pass framework: configuration, artifact store, cache keys, capped tool runners.
- Pipeline manager: pass DAG, parallel scheduling with memory admission, resume, cancellation.
- `chipc` CLI and self-test passes.

[Unreleased]: https://github.com/lomolisso/chiplet-compiler/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/lomolisso/chiplet-compiler/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lomolisso/chiplet-compiler/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lomolisso/chiplet-compiler/releases/tag/v0.1.0
