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
- `iteration_space` pass and the op table's iteration semantics: per class its affine accesses proven against the spec, dependences, the embedding relation, and routing of uniform contractions to systolic templates by a proven space-time mapping.

### Changed

- `export` records each symbol's integer size at the exported config and a census of the phase lowered on concrete shapes (`census_concrete.json`, integer FLOPs).
- `scripts/on-build-host --pick` chooses among iacoma227, iacoma231 and iacoma232 by their scarcest resource (idle cores, RAM, disk); both scripts run directly on the local host.

### Fixed

- JSON floats are parsed exactly (`serde_json` `float_roundtrip`); `import` read some float attributes one ulp off.
- `components` inlines a custom call's reducer computation as its region, so `dynamic_reduce_window` kernels carry their combiner and classes tell reducers apart.

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
