# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
