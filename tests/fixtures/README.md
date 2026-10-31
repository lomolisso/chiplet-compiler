# tests/fixtures/

Small artifacts shared by smoke and deep tests. The framework's fixtures are Rust code (`tests/src/lib.rs`:
`Fixture` passes, `TOY_CONFIG`); data fixtures come with the passes that read them.

| Path | Used by | Content |
|---|---|---|
| `liberty/` | smoke `env::doctor_dry_*` | header-only Liberty files: the Nangate45 name and the ASAP7 RVT TT group names (+ one LVT decoy) |
| `grami/two_paths.lg` | deep `env::grami_*` | two disjoint directed paths `1 -> 2 -> 3` (node labels), edge label 0, in GraMi's `.lg` format |
| `xls/add8.cc` | deep `env::xls_*` | 4-line xlscc function: 8-bit unsigned add with a 9-bit result |
| `export/` | smoke `export::*` | a real export of the toy Flax port (`python/tests/toys.py`) and its facts; regenerate with `make.py` |
