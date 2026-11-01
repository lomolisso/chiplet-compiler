# Documentation

| Document | Contents |
|---|---|
| [design.md](design.md) | The system design: passes, verification, datatypes, resources, decision log |
| [development.md](development.md) | Code conventions, pass anatomy, artifacts, configuration, build host and scratch, tests |
| [testing.md](testing.md) | The test crate: smoke and deep tests, fixtures, how to run them |
| [framework/core.md](framework/core.md) | `cc-core`: configuration, artifact store, cache keys, tool runners |
| [framework/pipeline.md](framework/pipeline.md) | `cc-pipeline`: pass contract, DAG, scheduler, admission, resume; how to add a pass |
| [framework/environment.md](framework/environment.md) | `chipc doctor` and the Liberty `.db` files |
| [framework/cli.md](framework/cli.md) | `chipc` commands |
| [framework/ir.md](framework/ir.md) | `cc-ir`: the IR, symbolic dimensions, the op-semantics table, graphs for matching |
| [framework/smt.md](framework/smt.md) | `cc-smt`: SMT-LIB scripts and the solver runner |
| [passes/](passes/) | One document per pass: `export`, `export-frontend` (the Python side), `import`, `components`, `normalize`, `selftest` |
| [toolchain-image.md](toolchain-image.md) | The Docker toolchain image: build, pins, cache, measurements |
| [third-party.md](third-party.md) | The submodules (GraMi, Allo, XLS) and their pins |
| [model-verification.md](model-verification.md) | Python checks of the model ports (`verify/`) |
| [tools/](tools/README.md) | Reference notes on the external tools the passes call |
| [schemas/](schemas/README.md) | JSON Schemas of every artifact kind |
