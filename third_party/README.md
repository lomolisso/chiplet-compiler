# third_party/

Git submodules (shallow), added and pinned by T02. Their versions are recorded in `config.toml` `[tools.*]` and checked by `chipc doctor`.

| Submodule | Upstream | Used by |
|---|---|---|
| `grami/` | https://github.com/ehab-abdelhamid/GraMi | `mine` (directed build) |
| `allo/` | https://github.com/cornell-zhang/allo | `softfloat`, `allo_emit`, `xls` |
| `xls/` | https://github.com/google/xls | `xls`, `characterize` (source for the Bazel-built xlscc/eval tools; release binaries for the rest) |

Tools are built inside the Docker image (`docker/`), never on the host.
