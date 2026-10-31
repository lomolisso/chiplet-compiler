# Toolchain image

The toolchain image (`docker/`). It holds every open-source tool the pipeline calls; `DockerRunner` starts one capped container per call. The licensed EDA tools are not in the image: they run on the host through `eda-run` (`docs/tools/eda_environment.md`).

| File | Role |
|---|---|
| `Dockerfile` | multi-stage build, legacy-builder syntax (this host has no buildx) |
| `build.sh` | the only supported way to build it: pins from `config.toml`, hard caps, log, peak-memory sampling |
| `cc-probe.py` | `/opt/cc/bin/cc-probe`: every tool's version as JSON; at build time it also writes `/opt/cc/versions.json` with the pins |
| `selftest.sh` | `/opt/cc/bin/cc-selftest [DIR]`: each tool does one tiny real job; gates the build and backs `chipc doctor` |
| `.hadolint.yaml` | lint config (CI runs hadolint v2.15.1) |

## Build

Everything here runs on the build host (docs/development.md, "Build host and scratch"):

```bash
scripts/on-build-host --status                 # load, memory, free /scratch (build.sh refuses below 100 GB free on /srv)
scripts/on-build-host docker/build.sh          # from the repo root; long: run it in the background
# scripts/on-build-host env CHIPC=chipc docker/build.sh   # use the built chipc instead of `cargo run`
scripts/on-build-host chipc doctor             # then check it (image.pins, image.tools, image.selftest.*)
```

- **Another host** gets a copy streamed, not rebuilt: `docker save chiplet-compiler/toolchain:<tag> | ssh <host> docker load`. The build-stage cache is not copied, so the next image change on that host rebuilds every stage once (~75 min).
- **Old tags** are removed by name (`docker rmi chiplet-compiler/toolchain:<old>`) once a new tag is verified; never prune globally.

- **Tag.** The image is `chiplet-compiler/toolchain:<tools.docker.tag>`, currently `0.3.0`: the release in which the image inputs last changed. Bump it (to the upcoming release's version) whenever `docker/`, `[tools.docker.build]`, `[tools.xls]`, `[tools.allo]` or `[tools.grami]` changes.
- **Pins.** `build.sh` gets every pin from `chipc doctor --build-args`, so `config.toml` is the only place they are written. It also checks that `third_party/` sits at the pinned commits.

| Pinned | How |
|---|---|
| base images | `ubuntu:24.04@sha256:…` (final), `ubuntu:22.04@sha256:…` (XLS Bazel stage) |
| apt packages | every `apt-get` reads one `snapshot.ubuntu.com` date (`apt_snapshot`). `ca-certificates` comes first from the regular archive, because the snapshot service is https-only. `--error-on=any` turns a failed index download into a build failure |
| PyPI | `UV_EXCLUDE_NEWER=pypi_exclude_newer`: uv resolves as of that date |
| downloads | uv, bazelisk, the XLS release tarball, z3, bitwuzla: version + sha256 |
| sources | XLS, Allo and GraMi from the submodules; LLVM fetched at Allo's gitlink inside the build |

- **Caps.** `docker build --memory = --memory-swap = tools.docker.build.memory_gb` (64 GB) and `--cpu-quota` = 16 CPUs. These apply to every RUN step and are hard limits: the kernel OOM-kills inside the step's cgroup. Inside the steps:
  - `ninja -j16` with `LLVM_PARALLEL_LINK_JOBS=2`;
  - Bazel `--jobs=16 --local_resources=cpu=16 --local_resources=memory=49152 --host_jvm_args=-Xmx4g`, which is explicit because Bazel would otherwise size itself from the host's 64 CPUs and 502 GiB.

## Stages

| Stage | Base | Contents |
|---|---|---|
| `base` | 24.04 | apt snapshot, Python 3.12, uv |
| `build` | base | compilers, cmake, ninja, git |
| `xls-release` | base | XLS release binaries (`codegen_main`, `opt_main`, `ir_converter_main`, `interpreter_main`, `prove_quickcheck_main`, DSLX tools, stdlib) |
| `solvers` | base | z3 (glibc-2.39 build) and bitwuzla. The "static" bitwuzla binary still needs `libgmp10` and `libmpfr6` |
| `grami` | base + JDK 17 | GraMi with `third_party/patches/grami/*.patch` applied; both `*_SUBGRAPHS` trees compiled |
| `llvm-allo` | build | LLVM `6b09f739` (clang, mlir, openmp; cmake flags from Allo's setup docs) and Allo installed editable into `/opt/cc/venv` |
| `xls-bazel` | 22.04 | `xlscc`, `eval_ir_main` and `check_ir_equivalence_main` from the same XLS commit; `synth_only` + `ac_datatypes` headers. Built as a non-root user, as upstream does |
| `final` | base | runtime libraries, JRE 17, yosys / iverilog / verilator, a compiler (Verilator needs one), and the copied stage outputs. `cc-selftest` runs, then `cc-probe` writes `versions.json` |

Layout and environment of the final image:
- `PATH=/opt/cc/bin:/opt/cc/venv/bin:/opt/xls/bin:/opt/solvers/bin:/opt/llvm-project/build/bin:…`
- `LLVM_BUILD_DIR=/opt/llvm-project/build`: Allo's ExecutionEngine loads `libmlir_*runner_utils.so` from there.
- `XLSCC_INCLUDE_DIRS=/opt/xls/include,/opt/xls/include/xls/contrib/xlscc/synth_only` and `XLSCC_DEFINES=__SYNTHESIS__,__xlscc__,__AC_OVERRIDE_OVF_UPDATE_BODY=,__AC_OVERRIDE_OVF_UPDATE2_BODY=`: pass both to every `xlscc` call (`--include_dirs`, `--defines`), as XLS's own `xls_cc_ir` rule does (`xls/contrib/xlscc/build_rules/xlscc_rules.bzl`).
- `XLS_STDLIB=/opt/xls/share/xls/dslx/stdlib`
- `GRAMI_DIRECTED_CP`, `GRAMI_UNDIRECTED_CP`
- `HOME=/tmp/home` (mode 1777): containers run as the host uid.

## Layer cache, disk, cleanup

- **Disk.** Layers and the build cache live in Docker's root, `/srv/docker` (local disk). Nothing is built on `/home` (NFS) or `/tmp` (8 GB). The context is only `docker/` and `third_party/` (`.dockerignore`).
- **Rebuilds.** They reuse every unchanged layer, so editing `selftest.sh` or `cc-probe.py` rebuilds only the end of the `final` stage. Changing a pin rebuilds from the stage that uses it.
- **Cleanup.** The daemon is shared, so remove only our dangling stage images:
  ```bash
  docker image prune --filter label=chiplet-compiler.stage --force
  ```
  Never run `docker system prune`.

## Measured (this host, 2026-10-30/31; caps 64 GB / 16 CPUs)

| | |
|---|---|
| Build time with an empty cache | about 75 min: base and downloads ~5 min; LLVM `6b09f739` + Allo ~38 min; XLS Bazel build (15,519 actions) 25–27 min (`Elapsed time` 1522–1594 s in three runs); final stage + `cc-selftest` ~2 min |
| Peak memory of a RUN step | LLVM: 23.7 GiB resident (27.0 GiB cgroup incl. page cache); Bazel: 7.6 GiB resident (19.4 GiB incl. cache). Well under the 64 GB cap; no step hit it (`memory.failcnt` 0) |
| Final image size | 2.78 GB (the 0.2.0 image, `sha256:cb662455…`; 0.3.0 adds the JAX venv) |
| Stage images kept for the cache | 8, the largest 12–15 GB each (`llvm-allo`, `xls-bazel`; layers are shared) on `/srv/docker` |

The image was first built across several invocations of `docker/build.sh` on 2026-10-31. Each invocation fixed one step and resumed from the layer cache, so every layer above was built by this command from an empty cache. The time above is the sum of the runs.

The three Bazel-built tools (`xlscc`, `eval_ir_main`, `check_ir_equivalence_main`) print `redacted` for `--version` (unstamped build). Their commit is the submodule pin, which `versions.json` records.

The build log is at `/scratch/$USER/chiplet-compiler/logs/docker-build-<UTC stamp>.log` on the build host.
