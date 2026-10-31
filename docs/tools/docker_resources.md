# Docker on this host: resource caps, permissions, image plan

All open-source heavy tools (XLS, xlscc, Allo + LLVM, GraMi, Yosys, Verilator, Icarus, z3, bitwuzla) run in our
toolchain image under hard memory/CPU caps. Licensed EDA tools stay on the host (`eda_environment.md`).

## 1. Host facts (checked 2026-10-30)

| Item | Value |
|---|---|
| Docker | Server 26.1.3, storage driver `overlay2`, **Cgroup Version 1**, cgroup driver `cgroupfs`, root dir `/srv/docker` (`docker info`) |
| Daemon config | `/etc/docker/daemon.json`: only `bip` and `default-address-pools`; no `userns-remap`, so container root = host root on bind mounts |
| Swap accounting | `memory.memsw.limit_in_bytes` exists under `/sys/fs/cgroup/memory/` → `--memory-swap` is enforceable; `docker info` shows no swap-limit warning |
| Access | user is in group `docker` (uid/gid 1291371). Docker group access is root-equivalent and the daemon/images are shared with other users of the machine |
| CPU / RAM | 64 CPUs, 502.9 GiB, swap (7 GiB) full |
| Disks | `/srv` local 796 G (463 G free) holds Docker; `/home/rid2` NFS 97 % used (315 G free); `/sim` NFS 3.9 T free; `/tmp` local **8 G** |
| Existing images | several multi-GB images from other projects already in `/srv/docker`; do not prune shared images |

The table above is the orchestration host, iacoma227 (2026-10-30). Since 2026-10-31 every build and container runs on
the **build host, iacoma231** (docs/development.md, "Build host and scratch"), checked on 2026-10-31: 96 CPUs,
376 GiB; Docker 26.1.3, cgroup v1, root dir `/srv/docker`; user `rid2` in `docker`; `/scratch` = `/srv/scratch` on the
1.7 T `/srv` disk shared with Docker (998 G free); same NFS home, `/software` and `~/pdks` as iacoma227.

## 2. Resource flags (`docker run`)

| Flag | Semantics (Docker docs) | Our default |
|---|---|---|
| `-m, --memory` | "The maximum amount of memory the container can use." Hard limit, min 6m. | per job, e.g. `16g`; never above `MemAvailable`/2 |
| `--memory-swap` | Total memory **+ swap**. Equal to `--memory` → no swap. Unset → may use swap equal to `--memory`. `-1` → unlimited swap. | **always = `--memory`** (host swap is full) |
| `--memory-swappiness` | 0 disables anonymous-page swapping. | `0` |
| `--memory-reservation` | soft limit under contention; no guarantee | unused |
| `--oom-kill-disable` | disables OOM killer; only with `-m` | never |
| `--cpus` | CFS quota, e.g. `1.5` = at most 1.5 CPUs (= `--cpu-period=100000 --cpu-quota=150000`) | per job, e.g. `8` |
| `--cpuset-cpus` | pin to listed CPUs | optional, avoid on a shared box |
| `--pids-limit` | "Tune container pids limit (set -1 for unlimited)" | `4096` (stops fork bombs from runaway builds) |
| `--init` | runs `docker-init` as PID 1 that forwards signals and reaps processes | always |
| `--rm` | remove container (and anonymous volumes) on exit | always |
| `--shm-size` | size of `/dev/shm` (default 64 MB (unverified)) | raise only if a tool needs it |
| `--ulimit` | set rlimits inside the container | `--ulimit core=0` |

OOM behaviour: when a container exceeds `--memory` (+swap), the kernel OOM killer kills processes **inside that
container's memory cgroup** (cgroup-v1 memory controller: limit hit → reclaim → OOM within the group). The main
process typically exits with 137 (SIGKILL) and `docker inspect -f '{{.State.OOMKilled}}' <id>` is `true`. Use
named containers (`--name chipc-<pass>-<job>`) without `--rm` only when you need to inspect `OOMKilled`; otherwise
read the exit code. Docker lowers the daemon's own OOM priority but not containers' (Docker docs).

Inside a container on cgroups v1, `/proc/meminfo` and `nproc` show **host** values (64 CPUs, 502 GiB) unless
`--cpuset-cpus` is used ((unverified) for nproc with `--cpus`; it is a quota, not a CPU mask). Tools that size
themselves from host resources (Bazel `HOST_RAM`/`HOST_CPUS`, `make -j$(nproc)`, `ninja` default jobs, rayon,
JVM default heap) will oversubscribe. Always pass explicit job counts:

| Tool | Flags |
|---|---|
| Bazel | `--jobs=N --local_ram_resources=<MB> --local_cpu_resources=N` (absolute numbers, not `HOST_RAM*…`); startup `--host_jvm_args=-Xmx4g`; Bazel ≥7 also has `--local_resources=memory=…,cpu=…` ((unverified) exact form) |
| ninja / cmake (LLVM) | `ninja -j N`; `-DLLVM_PARALLEL_LINK_JOBS=1..2` (links are the RAM peak) |
| make | `make -j N` |
| Java (GraMi) | `java -Xmx<N>g -XX:ActiveProcessorCount=N` |
| Rust in container | `RAYON_NUM_THREADS=N`, `cargo build -j N` |

## 3. Permissions and mounts

- No user-namespace remapping → files written by a root container into a bind mount are owned by host root.
  Run every job as the invoking user: `--user "$(id -u):$(id -g)"`. Images must not assume `$HOME` exists for
  that uid: set `-e HOME=/tmp/home` (or create it in the entrypoint) and keep caches (`UV_CACHE_DIR`,
  `XDG_CACHE_HOME`, Bazel output base) on mounted, writable paths.
- `/home/rid2` is NFS with root-squash in effect (observed 2026-10-30: the root daemon got `stat …: permission
  denied` binding a mode-700 directory under `/home/rid2`). A bind-mount source on `/home` must therefore be
  traversable by others (`o+x` on every parent; the repo path is: `/home/rid2` is 711, the rest 775), and nothing
  may be written there as root.
- Bind mounts: `-v`/`--mount type=bind`. With `-v`, a missing host dir is auto-created (as root); with `--mount`,
  a missing path is an error (Docker bind-mount docs). Use `--mount`, pre-create dirs from Rust.
- Standard mounts for a pass job:
  - repo read-only: `--mount type=bind,src=<repo>,dst=/work/repo,readonly`
  - pass output dir read-write: `--mount type=bind,src=<repo>/build/.<pass>,dst=/work/out`
  - `--workdir /work/out`, `--network none` (tools need no network at run time).
- Images are built once; jobs never `apt`/`pip install` at run time.

## 4. Image layering (design notes; as built: `docs/toolchain-image.md`)

Base: Ubuntu 22.04 or 24.04 (as built: 24.04 for the final image, 22.04 for the XLS Bazel stage). The XLS release tarball is built on Ubuntu 22.04 and
needs a newer glibc than RHEL 8's 2.28, the reason these tools cannot run on the host. 24.04 ships Python 3.12,
which Allo requires; on 22.04 use a uv-managed 3.12.

| Layer | Contents | Build-time caps / notes |
|---|---|---|
| 0 base | `ubuntu:<ver>` + build-essential, git, cmake, ninja, curl, ca-certs | |
| 1 python | `uv` (pinned) + Python 3.12 via `uv python install` | `UV_CACHE_DIR` in layer, then cleaned |
| 2 java | `openjdk-17-jdk-headless` (GraMi; see `grami.md` for the Java-version caveat) | |
| 3 xls-release | pinned XLS release tarball (`codegen_main`, `opt_main`, `ir_converter_main`, …), checksum-verified | download only |
| 4 xls-bazel | Bazel (via pinned bazelisk) build of `xlscc`, `eval_ir_main`, `eval_proc_main`, … at the **same commit** as the release | `--jobs` and RAM caps above; `docker build --memory` is supported only by the legacy builder; with BuildKit run the build inside a capped `docker run` and `docker commit`, or use a capped buildx builder ((unverified) which works best) |
| 5 llvm-allo | LLVM at Allo's pinned commit with MLIR + Python bindings, then `pip install` Allo | `ninja -j8`, `LLVM_PARALLEL_LINK_JOBS=2`; expect tens of GB peak at high -j (unverified numbers) |
| 6 eda-oss | yosys, iverilog, verilator, z3, bitwuzla (distro packages or pinned release binaries) | |
| 7 grami | GraMi built from pinned commit (`third_party/`), jar on `PATH` | |

Order layers from least to most frequently changed; the two expensive builds (4, 5) should be separate stages
whose outputs are copied into the final image (multi-stage build) so the final image omits Bazel/LLVM build trees.

## 5. Disk notes

- Everything below happens on the build host (iacoma231). Image layers and the legacy builder's stage images
  land in its `/srv/docker`, on the same disk as `/scratch`: `docker/build.sh` refuses to start below 100 GB free.
- Do not put build trees on `/home` (NFS, 97 % full) or `/tmp` (8 G). All build output goes to the build area
  `/scratch/$USER/chiplet-compiler/` (`target/`, `venv/`, `build/`, `logs/`, ephemeral `runs/` and `tmp/`;
  docs/development.md, "Build host and scratch"), including `paths.build_dir`, so pass outputs never touch NFS.
- Clean up: ephemeral runs when they have served their purpose; old image tags by name
  (`docker rmi chiplet-compiler/toolchain:<old>`). Never `docker system|image|builder prune` on the shared daemon.

## 6. Canonical job command

```bash
docker run --rm --init --name chipc-xls-<job> \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/home \
  --memory 16g --memory-swap 16g --memory-swappiness 0 \
  --cpus 8 --pids-limit 4096 --ulimit core=0 --network none \
  --mount type=bind,src="$REPO",dst=/work/repo,readonly \
  --mount type=bind,src="$REPO/build/.xls",dst=/work/out \
  --workdir /work/out \
  chiplet-toolchain:<tag> codegen_main --generator=pipeline … 
```
`ToolRunner` adds a wall-clock timeout (`docker kill` on expiry) and records exit code, `OOMKilled`, peak memory
(`memory.max_usage_in_bytes` of the container cgroup on v1, or `docker stats --no-stream` sampling).

## Sources

- Docker resource constraints: https://docs.docker.com/engine/containers/resource_constraints/
- `docker run` reference (`--pids-limit`, `--user`, `--init`, `--rm`, `-v` auto-create): https://docs.docker.com/reference/cli/docker/container/run/
- Bind mounts (`--mount` vs `-v` on missing paths): https://docs.docker.com/engine/storage/bind-mounts/
- Kernel cgroup-v1 memory controller: https://docs.kernel.org/admin-guide/cgroup-v1/memory.html
- Bazel flags `--local_ram_resources`/`--local_cpu_resources` (replace removed `--local_resources`; `HOST_RAM*.5` syntax): https://bazel.googlesource.com/bazel/+/219ef5d2f93d331653a1df0eadda1f49131625bc ; startup options `--host_jvm_args`, `--output_base`: https://bazel.build/reference/command-line-reference
- Host: `docker info`, `/etc/docker/daemon.json`, `/sys/fs/cgroup/memory/`, `df -h`, `free -g`, `id` (2026-10-30).
