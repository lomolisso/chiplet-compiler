#!/bin/bash
# Build the toolchain image. Usage, from the repo root:  docker/build.sh
# Every pin and cap comes from config.toml via `chipc doctor --build-args` (override the binary
# with CHIPC=...). Long build (hours): run it in the background; see docker/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

chipc=(${CHIPC:-cargo run -q -p cc-cli --})
declare -A arg
while IFS='=' read -r k v; do arg[$k]="$v"; done < <("${chipc[@]}" doctor --build-args)

# The image copies the submodules, so they must sit at the pinned commits.
check() { # <path> <expected sha> <what>
  local have
  have="$(git -C "$1" rev-parse HEAD)"
  [[ "$have" == "$2" ]] || { echo "build.sh: $3 is at $have, config pins $2" >&2; exit 1; }
}
check third_party/xls "${arg[XLS_COMMIT]}" third_party/xls
check third_party/allo "${arg[ALLO_COMMIT]}" third_party/allo
check third_party/grami "${arg[GRAMI_COMMIT]}" third_party/grami
llvm="$(git -C third_party/allo ls-tree HEAD externals/llvm-project | awk '{print $3}')"
[[ "$llvm" == "${arg[LLVM_COMMIT]}" ]] || { echo "build.sh: Allo records LLVM $llvm, config pins ${arg[LLVM_COMMIT]}" >&2; exit 1; }

# Shared machine: the cap must fit in what is free now, with headroom (spec §8).
mem="${arg[CAP_MEMORY_GB]}"
avail_gb=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
if (( avail_gb < mem + 16 )); then
  echo "build.sh: MemAvailable ${avail_gb} GB < cap ${mem} GB + 16 GB headroom; not starting" >&2
  exit 1
fi

logdir="${SCRATCH_DIR:-/srv/scratch/$USER}/chipc"
mkdir -p "$logdir"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$logdir/docker-build-$stamp.log"
peak="$logdir/docker-build-$stamp.peak"

# Peak memory of the build's RUN containers (cgroup v1). Ours: the legacy builder runs each RUN step
# in a container of an intermediate `sha256:` image, with our --memory as the limit (other users'
# containers on the shared daemon can share the limit, not both). Records the peak resident memory
# (memory.stat total_rss, sampled) and the cgroup peak (max_usage_in_bytes, includes page cache).
cap_bytes=$(( mem * 1024 * 1024 * 1024 ))
sample() {
  local rss_max=0 use_max=0 id image g r u
  while sleep 15; do
    while read -r id image; do
      [[ "$image" == sha256:* ]] || continue
      g="/sys/fs/cgroup/memory/docker/$id"
      [[ "$(cat "$g/memory.limit_in_bytes" 2>/dev/null)" == "$cap_bytes" ]] || continue
      r="$(awk '$1 == "total_rss" {print $2}' "$g/memory.stat" 2>/dev/null)"
      u="$(cat "$g/memory.max_usage_in_bytes" 2>/dev/null || echo 0)"
      (( ${r:-0} > rss_max )) && rss_max=$r
      (( u > use_max )) && use_max=$u
    done < <(docker ps --no-trunc --format '{{.ID}} {{.Image}}')
    echo "$rss_max $use_max" > "$peak"
  done
}
sample & sampler=$!
trap 'kill $sampler 2>/dev/null || true' EXIT

build_args=()
for k in "${!arg[@]}"; do
  [[ "$k" == CAP_* ]] || build_args+=(--build-arg "$k=${arg[$k]}")
done

echo "build.sh: ${arg[CAP_IMAGE]}  cap ${mem} GB / ${arg[CAP_CPUS]} CPUs  log $log"
start=$(date +%s)
# DOCKER_BUILDKIT=0: this host has no buildx; the legacy builder applies --memory to every RUN.
# `|| status=$?`: under `set -e` a failed build would otherwise end the script before the summary.
status=0
DOCKER_BUILDKIT=0 docker build --force-rm \
  --memory "${mem}g" --memory-swap "${mem}g" \
  --cpu-period 100000 --cpu-quota $(( arg[CAP_CPUS] * 100000 )) \
  "${build_args[@]}" -t "${arg[CAP_IMAGE]}" -f docker/Dockerfile . 2>&1 | tee "$log" || status=$?
end=$(date +%s)
peaks="$(awk '{printf "RSS %.1f GiB, cgroup incl. page cache %.1f GiB", $1/2^30, $2/2^30}' "$peak" 2>/dev/null || echo "?")"
echo "build.sh: exit $status after $(( (end - start) / 60 )) min, peak RUN-step memory: $peaks" | tee -a "$log"
exit "$status"
