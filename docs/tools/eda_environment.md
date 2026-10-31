# EDA environment on this host (eda-toggle.sh, eda-run, memory limits)

How chiplet-compiler's `EdaRunner` must launch the licensed Synopsys / Cadence tools.
All facts below were checked on the host on 2026-10-30 by reading the scripts and running
probes from a scratch directory (never from the repo).

## 1. The pieces

| File | Role |
|---|---|
| `~/bin/eda-toggle.sh` | Sourced from `~/.bashrc`. Defines `eda-on`, `eda-off`, `eda-status`, `eda-run`, `freepdk45-init`. |
| `~/pdks/setup_env.sh` | Part 1 (always): exports PDK path vars (see `pdks.md`). Part 2 (only if `[[ $- == *i* ]]`): `module load Synopsys_x86-64/2024` + `module load cadence/2024-08`, tool homes, `LD_LIBRARY_PATH` shim. |
| `/etc/modulefiles/software/Synopsys_x86-64/2024.lua` | Lmod module: prepends `27005@synopsys.webstore.illinois.edu` to `LM_LICENSE_FILE`; sets `VCS_HOME`, `VERDI_HOME`, `SYNOPSYS_LC_ROOT`; puts ~100 `…/W-2024.09/bin` dirs on `PATH`; does **not** set `SYNOPSYS`. |
| `/etc/modulefiles/software/cadence/2024-08.lua` | Lmod module: prepends `5280@cadence.webstore.illinois.edu` to `LM_LICENSE_FILE`; sets `CADENCE_CURR`; prepends `bin`, `tools.lnx86/bin` and `tools.lnx86/lib` of every product (GENUS211, CONFRML241, XCELIUM2403, INNOVUS211, INCISIVE152, …) to `PATH`/`LD_LIBRARY_PATH`. |

### What each function does (read from `~/bin/eda-toggle.sh`)

| Function | Behaviour |
|---|---|
| `eda-on` | Sources `setup_env.sh` into the **current** shell (tries `set -i` first if the shell is non-interactive), exports `_EDA_ACTIVE=1`. |
| `eda-off` | `module purge`; unsets tool-home vars (`CDSHOME MGC_HOME CNI_ROOT CNI_PLAT_ROOT VC_STATIC_HOME OA_UNSUPPORTED_PLAT MGC_CALIBRE_*`) and all PDK vars; strips `*/pdks/extra_libs/*`, `*/cadence-*`, `*/Synopsys-*` from `LD_LIBRARY_PATH` and FreePDK45 entries from `PYTHONPATH`. |
| `eda-status` | Prints ON if `_EDA_ACTIVE=1` or if `module list` shows Synopsys/cadence. |
| `eda-run CMD…` | `( case $- in *i*) ;; *) set -i ;; esac; . ~/pdks/setup_env.sh 2>/dev/null; exec "$@" )`: a subshell, so the caller's env is untouched; exit status is the tool's. With no args prints usage and returns 64. |
| `freepdk45-init [DIR]` | Copies Virtuoso/Calibre project files (`cds.lib`, `.cdsinit`, `lib.defs`, `.runset.calibre.*`) with `cp -n`. Not needed for our digital flow. |

Extra exports in the interactive section of `setup_env.sh`: `CDSHOME=/software/cadence-2024-08/IC231`,
`MGC_HOME`, `CNI_ROOT`, `CNI_PLAT_ROOT`, `VC_STATIC_HOME=/software/Synopsys-2024_x86_64/vc_static/W-2024.09`
(required by `vc_static_shell`), `OA_UNSUPPORTED_PLAT=linux_rhel70_gcc93x` (OpenAccess misparses RHEL 8.10),
`LD_LIBRARY_PATH=$EDA_PDK_ROOT/extra_libs/usr/lib64:…` (PulseAudio / xkbcommon libs), FreePDK45 CNI on
`PYTHONPATH`, and `MGC_CALIBRE_*_RUNSET_FILE`.

## 2. IMPORTANT: `eda-run` does not load the tools from a non-interactive shell

`set -i` is not a valid option in bash 4.4 (host has `bash-4.4.20-6.el8_10`):

```
$ bash probe.sh          # non-interactive
probe.sh: line 3: set: -i: invalid option
```

So in `bash -c '. ~/bin/eda-toggle.sh && eda-run dc_shell …'` the `$- == *i*` guard in `setup_env.sh` stays
false: only the PDK path vars are exported, **no modules are loaded, `dc_shell`/`genus` are not on `PATH`, and
`LM_LICENSE_FILE` is unset** (probe: `command -v dc_shell genus xrun` printed nothing). The same applies to
`eda-on` in a script. (Bash docs list `-i` as an invocation option only, not a `set` option:
https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html.)

Verified working form (probe on 2026-10-30, all eight tools resolved, `LM_LICENSE_FILE` and `VCS_HOME` set):

```bash
bash --norc -i -c '. "$HOME/bin/eda-toggle.sh"; eda-run "$@"' eda-run <tool> <args…>  </dev/null
```

- `-i` makes the shell interactive, so `eda-run`'s guard passes; `--norc` skips `~/.bashrc` side effects.
- Bash prints two harmless lines on stderr without a TTY: `cannot set terminal process group … Inappropriate
  ioctl` and `no job control in this shell`. `EdaRunner` should filter them from captured stderr.
- It needs the **inherited** environment of a normal login session: `MODULEPATH`, `SOFTPATH=/software` and
  Lmod's exported `module` function. Under `env -i` the probe got `MODULEPATH=unset` and no tools loaded.
  `EdaRunner` must therefore pass through the parent env (or explicitly set `MODULEPATH`, `SOFTPATH` and
  source `/usr/share/lmod/lmod/init/bash`).
- Alternative also verified: `bash -c '. /usr/share/lmod/lmod/init/bash && module load Synopsys_x86-64/2024 cadence/2024-08 && exec …'`.
  It skips the `setup_env.sh` extras (`extra_libs` on `LD_LIBRARY_PATH`, `VC_STATIC_HOME`,
  `OA_UNSUPPORTED_PLAT`), so prefer the `-i` form.
- Never run `eda-on` in the driver process: the Cadence module prepends ~25 `tools.lnx86/lib` dirs (old
  libstdc++/liblzma) to `LD_LIBRARY_PATH`, which breaks other C++ tools (comment in `eda-toggle.sh`; design §3.6).

## 3. EdaRunner contract (proposed)

| Concern | Rule |
|---|---|
| Launch | The `bash --norc -i -c … eda-run "$@"` form above, `stdin=/dev/null`, new process group (`setsid`). |
| CWD | A per-job dir under `build/.<pass>/eda/<job>/`. Tools drop files in CWD: Genus `genus.log`/`genus.cmd`, DC `command.log`/`default.svf`, Formality `formality.log`/`fm_shell_command.log`/`FM_WORK/`, VCS `csrc/`, `simv`, `simv.daidir/`, Xcelium `xrun.log`/`xcelium.d/`, LEC `lec.log` (typical names; (unverified) per version). Never run from the repo root. |
| Batch mode | Always a script + explicit exit: `dc_shell -f x.tcl` (ends with `quit`), `genus -batch -files x.tcl`, `lec -nogui -dofile x.do` (ends with `exit -force`), `fm_shell -f`, `pt_shell -f`, `sg_shell < x.tcl`. See `synopsys.md`, `cadence.md`. |
| TMPDIR | `/tmp` is only 8 GB on this host (`df -h /tmp`); Genus creates `/tmp/genus_temp_<pid>_…` (observed). `EdaRunner` exports `TMPDIR=<scratch_dir>/tmp/eda/<pass>-<n>-<tool>` (removed after a successful call); Genus nests its temp dir under it (verified 2026-10-30). |
| Timeouts | Wall-clock timeout per job; on expiry `kill -TERM -<pgid>`, wait, then `-KILL`. |
| Threads | Pass explicit caps in every script: DC `set_host_options -max_cores N` (default 1, max 16), Genus `set_db / .max_cpus_per_server N` (**default 8**), Formality `set_host_options -max_cores N` (max 8; >4 cores takes a 2nd license), VCS `-jN`. |
| Memory | Tool-native caps where they exist (Formality `set_host_options -max_memory <GB>`; DC `-minimize_peak_mem`), plus the RSS watchdog below. |
| Exit status | Do not trust exit codes alone: grep the log for `Error`/`ERROR` and the tool's license-failure messages. |

## 4. Licenses

- Synopsys: `27005@synopsys.webstore.illinois.edu`; Cadence: `5280@cadence.webstore.illinois.edu`; both via
  `LM_LICENSE_FILE` set by the Lmod modules (§1).
- Observed: `genus -help` printed `Error : Unable to complete periodic license check succesfully` (2026-10-30):
  Cadence license reachability may be flaky. `EdaRunner` should detect license errors and retry with backoff
  rather than mark the design as failed.
- Observed: every Cadence tool prints `WARNING This OS does not appear to be a Cadence supported Linux
  configuration.` (harmless).
- Observed: `dc_shell -version` without `extra_libs` on `LD_LIBRARY_PATH` fails with
  `libpulse-mainloop-glib.so.0: cannot open shared object file`, another reason to go through `eda-run`.
- `vcs` without `VCS_HOME` fails (`Cannot find 'vcsMsgReport' script`).
- Concurrent jobs each hold licenses; keep a small global semaphore (e.g. 2 Synopsys + 2 Cadence jobs) (sizing (unverified)).

## 5. Memory limiting on cgroups v1 (host tools)

Host facts: `/sys/fs/cgroup` is a v1 tmpfs hierarchy (`memory`, `cpu`, `pids` … controllers); Docker reports
`CgroupVersion 1`, driver `cgroupfs`; swap is full; ~89 GiB free / 209 GiB available at check time.

| Mechanism | Works? | Notes |
|---|---|---|
| `docker run --memory` | Yes | Only reliable hard cap here (see `docker_resources.md`). Not used for licensed tools (they live on the host under `/software`; containerizing them is (unverified)). |
| `ulimit -v` / `RLIMIT_AS` | Avoid | Limits *virtual* address space (setrlimit(2)). Multithreaded glibc programs create extra malloc arenas, up to a limit that is "a multiple of the number of available CPUs" (mallopt(3), `M_ARENA_MAX`/`M_ARENA_TEST`); each arena reserves a large address range (64 MiB per arena on 64-bit is from glibc source, (unverified) here). JVM/JIT and large mmaps also inflate VSZ. EDA tools then fail with spurious allocation errors or crash while RSS is small. |
| `RLIMIT_RSS` (`ulimit -m`) | No effect | Ignored by Linux since 2.4.30 (setrlimit(2)). |
| `systemd-run --user -p MemoryMax=` | Unreliable | `MemoryMax=` is a unified-hierarchy (v2) setting; v1 uses `MemoryLimit=`, and delegating the memory controller to an unprivileged user manager is not supported on v1 (systemd.resource-control(5)). Not tested here ((unverified)). |
| RSS watchdog | Yes | Recommended for host EDA jobs. |

RSS watchdog design (what `EdaRunner` implements):

1. Start the tool in its own process group (`setsid`).
2. Every 0.5–1 s, walk the group's PIDs (`/proc/<pid>/stat` field 5 = pgrp) and sum `VmRSS` from
   `/proc/<pid>/status` (or `Pss` from `/proc/<pid>/smaps_rollup` to avoid double-counting shared pages) (proc(5)).
3. If the sum exceeds the job budget, or `MemAvailable` in `/proc/meminfo` drops below a floor (swap is already
   full on this host), send `SIGTERM` to `-pgid`, then `SIGKILL` after a grace period; record `killed: memory` in
   the job result.
4. Check `free -g` / `MemAvailable` before launching (design §8).

## 6. Host-specific workarounds (2026-10-30)

| Symptom | Cause | Workaround |
|---|---|---|
| `lc_shell`: `symbol lookup error: /lib64/libkrb5.so.3: undefined symbol: krb5int_c_deprecated_enctype, version k5crypto_3_MIT` | LC W-2024.09 puts its bundled `libk5crypto.so.3` and `libcrypto.so.1.1` (`lc/W-2024.09/linux64/nwtn/shlib/`) first; the host's `krb5-libs` 1.18.2-34 needs the system pair | Fixed in the EDA environment, not in chipc: the wrapper `~/pdks/extra_bin/lc_shell` runs the real `lc_shell` with `LD_PRELOAD="/lib64/libcrypto.so.1.1 /lib64/libk5crypto.so.3"`. `setup_env.sh` puts `$EDA_PDK_ROOT/extra_bin` first on `PATH` (interactive part, after `extra_libs`), and `eda-off` strips it. So both `eda-on` shells and `eda-run` get it, and only `lc_shell` sees the preload. Preloading `libk5crypto` alone fails with `undefined symbol: EVP_KDF_ctrl, version OPENSSL_1_1_1b` |
| `vcs -ID`: `Cannot find VCS compiler … linux/bin/vcs1` | without `-full64` VCS looks for the 32-bit compiler | `vcs -full64 …` (always) |
| `dc_shell -version` exits 1 | prints the version, then exits 1 | judge by the `dc_shell version - W-2024.09` line |

## Sources

- `~/bin/eda-toggle.sh`, `~/pdks/setup_env.sh`, `~/pdks/README.md` (read 2026-10-30).
- `/etc/modulefiles/software/Synopsys_x86-64/2024.lua`, `/etc/modulefiles/software/cadence/2024-08.lua`.
- Probes `probe_seti.sh`, `probe_i.sh`, `probe_norc.sh`, `probe_final.sh` (run from a scratch dir, 2026-10-30).
- DC man page `set_host_options(2)`: `/software/Synopsys-2024_x86_64/syn/W-2024.09/doc/syn/man/cat2/set_host_options.2`;
  Formality `set_host_options(2)`: `/software/Synopsys-2024_x86_64/fm/W-2024.09/doc/fm/man/cat2/set_host_options.2`;
  DC `dc_shell(1)` (`-minimize_peak_mem`); Genus attribute ref `/software/cadence-2024-08/GENUS211/doc/genus_attref/synthesis.html` (`max_cpus_per_server`, default 8).
- setrlimit(2): https://man7.org/linux/man-pages/man2/setrlimit.2.html ; mallopt(3): https://man7.org/linux/man-pages/man3/mallopt.3.html ;
  proc(5): https://man7.org/linux/man-pages/man5/proc.5.html ;
  systemd.resource-control(5): https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html ;
  bash `set` builtin: https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html
