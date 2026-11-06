# Cadence tools (2024-08 bundle) for chiplet-compiler

The industry flow is Design Compiler for synthesis and, in later versions, Innovus for place and route
(design §3.7). No other Cadence tool is part of a flow. Licensed host tools are launched only through
`EdaRunner` / `eda-run` (see `eda_environment.md`). Install root `/software/cadence-2024-08/`.

| Product dir | Tool | Status |
|---|---|---|
| `INNOVUS211` | `innovus` | place and route of the industry flow, later versions (`PnrFlow`); version not checked yet |
| `XCELIUM2403` | `xrun` | `xrun 24.03-s005` (checked 2026-10-30); `chipc doctor` records its version only |

## 1. Launching

Every call goes through `eda-run` in an interactive bash that inherits the session's Lmod environment
(`eda_environment.md` §2): the Cadence module (`cadence/2024-08`) prepends about 25 `tools.lnx86/lib`
directories (old libstdc++ and liblzma) to `LD_LIBRARY_PATH`, so it is never loaded into the driver's
environment. Each call runs in its own job directory (Cadence tools write their logs into the working
directory) with `TMPDIR` on the scratch disk.

## 2. Innovus (later versions)

What the place-and-route flow will need, from the kits' manifests (`pdks/<kit>.toml`, `pdks.md`): the
LEF files (ASAP7 ships 1x and 4x LEFs; the manifest names the 1x set), the Liberty timing libraries and
the synthesized netlist. A `PnrFlow` implementation is added when place and route enters the design.

## 3. Observed issues on this host

- A Cadence tool printed `Error : Unable to complete periodic license check succesfully` (2026-10-30):
  license reachability (`5280@cadence.webstore.illinois.edu`) may be flaky. The message is one of
  `[tools.eda] license_error_patterns`: `EdaRunner` retries with backoff and never records a license
  failure as a design result.
- Every Cadence tool prints `WARNING This OS does not appear to be a Cadence supported Linux configuration.`
  (harmless).
- The module file references `WXE2303`, but the install dir is `VXE2303` (harmless dangling PATH entry).

## Sources

- `xrun -version` (run 2026-10-30); `/etc/modulefiles/software/cadence/2024-08.lua`.
