# Cadence tools (2024-08 bundle) for chiplet-compiler

Licensed host tools, launched only through `EdaRunner` / `eda-run` (see `eda_environment.md`). Install root
`/software/cadence-2024-08/`. Versions checked on 2026-10-30 (run from a scratch dir):

| Product dir | Tool | Version printed |
|---|---|---|
| `GENUS211` | `genus` | `Genus(TM) Synthesis Solution, Version: 21.19-s055_1` (built May 29 2024) |
| `CONFRML241` | `lec` | `LEC version 24.10-p100 (30-May-2024)` |
| `XCELIUM2403` | `xrun` | `xrun 24.03-s005` |
| `INNOVUS211` | `innovus` | not checked (out of scope) |
| `STRATUS2401` | Stratus HLS | not used |

JasperGold (`jg`) is not installed. Local documentation: `GENUS211/doc/genus_comref/*.html` (commands),
`GENUS211/doc/genus_attref/*.html` (attributes), `CONFRML241/doc/conformal.hlp` (plain-text LEC command help),
`xrun -helpall`.

## 1. What we use each tool for

| Tool | Our use |
|---|---|
| Genus | Second synthesis engine (cross-check DC area/timing); reads `.lib` directly, so no `.db` step. |
| Conformal LEC | Equivalence RTL ↔ Genus netlist (and RTL ↔ DC netlist as an independent check of Formality). |
| Xcelium | Second RTL simulator (cross-check VCS), gate-level sim with `stdcells.v` / ASAP7 Verilog models. |
| Innovus | Out of scope (no place & route). Note only: needs LEF; ASAP7 ships 1x and 4x LEFs (`pdks.md`). |

## 2. Genus

Batch invocation (from `genus -help`): `genus -batch -files syn.tcl -log genus -overwrite -no_gui`
- `-batch`: exit after processing files; `-files <f>`: command file; `-execute "<cmds>"`: run before `-files`;
  `-abort_on_error`: exit on script error (use it); `-log <prefix>`: log/cmd file prefix; `-overwrite`: overwrite
  default logfiles; `-disable_user_startup`: only read master init file (use it for reproducibility).
- Genus prints `TMPDIR is being set to /tmp/genus_temp_<pid>_<host>_<user>_XXXX` (observed). `/tmp` is 8 GB here;
  give each job a `TMPDIR` on `/srv` or `/sim`. Genus nests its temp dir under a caller-set `TMPDIR` (verified
  2026-10-30: `TMPDIR is being set to <TMPDIR>/genus_temp_…`; `chipc doctor` check `license.genus`).
- `genus -batch -execute "…; exit"` prints `Checking out license: Genus_Synthesis` on checkout (observed).
- `lec -XL … -dofile` with only `exit -force` exits with status 2 (no comparison done) and prints
  `// Check out Conformal_Ultra 24.1 license` / `// Check out Conformal_Asic 24.1 license` (observed).
- Without `DISPLAY` Genus prints "GUI mode requires a valid DISPLAY. Reverting to shell mode." (harmless); pass `-no_gui`.

Script template (commands from `genus_comref`, attributes from `genus_attref`):

```tcl
set_db / .max_cpus_per_server 4          ;# default 8 (!); always set explicitly
set_db / .auto_super_thread false        ;# no extra super-threading servers (attribute exists; effect (unverified))
read_libs /abs/stdcells.lib              ;# ASAP7: list all 5 group .libs of one VT/corner
read_hdl -language sv fu.sv              ;# read_hdl file_list [-language {v2001|v1995|sv|vhdl}]
elaborate fu
check_design -unresolved > check.rpt     ;# option name (unverified)
create_clock -name clk -period $PERIOD [get_ports clk]   ;# lib time unit; no source => virtual clock
set_input_delay  0 -clock clk [all_inputs]
set_output_delay 0 -clock clk [all_outputs]
syn_generic
syn_map
syn_opt
report_qor    > qor.rpt
report_area   > area.rpt
report_timing -max_paths 10 > timing.rpt
report_gates  > gates.rpt
write_hdl > fu_genus.v
write_sdc > fu_genus.sdc
write_do_lec -golden_design fu.sv -revised_design fu_genus.v -log_file lec.log > lec.do
exit
```

Command semantics (`genus_comref/synthesis.html`):
- `syn_generic`: elaborated, constrained design → generic gates; datapath optimization (adders, multipliers,
  shifters). `syn_map`: generic → technology library cells, then incremental area/power recovery.
  `syn_opt`: incremental timing/area/power optimisation of a mapped design. Effort levels via the
  `syn_generic_effort`, `syn_map_effort`, `syn_opt_effort` root attributes.
- `read_libs [filenames]` loads Liberty; searches `lib_search_path` if not found.
- `elaborate [module]`: without a module, every uninstantiated module becomes a top.
- `create_clock` with `-name` and no source creates a virtual clock; use it for XLS `--generator=combinational` output.
- `report_qor`: critical-path slack, TNS, gates on critical path, instance count, total/cell area, runtime, host.
- `report_area [-depth N] [-summary]`: per-hierarchy cell counts and areas.
- `write_do_lec`: writes a Conformal dofile for golden RTL vs revised netlist, a convenient start for our LEC jobs,
  but generate our own dofile for the DC netlist.

Resource attributes (`genus_attref/synthesis.html`): `max_cpus_per_server`: "maximum number of active CPUs Genus
is allowed to use per server for super-threading and multi-threading", **default 8**, affects `syn_generic`,
`syn_map`, `syn_opt`. `super_thread_servers` lists extra hosts (leave unset). No native memory cap found in the
attribute reference ((unverified) beyond a search for "memory" in `max_cpus_per_server` context); rely on the
`EdaRunner` RSS watchdog.

## 3. Conformal LEC

Invocation (from `lec -help`): `lec -XL -nogui -dofile lec.do -logfile lec.log -nobanner -nolicwait`

| Flag | Meaning |
|---|---|
| `-Dofile f` / `-LOGfile f` | run script / log file |
| `-NOGui` | no GUI |
| `-L` / `-XL` / `-GXL` | license tier: `-L` = LEC + Verilog + VHDL; **`-XL` = L + datapath + LVR**, needed for multiplier-heavy FUs (`analyze datapath` requires XL); `-GXL` adds memory + LTX (`flatten` requires GXL) |
| `-NOLIcwait` | exit immediately if licenses are busy instead of waiting (use it; retry in `EdaRunner`) |
| `-TCLmode` | Tcl command mode (default is the Conformal command language) |
| `-NODEFault` | skip `.conformal_lec` init file |

Dofile template (command language; syntax from `conformal.hlp`):

```
set log file lec.log -replace
read library -liberty /abs/stdcells.lib -both
read design -systemverilog fu.sv -golden
read design -verilog fu_netlist.v -revised
set system mode lec
add compared points -all
analyze datapath -verbose          // datapath (multiplier) learning; XL license
set parallel option -threads 4     // or: compare -threads 4
compare
report verification
report compare data -class nonequivalent
report unmapped points -notmapped
exit -force
```

- `READ LIBRARY <files> [-Liberty | -VErilog …] [-Both | -Golden | -Revised]` (Setup mode).
- `set system mode lec` maps key points; `compare` proves them.
- Exit status (`EXIT` help): "A nonzero status code means there is a potential error; that is, either no
  comparison was done or unmapped, abort, or nonequivalent points exist." → `EdaRunner` can use the exit code,
  and should still parse `report verification` for the counts.
- Parallel compare consumes extra licenses (`set parallel option -license …`; `compare -threads N` "checks out
  additional … licenses"); keep threads ≤ 4.
- Hard multipliers in large FUs may abort; the documented remedies are `analyze datapath` and hierarchical compare
  (`analyze hier_compare`, XL). Record aborts as "inconclusive", never as pass.

## 4. Xcelium

```
xrun -64bit -sv -timescale 1ns/1ps -access +r -l xrun.log -xmlibdirname ./xcelium.d -exit tb.sv fu.v
```
From `xrun -helpall`: `-64bit`, `-access <+/-rwc>` (debug visibility; omit for speed), `-l <file>`, `-clean`
(delete previous `xcelium.d`), `-elaborate` (compile/elaborate only), `-exit` (exit instead of Tcl prompt).
`-sv`, `-timescale`, `-xmlibdirname` are standard xrun options ((unverified) against `-helpall` output).
For gate-level sims add the cell models: `$NANGATE45_VERILOG` or `$ASAP7_SC_VERILOG_DIR/*_RVT_TT_*.v`.

## 5. Observed issues on this host

- `genus -help` printed `Error : Unable to complete periodic license check succesfully` (2026-10-30): Cadence
  license reachability (`5280@cadence.webstore.illinois.edu`) may be flaky. Detect license errors in logs, retry
  with backoff, and never record a license failure as a design result.
- Every Cadence tool prints `WARNING This OS does not appear to be a Cadence supported Linux configuration.` (harmless).
- Genus writes `genus.log`, `genus.cmd` (and numbered variants) into CWD; a version check run from the repo
  root leaves these in the checkout. Always use a job dir as CWD.
- The Cadence module puts ~25 `tools.lnx86/lib` dirs first on `LD_LIBRARY_PATH` (old libstdc++/liblzma): never
  load it into the driver's environment.
- The module file references `WXE2303`, but the install dir is `VXE2303` (harmless dangling PATH entry).

## 6. Resource flags summary

| Tool | Threads | Memory | Licenses |
|---|---|---|---|
| genus | `set_db / .max_cpus_per_server N` (default 8) | none native → watchdog | `-lic_startup`, `-lic_stack` |
| lec | `set parallel option -threads N` / `compare -threads N` | none native → watchdog | each thread may take a license; `-nolicwait` |
| xrun | single-threaded sim by default | - | - |

## Sources

- `genus -help`, `genus -version`, `lec -help`, `lec -version`, `xrun -version`, `xrun -helpall` (run 2026-10-30).
- `/software/cadence-2024-08/GENUS211/doc/genus_comref/{synthesis,inout,constraints,sdc_commands}.html`;
  `/software/cadence-2024-08/GENUS211/doc/genus_attref/synthesis.html` (`max_cpus_per_server`).
- `/software/cadence-2024-08/CONFRML241/doc/conformal.hlp` (READ LIBRARY, READ DESIGN, ANALYZE DATAPATH, COMPARE,
  SET PARALLEL OPTION, EXIT incl. exit status codes, `analyze hier_compare` sample dofile).
- `/etc/modulefiles/software/cadence/2024-08.lua`.
