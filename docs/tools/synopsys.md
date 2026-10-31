# Synopsys tools (W-2024.09) for chiplet-compiler

Licensed host tools, launched only through `EdaRunner` / `eda-run` (see `eda_environment.md`). Versions were
checked on 2026-10-30: `dc_shell version - W-2024.09` (build Aug 27 2024), `Formality (R) Version W-2024.09`.
Install root: `/software/Synopsys-2024_x86_64/<tool>/W-2024.09/`. Man pages (cat format, strip overstrike with
`sed 's/.\x08//g'`) live under each tool's `doc/<tool>/man/cat{1,2,3}`; they are the primary source for every
option below.

## 1. What we use each tool for

| Tool | Binary | Our use |
|---|---|---|
| Library Compiler | `lc_shell` | `.lib` → `.db` for FreePDK45/ASAP7 (DC needs `.db`). See `pdks.md` §4. |
| Design Compiler | `dc_shell` | Synthesize XLS-generated Verilog FUs; area / timing / QoR reports; netlist + SVF for Formality. |
| Formality | `fm_shell` | Equivalence: XLS RTL vs DC netlist (and RTL vs RTL when merging). |
| PrimeTime | `pt_shell` | Optional sign-off STA / power on DC netlists. |
| VCS | `vcs` → `simv` | Simulate generated RTL against test vectors from Allo/XLS functional models. |
| SpyGlass | `sg_shell` | Lint of generated RTL (`lint/lint_rtl` goal). |
| VC Static | `vc_static_shell` | Alternative lint/formal; needs `VC_STATIC_HOME` (set by `setup_env.sh`). Not planned. |
| Fusion Compiler, ICC2 | `fc_shell`, `icc2_shell` | Out of scope (no P&R). |

## 2. Batch invocation

| Tool | Command (cwd = job dir) | Notes |
|---|---|---|
| lc_shell | `lc_shell -f make_db.tcl` | Script ends with `quit`. |
| dc_shell | `dc_shell -f syn.tcl -output_log_file dc.log` | `-f`: if the last statement is `quit`, no prompt is shown (`dc_shell(1)`). Add `-no_home_init` so `~/.synopsys_dc.setup` cannot change results. `-minimize_peak_mem` trades runtime for lower peak memory. `-x "cmd; cmd"` runs inline commands. |
| fm_shell | `fm_shell -work_path ./fm -f fm.tcl` | `-work_path` puts `formality.log`, `FM_WORK/`, `formality_svf/` in a chosen dir (`fm_shell(1)`). |
| pt_shell | `pt_shell -f sta.tcl -output_log_file pt.log` | (`pt_shell(1)`). |
| vcs | `vcs -full64 -sverilog -j4 -timescale=1ns/1ps -o simv -l comp.log tb.sv dut.v` then `./simv -l sim.log` | `-jN` parallel compile, no space; `-full64` 64-bit (`doc/help_vcs.txt`). simv warns about ASLR/`$save`; pass `-no_save` to simv to avoid re-exec (observed message). |
| sg_shell | `sg_shell < lint.tcl` | Form used by the shipped example `SPYGLASS_HOME/examples/sg_shell/interactive_waivers/RUNME`. |

Always give every script an explicit `quit`/`exit`; otherwise the tool waits on stdin (we pass `/dev/null`, so it
exits, but logs look like a crash).

## 3. Library Compiler

```tcl
read_lib /abs/path/stdcells.lib                       ;# compiles + screens the library
write_lib NangateOpenCellLibrary -format db -output NangateOpenCellLibrary.db
quit
```
`write_lib` takes the Liberty `library (NAME)`, not the file name, and silently overwrites (`write_lib(2)`).
LC has been a separate install from DC since K-2015.06 (`read_lib(2)`); here `SYNOPSYS_LC_ROOT` is set by the module.

## 4. Design Compiler script template

```tcl
set_host_options -max_cores 4                  ;# default 1 (no parallelism), max 16
set search_path   [list . /abs/build/libs]
set target_library [list NangateOpenCellLibrary.db]   ;# ASAP7: the 5 group .dbs of one VT/corner
set link_library   [concat * $target_library]         ;# "*" = designs in memory (link_library(3))
set_svf ./fu.svf                               ;# setup-verification guidance for Formality
analyze -format sverilog [list fu.sv]
elaborate fu
current_design fu
link
check_design > check_design.rpt
# Pipelined XLS output (has clk): period in the LIBRARY time unit (ns Nangate, ps ASAP7)
create_clock -name clk -period $PERIOD [get_ports clk]
set_input_delay  0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0 -clock clk [all_outputs]
# Combinational XLS output: use a virtual clock (create_clock -name vclk -period $P, no port)
# with the same I/O delays, or set_max_delay $P -from [all_inputs] -to [all_outputs].
compile_ultra -no_autoungroup
report_qor                      > qor.rpt
report_area -hierarchy -nosplit > area.rpt
report_timing -max_paths 10 -nosplit > timing.rpt
report_reference -nosplit       > ref.rpt
write_file -format verilog -hierarchy -output fu_netlist.v
write_sdc fu.sdc
quit
```

Notes from the man pages:
- `compile_ultra` auto-adds `dw_foundation.sldb` to `synthetic_library` for that command if the DesignWare license
  checks out, so arithmetic may map to DesignWare architectures (`compile_ultra(2)`). Record whether DW was used
  (log): it affects reproducibility across sites.
- By default `compile_ultra` ungroups small hierarchies and all DesignWare hierarchies; `-no_autoungroup`
  preserves user hierarchy (needed if we report per-sub-FU area) (`compile_ultra(2)`). Boundary optimization is
  on by default; `-no_boundary_optimization` keeps port functions intact.
- `-retime`, `-gate_clock`, `-scan` exist; we do not use them (XLS already pipelines; retiming would move
  registers XLS placed).
- `report_area -hierarchy` gives per-instance area; units are the library's (µm² for both kits).
- `set_host_options -max_cores`: tool also reduces cores on a loaded host (UIO-231 warning) unless
  `disable_multicore_resource_checks` is true (`set_host_options(2)`). Keep the default checks on this shared machine.

Parsing: prefer `report_qor` and `report_area` totals ("Total cell area"); grep `Error:` lines in the log; treat
`Warning: Design 'X' has N unresolved references` (from `link`/`check_design`) as failure (exact text (unverified)).

## 5. Formality script template

```tcl
set_host_options -max_cores 4 -max_memory 32   ;# cores max 8; >4 cores consumes a 2nd license; GB limit
set_svf ./fu.svf                                ;# from DC
read_db -technology_library /abs/build/libs/NangateOpenCellLibrary.db   ;# or: read_lib $env(NANGATE45_LIB)
read_sverilog -r fu.sv
set_top r:/WORK/fu
read_verilog -i fu_netlist.v
set_top i:/WORK/fu
match
report_unmatched_points > unmatched.rpt
verify
report_failing_points   > failing.rpt
report_status           > status.rpt
exit
```
- `-r` / `-i` read into the reference / implementation containers (`read_verilog(2)`, `read_db(2)`).
- `-max_memory`: Formality periodically checks its RSS-sum against the limit and stops the command when reached;
  default (negative) disables checks (`set_host_options(2)`, fm). Use it; it is the only native memory cap here.
- Result strings in the docs: `Verification SUCCEEDED` / `Verification FAILED`; also handle inconclusive/aborted
  points (`verify(2)`: verification fails if any compare point fails or aborts). Parse the log; exit code
  semantics are (unverified).
- Formality `read_lib` reads `.lib` directly, so `.db` is optional for Formality (`read_lib(2)`, fm).

## 6. PrimeTime (optional)

```tcl
set_host_options -max_cores 4
set link_path [list * /abs/build/libs/NangateOpenCellLibrary.db]
read_verilog fu_netlist.v
link_design fu
read_sdc fu.sdc
update_timing
report_timing -max_paths 10 > pt_timing.rpt
# power needs activity: read_saif from a VCS run, then report_power
quit
```
Commands confirmed present in PT man pages (`read_verilog`, `link_design`, `read_sdc`, `update_timing`,
`report_timing`, `read_saif`, `report_power`, `set_host_options -max_cores`). `link_path` usage is standard PT
practice ((unverified) against the local man page).

## 7. SpyGlass lint

```tcl
new_project lint_fu -force
read_file -type hdl fu.sv
current_goal lint/lint_rtl -top fu
run_goal
write_report moresimple > lint_moresimple.rpt
exit
```
Pattern taken from `SPYGLASS_HOME/examples/sg_shell/interactive_waivers/test.tcl`; the goal file exists at
`SPYGLASS_HOME/GuideWare/latest/block/rtl_handoff/lint/lint_rtl.spq`. `-top` and `exit` syntax (unverified).
Generated XLS RTL will trigger style warnings (e.g. unused bits); keep a reviewed waiver list rather than editing RTL.

## 8. Resource flags summary

| Tool | Threads | Memory | Other |
|---|---|---|---|
| dc_shell | `set_host_options -max_cores N` (≤16) | `-minimize_peak_mem` | `-no_home_init` |
| fm_shell | `set_host_options -max_cores N` (≤8; >4 = 2 licenses) | `-max_memory GB` | `-work_path` |
| pt_shell | `set_host_options -max_cores N` | - | |
| vcs | `-jN` (compile) | - | `simv -no_save` |
| lc_shell / sg_shell | - | - | |

Plus the external RSS watchdog and wall-clock timeout from `EdaRunner` for every job.

## 9. Pitfalls

- `dc_shell` fails to start without `~/pdks/extra_libs/usr/lib64` on `LD_LIBRARY_PATH` (`libpulse-mainloop-glib.so.0`
  missing) (observed); `eda-run` adds it.
- `vcs` needs `VCS_HOME` (module sets it); otherwise `Cannot find 'vcsMsgReport' script` (observed).
- The Synopsys module does not set `SYNOPSYS`; the tool wrappers locate their own root (DC/FM ran fine).
- Units: DC/PT clock periods are in the target library's `time_unit` (ns for Nangate45, ps for ASAP7).
- Tools write logs and work dirs into CWD; always run in `build/.<pass>/…/<job>/`.
- `lc_shell` W-2024.09 crashes on this host without the `~/pdks/extra_bin/lc_shell` wrapper (put on `PATH` by
  `setup_env.sh`; `eda_environment.md` §6). `vcs` needs `-full64`. `dc_shell -version` exits 1 after printing the version.

## Sources

- Man pages under `/software/Synopsys-2024_x86_64/syn/W-2024.09/doc/syn/man/`: `cat1/dc_shell.1`, `cat1/lc_shell.1`,
  `cat2/{set_host_options,compile_ultra,read_lib,write_lib,report_area,write_file,set_svf,report_reference}.2`,
  `cat3/{link_library,target_library,search_path,synthetic_library}.3`.
- Formality: `/software/Synopsys-2024_x86_64/fm/W-2024.09/doc/fm/man/cat1/fm_shell.1`,
  `cat2/{set_host_options,read_verilog,read_sverilog,read_db,read_lib,set_top,match,verify,report_failing_points}.2`.
- PrimeTime: `/software/Synopsys-2024_x86_64/prime/W-2024.09/doc/pt/man/cat1/pt_shell.1`, `cat2/{set_host_options,read_lib,link_design,read_sdc,read_saif}.2`.
- VCS: `/software/Synopsys-2024_x86_64/vcs/W-2024.09/doc/help_vcs.txt`.
- SpyGlass: `/software/Synopsys-2024_x86_64/spyglass/W-2024.09/SPYGLASS_HOME/{SpyGlass_Version,examples/sg_shell/interactive_waivers/,GuideWare/latest/block/rtl_handoff/lint/}`.
- Module file `/etc/modulefiles/software/Synopsys_x86-64/2024.lua`; version probes run 2026-10-30 from a scratch dir.
