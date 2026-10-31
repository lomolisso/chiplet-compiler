# PDKs on this host: FreePDK45 / Nangate45 and ASAP7

What the synthesis passes need from the two process kits: which Liberty/LEF/Verilog files, which
corners, units, and how to get Synopsys `.db` files. Paths and numbers were read from the files on
2026-10-30.

## 1. Environment variables (from `~/pdks/setup_env.sh`, part 1: always exported when sourced)

| Variable | Value |
|---|---|
| `EDA_PDK_ROOT` | `$HOME/pdks` |
| `FREEPDK45_ROOT` | `$EDA_PDK_ROOT/FreePDK45` |
| `NANGATE45_DIR` | `$FREEPDK45_ROOT/Nangate45_OpenCellLib` |
| `NANGATE45_LIB` / `_LEF` / `_TECH_LEF` / `_GDS` / `_VERILOG` | `$NANGATE45_DIR/{stdcells.lib, stdcells.lef, rtk-tech.lef, stdcells.gds, stdcells.v}` |
| `ASAP7_ROOT` | `$EDA_PDK_ROOT/asap7` (OpenROAD `asap7` repo with submodules) |
| `ASAP7_PDK_ROOT` | `$ASAP7_ROOT/asap7_pdk_r1p7` (PDK r1.7) |
| `ASAP7_SPICE_MODELS`, `ASAP7_CDSLIB`, `ASAP7_CALIBRE_DIR` | `$ASAP7_PDK_ROOT/{models/hspice, cdslib, calibre}` |
| `ASAP7_SC_DIR` | `$ASAP7_ROOT/asap7sc7p5t_28` (7.5-track, v28, README-recommended) |
| `ASAP7_TECH_LEF` / `ASAP7_TECH_LEF_4X` | `$ASAP7_SC_DIR/techlef_misc/asap7_tech_{1x,4x}_201209.lef` |
| `ASAP7_SC_LEF_DIR`, `ASAP7_SC_NLDM_DIR`, `ASAP7_SC_CCS_DIR`, `ASAP7_SC_GDS_DIR`, `ASAP7_SC_VERILOG_DIR` | `$ASAP7_SC_DIR/{LEF, LIB/NLDM, LIB/CCS, GDS, Verilog}` |
| `ASAP7_SC7P5T_27_DIR`, `ASAP7_SC6T_26_DIR`, `ASAP7_SRAM_DIR` | older 7.5T v27, 6-track v26, `asap7_sram_0p0` |

Because these are exported outside the interactive guard, Tcl scripts can use `$env(NANGATE45_LIB)` etc. in
any job launched through `eda-run` (see `eda_environment.md`). Pass resolved absolute paths into our generated
scripts anyway, and record them (plus file hashes) in the pass's `verify.json`.

## 2. FreePDK45 + Nangate45 Open Cell Library

- FreePDK45 v1.4 (2011-04-07), NCSU; Apache-2.0 except SVRF (`$FREEPDK45_ROOT/README.txt`).
- `stdcells.lib`: `library (NangateOpenCellLibrary)`, characterized with NGLibraryCharacterizer v2011.01,
  **"Characterization Corner : typical"**: the only corner present. `nom_voltage 1.10`, `nom_temperature 25`,
  `nom_process 1.00`, `default_operating_conditions : typical`, `delay_model : table_lookup` (NLDM).
- Units in the file: `time_unit "1ns"`, `capacitive_load_unit (1,ff)`, `leakage_power_unit "1nW"`,
  `voltage_unit "1V"`, `current_unit "1mA"`, `pulling_resistance_unit "1kohm"`. Area values are µm²
  (e.g. `INV_X1` has `area : 0.532`; Liberty has no area-unit attribute, so µm² is the kit's convention).
- 135 cells, including `FA_X1`, `HA_X1`, `DFF_X1/X2`, `DFFR_*`, `DFFS_*`, `DFFRS_*`, scan `SDFF*`, `CLKGATE_X*`,
  `CLKGATETST_X*`.
- Other files: `stdcells.lef`, `rtk-tech.lef` (tech LEF), `stdcells.gds`, `stdcells.v` (functional models for
  gate-level simulation), `klayout.lyp`. No `.db`.

Use for synthesis: `$NANGATE45_LIB` as the single target library. Report timing in ns.

## 3. ASAP7 (asap7sc7p5t_28)

- Predictive 7 nm FinFET PDK (ASU/ARM); BSD-3-Clause; cite Clark et al., Microelectronics J. 53 (2016) and
  Vashishtha et al., ICCAD 2017 (`$ASAP7_ROOT/README.md`).
- Liberty is split by **cell group × VT × corner**, each its own Liberty `library`:
  - groups: `AO` (42 cells), `OA` (34), `INVBUF` (37), `SEQ` (33), `SIMPLE` (56) (counts for RVT TT);
  - VT flavors: `LVT`, `RVT`, `SLVT`, `SRAM` (cell suffix `_ASAP7_75t_{L,R,SL,SRAM}`);
  - corners: `TT` (`PVT_0P7V_25C`), `FF` (`PVT_0P77V_0C`), `SS` (`PVT_0P63V_100C`);
  - file names carry different date stamps: `AO/OA/SIMPLE …_nldm_211120.lib`, `INVBUF …_nldm_220122.lib`,
    `SEQ …_nldm_220123.lib`. Glob them; do not hard-code one stamp.
- `LIB/NLDM/*.lib` are extracted (the `.lib.7z` originals sit next to them; ~602 MB total). `LIB/CCS/` holds only
  `.7z` archives (not extracted).
- Units in the file: `time_unit "1ps"`, `capacitive_load_unit (1,ff)`, `leakage_power_unit "1pW"`,
  `voltage_unit "1V"`, `current_unit "1mA"`. **Timing is in ps** (Nangate is ns): normalise before comparing
  PDKs, and give `clock_period_ps` to XLS and `create_clock -period` in the library's time unit.
- Area in Liberty is at **1x** scale in µm²: `INVx1_ASAP7_75t_R` has `area : 0.04374` = 0.162 × 0.27, the
  `SIZE` in the 1x LEF.
- LEFs: `LEF/asap7sc7p5t_28_{L,R,SL,SRAM}_1x_220121a.lef` (1x) and `LEF/scaled/…_4x_220121a.lef`
  (4x: INVx1 is `SIZE 0.648 BY 1.08`), with matching `asap7_tech_{1x,4x}_201209.lef`. The 4x set exists for
  P&R tools that cannot handle sub-20 nm geometry (commonly stated rationale, (unverified); no statement in the
  local README). We do not do P&R; if a placed-area figure is ever taken from a 4x flow, divide by 16.
- Verilog models: `Verilog/asap7sc7p5t_{AO,INVBUF,OA,SEQ,SIMPLE}_{LVT,RVT,SLVT,SRAM}_TT_*.v` (TT only).
- Sequential cells (RVT): `DFFHQNx1/2/3` (QN output only), `DFFHQx4`, `DFFLQN*`, `DFFASRHQNx1` (the only
  async set/reset flop), latches `DHL*`/`DLL*`, clock gates `ICG*`, scan `SDFH*`/`SDFL*`. There is **no plain
  async-reset DFF**, so generate RTL with synchronous reset (or none) for ASAP7 (see `xls.md`, codegen `--reset`).
- Arithmetic cells: `FAx1`, `HAxp5`, `MAJIxp5`, `MAJx2`, `MAJx3` (SIMPLE group).

Default selection for chiplet-compiler synthesis: **RVT, TT**, all five groups:

```
$ASAP7_SC_NLDM_DIR/asap7sc7p5t_{AO,OA,INVBUF,SEQ,SIMPLE}_RVT_TT_nldm_*.lib
```

A Synopsys user enablement for ASAP7 exists at https://github.com/snishizawa/asap7_snps (linked from the ASAP7
README); not installed here.

## 4. Making `.db` files (Library Compiler)

Neither kit ships `.db`. Design Compiler's `target_library`/`link_library` need `.db`. Formality (`read_lib`)
and Genus/Conformal read `.lib` directly; PrimeTime's `read_lib` compiles `.lib` through Library Compiler.

`lc_shell` script (one `write_lib` per Liberty `library`; the argument is the **library name inside the
file**, not the file name, `write_lib(2)`):

```tcl
# make_db.tcl  (run: eda-run lc_shell -f make_db.tcl, cwd = build/.synth/libs/)
read_lib /home/rid2/pdks/FreePDK45/Nangate45_OpenCellLib/stdcells.lib
write_lib NangateOpenCellLibrary -format db -output NangateOpenCellLibrary.db
# ASAP7: repeat per group file, e.g.
read_lib /home/rid2/pdks/asap7/asap7sc7p5t_28/LIB/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
write_lib asap7sc7p5t_SEQ_RVT_TT_nldm_220123 -format db -output asap7sc7p5t_SEQ_RVT_TT_nldm_220123.db
quit
```

Practical rule: parse `library (NAME)` from each `.lib` in Rust, emit one `read_lib <path>` +
`write_lib <NAME> -format db -output <NAME>.db` per file, cache the result under `build/` keyed by the `.lib`
hash. `write_lib` overwrites silently. Library Compiler's `read_lib` runs the library screener; keep its warnings
in the log for review. The Library Compiler install is separate from DC since K-2015.06 (`read_lib(2)` note).

## 5. Pitfalls

| Pitfall | Mitigation |
|---|---|
| ns (Nangate) vs ps (ASAP7) | Store periods in ps internally; convert when writing SDC per library `time_unit`. |
| Leakage nW vs pW | Normalise in report parsers. |
| One ASAP7 corner = 5 libraries | Always read all 5 groups of one VT/corner; missing `SEQ` means no flops. |
| Mixing VT flavors | Don't, unless the experiment asks for multi-VT. |
| ASAP7 `SRAM` VT flavor | Cell flavor for SRAM-periphery VT, not memory macros (macros are in `asap7_sram_0p0`). |
| Liberty area units | No unit attribute; both kits use µm² at 1x. |
| Large libs | ASAP7 RVT TT libs are 0.3–22 MB each (AO/OA ~20 MB); cache `.db`s rather than rebuilding per job. |

## Sources

- `~/pdks/setup_env.sh`, `~/pdks/README.md`, `$FREEPDK45_ROOT/README.txt`, `$ASAP7_ROOT/README.md`,
  `$ASAP7_SC_DIR/techlef_misc/README.txt` (read 2026-10-30).
- Liberty headers: `$NANGATE45_LIB` lines 37–74; `$ASAP7_SC_NLDM_DIR/asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib`
  lines 34–71; SEQ FF/SS headers; LEF `SIZE` of `INVx1_ASAP7_75t_R` in 1x and 4x LEFs.
- Synopsys man pages: `/software/Synopsys-2024_x86_64/syn/W-2024.09/doc/syn/man/cat2/{read_lib,write_lib}.2`;
  Formality `/software/Synopsys-2024_x86_64/fm/W-2024.09/doc/fm/man/cat2/{read_lib,read_db}.2`;
  PrimeTime `/software/Synopsys-2024_x86_64/prime/W-2024.09/doc/pt/man/cat2/read_lib.2`.
- ASAP7 repo: https://github.com/The-OpenROAD-Project/asap7 ; L. T. Clark et al., "ASAP7: A 7-nm finFET
  predictive process design kit," Microelectronics J. 53:105–115, 2016, https://doi.org/10.1016/j.mejo.2016.04.006
