# `characterize`

1. **Purpose.** The primitive library A: area, delay and leakage of every hardware primitive the
   functional units are built from, per cell library (the kits of `pdks/`: Nangate45, ASAP7) and timing
   point, through the configured synthesis flow; every netlist passes cheap structural checks, and,
   when `lec` is on, an equivalence check against its RTL. `merge` weighs its clique by the area a shared primitive saves, bounds the
   logic depth of a merged unit by the primitives' delays, and prices precision decomposition with the
   split multipliers. Position: a parallel branch that depends only on the environment (`chipc doctor`,
   docs/framework/environment.md); no model is read, so `--models` never reruns it.

2. **Background.**
   - Design §4.9, §6.2, §6.4; the op table and the scalar op semantics (docs/framework/ir.md,
     docs/framework/smt.md).
   - The two synthesis flows (design §3.7, `SynthFlow`, docs/framework/adapters.md): the industry flow's
     Design Compiler with Formality (docs/tools/synopsys.md) and the open flow's Yosys with ABC (Wolf,
     Yosys Open SYnthesis Suite; Brayton and Mishchenko, "ABC: an academic industrial-strength
     verification tool", CAV 2010); the kits and their manifests (docs/tools/pdks.md); XLS
     (docs/tools/xls.md).
   - Equivalence of a gate netlist and its RTL, when `lec` is on: combinational equivalence checking with
     the synthesis tool's guidance file (SVF) for Formality; Yosys' `equiv_make` / `equiv_simple` /
     `equiv_induct` (signal pairs by name, SAT per pair, induction over registers) for the Yosys flow. The
     synthesis tools are trusted (design §5), so the check is off by default.
   - Cost growth: an `n x n` array or tree multiplier has Θ(n²) cells (Wallace, IEEE TEC 13(1), 1964),
     a ripple or prefix adder Θ(n) to Θ(n log n); a w-bit multiply composes from ⌈w/n⌉² n-bit products
     with shifts and adds (schoolbook; Jouppi et al., ISCA 2017 for the temporal form, Sharma et al.,
     ISCA 2018 for the spatial one).

3. **Inputs and outputs.**
   - Inputs (by path, not artifacts): each kit's PDK manifest (`pdks/<kit>.toml`), its Liberty file (the
     one the manifest names, or for a kit of cell groups the merged library
     `build/.env/out/pdk_lib/<merged>.lib`, e.g. `asap7sc7p5t_RVT_TT`) and its `.db`
     `build/.env/out/pdk_db/<library>.db`, as `chipc doctor` writes them. Their hashes and doctor's tool
     versions enter the cache key (`fingerprints`).
   - `out/library/catalog.json` (schema `characterize_catalog`): every primitive with its key
     (`<op>[.<variant>]@<type>[><type>]`: `add@i32`, `compare.LT@ui8`, `convert@f32>bf16`, `mux4@w8`,
     `register@w8`, `accumulator.add@i32`, `split@w32n8`), kind, op, family, types, width, the
     `provisional` and `pending` marks, dependencies and its RTL source (DSLX or Verilog).
   - `out/library/library.json` (`Library`, schema `characterize_library`, `cc.characterize.library/2`):
     the flow (`tool`), the equivalence checker (`lec_tool`, absent when the check is off), the tool
     versions, the kits (library name, corner, time unit, driving cell, load, clock, the `.lib` and `.db`
     hashes), the targets (kit, target, period), and one row per (primitive, kit, target): `measured`
     (`area_um2`, `seq_area_um2`, `delay_ps`, `slack_ps`, `leakage_nw`, `children`: the area of each direct
     child instance: split pieces, DesignWare blocks), `estimator` (the flow), `lec_tool` and `lec`
     (verdict and compare point counts; both absent when the check is off), `inconclusive`, `designware`,
     `provisional`, the netlist path and hash, the report paths.
   - `out/library/coverage.json` (`Coverage`): each primitive `measured`, `provisional` or `pending`
     (with its reason) and its row count.
   - `out/library/rtl/<id>.{x,v}`, `netlist/<kit>/<target>/<id>.v`,
     `reports/<kit>/<target>/<id>.{area,qor,timing,power,check,lec}.rpt` and `<id>.log` (the design's
     part of the session log, then `@@lec` and its part of the equivalence log).
   - `out/library/summary.json` (`Summary`, `cc.characterize.summary/2`): the flow, whether the
     equivalence check ran (`lec`), primitives by kind, measured / provisional / pending, rows, rows
     proven equivalent, DesignWare rows, tool sessions run and reused, tool time.
   - `views/library.md`: area and delay per op and type, area against width with the fitted exponents
     and the flags, the pending primitives. Outside `out/`: `eda/` (the job directories) and `alib/`
     (Design Compiler's library analysis).

4. **Algorithm.**
   1. *Catalog* (`catalog.rs`). The scalar types: bool, signed and unsigned integers of `int_widths`, the
      `float_formats`. For every element-wise op of the op table and every type (every ordered pair for
      `convert`, every direction EQ, NE, LT, LE for `compare`; GT/GE are LT/LE with exchanged operands),
      the primitive exists iff `cc_smt::ops::apply` defines the op on those types: the catalog follows the
      semantics, not a list. `complex`, `real` and `imag` are wiring (a complex op is a pair of float
      rows). Structural primitives: an N-input multiplexer per `mux_inputs` and `mux_widths` (binary
      select), a register per `register_widths` (D flip-flops, no reset: ASAP7 has no plain async-reset
      flop), an accumulator per commutative op of the table at every type it has RTL for (the combiner
      with its result register in the loop, and a `start` input that loads the first element: without a
      defined start value DC assumed one and removed the register of every multiply accumulator, which
      the flip-flop check caught), a split multiplier per `split_multipliers` (w, n), and last, so that
      adding them keeps the earlier job batches, a `rescale@i<n>` per signed width of `int_widths` when
      `convert@i32>f32` has RTL (`merge`'s fixed requantization, `cc_smt::ops::rescale`).
   2. *RTL.* Integer and bool ops are DSLX mirroring `cc_smt::ops`: wrapping arithmetic, XLA's results for
      `x / 0 = -1`, `INT_MIN / -1 = INT_MIN`, `x % 0 = x`, `INT_MIN % -1 = 0`, a remainder with the sign of
      the dividend. Float ops call XLS's `apfloat` library (`add`, `sub`, `mul`, `eq_2`, `lt_2`, `lte_2`,
      `negate`, `abs`, `floor_with_denorms`, `ceil_with_denorms`, `round`, `is_nan`/`is_inf`,
      `upcast_with_denorms`, `downcast_rne`, `cast_from_fixed_using_rne`, `to_int`/`to_uint`); maximum
      and minimum are written per the spec (NaN propagates, -0 < +0). A conversion between formats where
      neither contains the other widens exactly to Float32 and rounds once. These rows are
      **provisional**: `apfloat` flushes subnormals and models IEEE special values for every format (E4M3FN
      has no infinities); third-party float IP cores (design §4.12) replace them. Float ops without an `apfloat` function
      (divide, remainder, sqrt, rsqrt, exponential, log, log_plus_one, sine, cosine, tanh, power,
      reduce_precision) are **pending**: no RTL, no rows. One container converts every DSLX file
      (`ir_converter_main`, `opt_main`, `codegen_main --generator=combinational`); registers,
      accumulators and split multipliers are Verilog templates that instantiate the generated modules.
      The split multiplier `split@w<w>n<n>` keeps every partial product `a_i * b_j` with `(i + j) n < w`
      on its own `n x n` unsigned instance and sums them shifted: the low w bits of a product are the same
      for signed and unsigned operands, so one split serves both families. `rescale@i<n>` is a template
      too: `rescale_core_i<n>` decodes the f32 scale, multiplies `n x 24` bits exactly, shifts with
      rounding half up (or left by at most 7 places), saturates to i32, and `rescale_i<n>` converts the
      result with the characterized `convert_i32_f32`. Its rows are provisional with that conversion; the
      deep test `rescale_rtl_matches_its_reference` simulates the core (iverilog, in the image) against
      `cc_smt::ops::rescale_reference` on edge cases and 3000 operands per width.
   3. *Jobs.* The designs go in batches of `batch`, one job per (kit, target, batch): a directory
      `eda/<kit>-<target>-<hash>/` named by the hash of the flow's name, the `.lib` and `.db` hashes, the
      image and the full text of every synthesis script and RTL file, with the RTL copied in. The Design
      Compiler scripts contain absolute paths (the `.db` directory, `alib/`), so a copy of a build
      directory elsewhere reuses no job. `synth.done` marks a finished synthesis and `lec.done` a finished
      equivalence run with the hash of its scripts: a later run reuses both, a changed equivalence
      script (or turning `lec` on) reruns only the check, so an interrupted run resumes and a config
      change reruns only what it touches. The deep test `characterize_job_keys_match_the_standing_library`
      checks that the job directories this code computes are those of the library in the build area's
      `build/.characterize/`. Jobs run `threads` at a time; each EDA call also holds one of `tools.eda.max_licenses`.
      Each session gets `memory_gb / threads` and `cores`.
   4. *Synthesis* through the configured `SynthFlow` (`eda/flows.rs`, scripts in `eda/scripts.rs`), every
      design: a virtual clock (the `clk` port for registers and accumulators) at the target's period in
      the library's time unit (ns for Nangate45, ps for ASAP7), input and output delays 0, every input
      driven by the kit's driving cell (its manifest's, or `driving_cell`), every output loaded with
      `load_fanout` times that cell's input capacitance (read from the Liberty file).
      - Design Compiler, one session per design (a session that synthesized one design gives the next
        a different netlist: a 32-bit divider after a 16-bit one came out at 15 614 instead of 9 340 µm²,
        and only the second one was provable): `compile_ultra -no_autoungroup` (hierarchy kept, so a split multiplier reports
        its pieces) with DesignWare parts kept as blocks (`compile_ultra_ungroup_dw false`), and
        `alib_library_analysis_path` on `alib/`. The ASAP7 group libraries are merged
        into one by doctor because DC's library analysis needs one library with inverters and flops: on
        the five group `.db` files it leaves placeholders and repeats the analysis (about 100 s) on every
        compile. With `lec`, Formality then checks RTL against netlist with the SVF, one design after the
        other in one session, each verification stopped after `lec_timeout_min`; a design not proven is verified
        again with each of `lec_strategies` (Formality's alternate strategies,
        `verification_alternate_strategy`), and fails if none proves it. The default strategy alone
        left a flattened 32-bit integer divider unverified after an hour (Formality's own limit is 36
        hours), and strategies s2 and s3 after 5 minutes each; with the DesignWare divider kept as a
        block Formality proves it in 8 s. Keeping the blocks costs DC the optimization across their
        boundaries.
      - Yosys: `synth -flatten`, `dfflibmap`, ABC mapping at the period (`map -D`, then `stime -p`),
        `stat -liberty`; with `lec`, Yosys' equivalence flow between the RTL and the netlist in the same
        script (`equiv_make` pairs
        signals by name, registers included; `equiv_simple -seq 2`, `equiv_induct -seq 2`,
        `equiv_status -assert`; 600 s at most). ABC's `cec` was tried first: it matches AIGER ports by
        position (the RTL's and the netlist's orders differ) and leaves an f32 multiplier undecided after
        15 minutes. The mapping is plain `map` without choices (`dch`), which keeps the netlist close to
        the RTL's structure for the SAT-based proofs. ABC reports no path for a design that is only
        flip-flops, so Yosys rows of registers may have no delay.
        Rows record `estimator = yosys`; this flow needs no license and is chosen only by `tool`, never as
        a fallback after a license failure (which fails the pass after `EdaRunner`'s retries).
      - The `clock` target is the kit's clock (its manifest's `clock_ps`, or the section's); `fast` is `fast_fraction` of it, tight
        enough that synthesis builds its fastest structure: its delay is the primitive's minimum and its
        area what that costs. At the clock target a tool spends slack on area, so its delay is close to
        the period for anything that fits; the minimum delay is what a logic-depth bound needs.
   5. *Rows.* Each session log is split per design at its markers; the flow parses the reports: area
      (`Total cell area`), the longest path (the critical path length of every path group) and its slack,
      and the cell leakage (normalised to nW). A design fails on a tool error line, an unresolved
      reference, a cheap check (below: ports, outputs driven, flip-flops, latches, foreign cells, area), a
      missing value, or, with `lec`, an equivalence not proven. A series of one op whose area at a wider
      width is below half its area at a narrower one fails too (`sanity::violations`). The pass fails after
      writing everything. A conversion that is only wires (sign extension, truncation) has area 0 and no
      timing path.

   Cost: one synthesis per (primitive, kit, target), plus one equivalence check with `lec`; the session
   time is dominated by the float primitives (tens of seconds each), integer primitives take seconds.

5. **Config** (`[passes.characterize]`): `tool = "dc"` (the synthesis flow: `"dc"` or `"yosys"`; another
   name is an error that lists them), `lec = false` (the equivalence check), `pdks = ["nangate45",
   "asap7"]` (kits by manifest; `[]` is every manifest), `clock_ps = {}` and `driving_cell = {}` (per kit,
   overriding the manifest's `clock_ps`, 2000 and 1000, and `driving_cell`, `BUF_X1` and
   `BUFx2_ASAP7_75t_R`), `targets = ["clock", "fast"]`, `fast_fraction = 0.05`, `ops = []`, `int_widths =
   [2, 4, 8, 16, 32]`, `float_formats = ["f32", "f16", "bf16", "f8e4m3fn"]`, `mux_inputs = [2 … 8]`,
   `mux_widths = [1, 2, 4, 8, 16, 32]`, `register_widths = [1, 2, 4, 8, 16, 32]`, `split_multipliers =
   [[8, 4], [16, 8], [32, 8], [32, 16]]`, `load_fanout = 4`, `lec_timeout_min = 2`, `lec_strategies = []`
   (both with `lec`), `batch = 24`, `cores = 2` (per Design Compiler session), `views = true`. The corner
   is the manifest's: Nangate45's only corner (typical) and ASAP7's `vt`/`corner` (RVT, TT). Resources:
   `threads = 2`, `memory_gb = 8`, `timeout_min = 600`.

6. **Verification** (`verify.rs`; it reads the copies in `out/library/`, the Liberty files and the
   config, never the run's results):
   - `catalog`: the catalog regenerated from the config equals `catalog.json`.
   - `rtl`: every source in `rtl/` equals the catalog's, and every primitive with RTL has its Verilog.
   - `complete`: exactly one row per primitive with RTL, kit and target, none for anything else; the
     coverage states each primitive as the catalog does, every pending one with its reason.
   - `<kit>:libraries`: the `.lib` and `.db` the rows were measured on are the ones there now, with the
     same driving cell, load and clock.
   - `<kit>:<target>:lec` (skipped when `lec` is off; then no row may carry an equivalence result): every
     equivalence report, re-read, proves the netlist equal to its RTL. A
     failing point always fails. A check that ran out of time (`lec_timeout_min`) with no failing point
     and some unverified ones passes only when every sanity check of that design passes; the row is
     marked `inconclusive`, counted and listed (in the view too).
   - `<kit>:<target>:sanity` (the cheap checks, always on): per design, from its log, reports, netlist and
     RTL: the netlist's top module has exactly the RTL's ports (names, directions, widths); every output
     bit is driven (the output, the bit or a range holding it appears in an `assign` or on an instance's
     port); it has as many flip-flops (cells with
     an `ff` group in the Liberty file) as the RTL describes (a register's or accumulator's width,
     otherwise none); no latch cell; no cell outside the library (no generic cells left); cells unless
     the RTL is only wiring; a positive area and (except a register in a flow without flop-only paths:
     Yosys) a delay in the reports; no tool error line and no unresolved reference. These checks catch a netlist that is not the design that was
     meant, which the equivalence check alone does not: the first full run found every multiply
     accumulator with no flip-flops (no start value, so DC assumed one and removed the register) while
     Formality called them equivalent.
   - `<kit>:<target>:rows`: every value of `library.json` equals the one re-read from its report copies
     through the flow's parsers; the estimator and `lec_tool` are the config's; every netlist hash
     matches.
   - `trends`: area against width per series (integer and bit-vector primitives): a wider width with less
     than half the area of a narrower one fails (a clear violation: logic was lost); a smaller decrease,
     or a growth exponent off the expected one (multiply about 2, add and subtract about 1), is a note.
   - `coverage.datapaths`: when `build/.datapaths/out` exists, every vertex of every datapath (template
     stages included) at every width tuple, and every single-op accumulator, maps to a catalog key
     (measured, provisional or pending; complex ops to their float rows, `const` needs none). A gap fails
     the check; without the datapaths it is skipped. The datapaths manifest's hash is recorded.

   Negative tests (`tests/smoke/characterize.rs`, on stand-in tools and the test kit `mini`): a failed
   equivalence report, a latch, a cell outside the library, an unresolved reference, a tool error, a port
   of another width, an output not driven, a missing or extra flip-flop, an edited value, a missing row,
   an edited catalog or RTL file, a datapath type the catalog lacks, an area below half of a narrower
   width's; an inconclusive check passes with clean sanity checks and fails with a wrong port; a mild
   off-trend point stays a note; with `lec` off no equivalence check runs, and turning it on runs only
   the check; a flow no registry has is a configuration error.

7. **Resources.** Full catalog at the defaults (2026-11-02, iacoma227, DC + Formality W-2024.09, 2
   parallel sessions): 681 primitives, 635 with RTL, 2 540 rows. Synthesis about 5.3 h of wall time (one
   DC session per design: median 7.7 s, a 32-bit divider 7 to 20 minutes, the fast target slower than the
   clock target); the equivalence checks of every row about 39 minutes (median 1 s per design, the slowest
   proven 47 s, 7 rows stopped at 2 minutes). chipc's peak RSS 0.87 GB, one DC or Formality session
   about 0.5 to 0.8 GB; `out/library/` 226 MB. Results: 2 533 rows proven equivalent, 7 inconclusive
   (divide@ui32 on both kits and targets, remainder@i32 on Nangate45, remainder@ui32 on Nangate45 fast,
   each with 0 failing points), 0 sanity problems, 256 area-against-width series without a flag; the
   datapaths of 830 classes use 111 library keys (59 provisional, 20 pending), no gap. Examples (area
   µm² / delay ps):

   | primitive | Nangate45 clock | Nangate45 fast | ASAP7 clock | ASAP7 fast |
   |---|---|---|---|---|
   | `add@i32` | 261.5 / 1760 | 297.4 / 420 | 10.4 / 994 | 25.8 / 161 |
   | `multiply@i32` | 2183.6 / 2000 | 2245.3 / 1200 | 136.8 / 999 | 214.8 / 461 |
   | `split@w32n8` | 2802.6 / 2000 | 3349.7 / 1440 | 166.9 / 996 | 308.2 / 535 |
   | `divide@i32` | 9339.5 / 8880 | 9111.6 / 9150 | 752.5 / 3443 | 772.0 / 3402 |
   | `add@f32` | 1888.9 / 2660 | 1996.9 / 2500 | 144.4 / 1000 | 151.4 / 966 |
   | `multiply@f32` | 3206.4 / 2270 | 3186.1 / 2140 | 248.1 / 1000 | 296.0 / 812 |
   | `multiply@bf16` | 550.6 / 1990 | 713.9 / 1390 | 42.0 / 983 | 58.5 / 540 |
   | `mux4@w32` | 117.0 / 500 | 196.3 / 290 | 7.8 / 247 | 8.8 / 98 |
   | `register@w32` | 144.7 / 120 | 195.8 / 100 | 10.7 / 54 | 14.6 / 44 |
   | `accumulator.add@i32` | 380.6 / 1910 | 531.2 / 520 | 24.8 / 981 | 39.9 / 219 |

   With `lec = false` and the job directories of an earlier run in place (2026-11-05, iacoma227), the
   whole library is rebuilt from them in 8 s: 108 jobs reused, 0 run, every check green on the 2 560
   rows (no undriven output, 260 series without a violation).

   Deep test (`tests/deep/characterize.rs`: doctor, then `add` and `multiply` at 8 bits and f32, their
   accumulators, a 2-input mux, a register and a 16-from-8 split multiplier on both kits and targets,
   through Design Compiler without and then with Formality, and Yosys): DEEPTIME.

8. **Limitations.**
   - Float rows are provisional (XLS `apfloat`: subnormals flushed, IEEE special values for E4M3FN) and
     12 float ops are pending until the third-party float IP cores are integrated; `merge` must treat pending
     primitives as unpriced.
   - Planned changes (design §4.9, §4.12): float primitives as wrappers around the flow's IP cores, float
     `remainder` dropped (its only use is constant computation).
   - The cheap checks find a netlist that is not the design meant (missing ports, floating outputs, lost
     registers, lost logic); they do not prove a netlist computes the RTL's function. That is the
     equivalence check's job, which trusts nothing but is off by default (design §5).
   - With `lec`, Formality does not finish some timing-driven DesignWare 32-bit dividers and remainders
     (divide@ui32, remainder@i32, remainder@ui32): for divide@ui32 20 of 32 outputs pass, 12 stay
     unverified, none fails, with the default strategy, s1 to s3, and datapath effort `high` and
     `unlimited` (30 minutes each). Those rows are measured, sanity-checked and marked `inconclusive`,
     not proven. Equivalence checking of dividers is known to be hard for general-purpose engines: ABC's
     `cec` needs 172 s for a 32-bit textbook divider and more than 72 h at 48 bits (Scholl and Konrad,
     DATE 2020); the methods that scale exploit the divider's layer structure. Every other divider,
     the signed 32-bit one included, is proven.
   - One corner per kit (Nangate45 has only typical; another ASAP7 VT or corner is another manifest).
   - Areas are cell areas from the Liberty files at 1x scale (ASAP7 also ships 4x LEFs for P&R tools;
     an area from a 4x flow would be 16 times larger); no wires (no wire-load model, no placement).
   - Leakage only: dynamic power needs switching activity, which a primitive alone does not have.
   - Delays are those of a primitive with ideal clocking and fixed driving cell and load; a merged unit
     re-synthesized as a whole will differ.
   - The Yosys flow's equivalence check (with `lec`) is SAT-based (`equiv_simple`/`equiv_induct`,
     `lec_timeout_min`):
     it proves the integer, multiplexer, register and accumulator primitives, but not the float
     primitives (f32 add and multiply leave hundreds of unproven signal pairs) nor every split multiplier
     (16 from 8 on ASAP7); those rows end inconclusive. Formality, with DC's guidance file, proves all of them
     in seconds. Without the check, Yosys characterizes every primitive, floats included.
   - Design Compiler may map arithmetic to DesignWare parts (recorded per row); another site without the
     DesignWare license gets other numbers.
