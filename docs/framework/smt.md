# `cc-smt`: SMT-LIB terms, op semantics and the solver runner

The passes state proof obligations as SMT-LIB 2 files (Barrett, Fontaine, Tinelli, *The SMT-LIB Standard:
Version 2.6*) and run an external solver on them out of process, through a capped tool runner
(docs/framework/core.md). The file is the certificate: any solver can re-check it. `normalize` (rule
proofs) and `iteration_space` (access and mapping proofs) write their queries with the string helpers;
`datapaths` and `merge` use the typed term builder and the op semantics.

## Scripts

`Script` collects comments, declarations (`constant`, `fun`, `define`) and assertions and renders one
query: `(set-logic ALL)`, `(set-option :produce-models true)`, the declarations, the assertions,
`(check-sat)`, `(get-model)`. Logic `ALL` is the one both solvers accept for any mix of uninterpreted
functions, bit-vectors, floating point and arrays (checked with Bitwuzla 0.9.1 and Z3 5.3.3 in the
toolchain image). String helpers build terms: `bv(w)`, `bv_lit(v, w)`, `and`, `or`, `distinct`, `app`.

What both solvers accept (observed in the image, 2026-11-02): bit-vectors, the IEEE interchange float
sorts (Float16, Float32, Float64, Float128), uninterpreted functions over bit-vectors and over those float
sorts. Bitwuzla rejects other float sorts (they need its `--fpexp` build option; it parses a `to_fp`
into such a sort without that check, so `cc-float` routes a query by both), equalities over
uninterpreted sorts, and `fp.to_ieee_bv`. Queries that both solvers must answer stay within that set.

## Terms (`term.rs`)

`Ctx` is a hash-consed DAG of typed terms (`T`) for one query.

- **Sorts:** `Bool`, `Bv(w)`, `Fp { e, s }` (`s` counts the hidden bit).
- **Constructors check sorts:** `app(op, args)` returns a `SortError` for an ill-sorted application, so
  a wrong query is a Rust error, not a solver error. `Op` lists the operators in use: core (`not`, `and`,
  `or`, `xor`, `=>`, `ite`, `=`, `distinct`), QF_BV (arithmetic, bitwise, shifts, comparisons, `concat`,
  `extract`, extensions), QF_FP (arithmetic with a rounding mode, `fp.rem`, `fp.roundToIntegral`,
  predicates, conversions from bits, floats and integers, `fp.to_sbv`/`fp.to_ubv`), and declared
  uninterpreted functions (`fun`, then `Op::Uf`).
- **Hash-consing and canonical order:** equal applications are one node; the arguments of operators
  whose SMT-LIB semantics is commutative (`and`, `or`, `xor`, `=`, `bvadd`, `bvmul`, `bvand`, `bvor`,
  `bvxor`, `fp.add`, `fp.mul`, `fp.eq`) are sorted by node id. Two sides of an equivalence built from the
  same values therefore share their nodes, and the solver sees a difference only where there is one.
  `and`/`or` drop literal `true`/`false` operands; `ite` with equal branches and `=` of a node with itself
  fold.
- **Bits of a float:** `bits_of(x)` is an auxiliary bit-vector `b` defined by `to_fp(b) = x` and, when
  `x` is NaN, `b` = the canonical quiet NaN (sign 0, top significand bit set). The definition is a side
  condition asserted with the query and makes `b` a function of `x`.
- **Rendering:** `render(script, roots)` declares every variable and function the roots reach, defines
  every application used more than once (`define-fun t<id>`), and asserts the roots and side conditions.
- **Encodings used:** `note("<format>.<what>")` records an encoding that needs a lemma (below);
  `used()` lists them.

## Floats (`float.rs`)

Bitwuzla has only the interchange sorts, and both solvers must answer every obligation. The numerics of
design §6.4 are stated directly in SMT-LIB, in sorts only Z3 has, by the encoding lemmas; the queries use
encodings in interchange sorts that the lemmas prove equal to that statement.

- **Subnormal rule** (design §6.4): f32 and bf16 flush (`Format::flushes`), every other format keeps
  gradual underflow. In a flushing format every operator reads a subnormal operand as zero of its sign
  (`daz`: `ite(fp.isSubnormal x, ±0, x)` on an FP term, a zero exponent field on a bit pattern), and a
  result whose value, rounded to the format's precision with an unbounded exponent, is below `2^emin` is
  zero of its sign (FTZ). That rounded value is below `2^emin` exactly when `|x| < 2^emin −
  2^(emin − m − 2)` (`Format::flush_threshold`; the midpoint under `2^emin` rounds up to it, its even
  neighbour). The rule differs from "flush what gradual underflow makes subnormal" only for results in
  `[2^emin − 2^(emin − m − 1), 2^emin − 2^(emin − m − 2))`, which IEEE rounds up to `2^emin` and the rule
  flushes (the smoke tests have an f32 and a bf16 case).
- **Native formats** (f16, f32, f64) are FP terms with the theory's operations (Brain, Tinelli, Rümmer,
  Wahl, "An automatable formal semantics for IEEE-754 floating-point arithmetic", ARITH 2015), round to
  nearest even. f16 and f64 use them as they are. The f32 arithmetic (`arith`) is:
  - add, subtract: the Float32 operation, flushed when subnormal: a sum that underflows is exact (Hauser,
    "Handling floating-point exceptions in numeric programs", ACM TOPLAS 18(2), 1996), so it is below
    `2^emin` exactly when it is subnormal;
  - multiply, divide (`scaled_op`): one Float32 operation. When the exponent fields say the result may
    be below `2^-125` (`e_x + e_y < -125`, or `e_x − e_y < -124`), the left operand is scaled by `2^32`
    (its exponent field moved: exact, it is then below `2^4`); the result `r` is then the rounding to 24
    bits of the exact value times `2^32`, normal wherever the decision matters, and the result is zero of
    its sign when `|r| < 2^-94`, else `r · 2^-32` (exact). Otherwise the result is at least `2^-125` and
    the operation's own. A Float64 product or quotient would state the same, but made Z3 take minutes to
    find a counterexample in `merge`'s sanity queries (227 s for one with two divides) where this
    encoding takes seconds;
  - square root: the Float32 operation (the root of a normal value is normal);
  - the remainder, which is exact: flushed when subnormal.
- **Carrier formats** (bf16, the f8 formats) are their bit patterns. `to_f32` decodes a pattern exactly
  into Float32 (every value of such a format is a Float32); an operation runs in Float32; `from_f32`
  rounds back to nearest even on the Float32 bits: the significand is shifted by the exponent distance
  (more for results in the format's subnormal range), the guard and sticky bits decide the increment, and
  the packed magnitude `((exponent - 1) << m) + N` carries into the exponent. bf16 flushes: zero of its
  sign below the threshold (an f32 test, so the f32 result must round like the exact one there; the
  lemmas decide it). An arithmetic result overflows to infinity (IEEE formats) or to the NaN pattern
  (finite-NaN `FN` formats: design §6.4, an E4M3FN result beyond 448 is NaN, as ml_dtypes' default
  conversion, `float8.h`, `ConvertFrom<kSaturate = false>`); infinities and NaN of an FN format become NaN.
- **Conversions into an 8-bit format saturate** (`convert_from_f32`; design §6.4, OCP OFP8 Rev 1.1's
  saturating mode): overflow and infinities give the largest finite value of the sign (448, 57344), NaN
  stays NaN, from floats, integers and booleans alike. The models clip to ±448 before converting, so both
  modes agree on their data.
- **Wide sources** (integers of more than 24 bits, f64) to a carrier format go exactly to Float64, then
  to Float32 rounded to odd (truncate, set the last bit if inexact), then to the format: rounding to odd
  first and to nearest second equals one rounding to nearest when the intermediate has at least two more
  bits (Boldo, Melquiond, "Emulation of FMA and correctly rounded sums: proved algorithms using rounding
  to odd", IEEE TC 57(4), 2008). Through Float32 rounded to nearest the lemma below is `sat`: some 32-bit
  integers round differently. f64 to f32 is `native_from_f64`.
- **The arithmetic without its multiplier:** `arith_exact(ctx, f, op, args, v)` is `arith` with the
  theory's operation replaced by its definition, the rounding of `v` (a Float64 term the caller
  constrains to be the exact result); every decision of the encoding is then a function of `v`. Proofs
  that would relate two multipliers (or dividers, roots) of different sorts use it: the f32 glue proofs of
  `cc-float` (docs/framework/float.md), with the exact value constrained to its binade range.
- **Lemmas:** `lemmas(used)` writes one query per encoding used, each `unsat` in Z3, which has every
  sort. The reference rounds in the format's own sort with two more exponent bits (rounding with an
  unbounded exponent: no operation of two values underflows or overflows there), flushes below `2^emin`
  keeping the sign, and narrows exactly; gradual formats round in their own sort.

  | lemma | statement | Z3 |
  |---|---|---|
  | `bf16.decode` | `to_f32(b)` is the exact widening of `(_ FloatingPoint 8 8)` | 2 s |
  | `bf16.round` | `from_f32(x)` is `x` rounded in `(_ FloatingPoint 10 8)`, flushed, narrowed to `(_ FloatingPoint 8 8)` | 2 s |
  | `bf16.{mul,div,sqrt}` | DAZ, the op in Float32 and `from_f32` equal the op in `(_ FloatingPoint 10 8)` on the DAZ operands, flushed and narrowed | 42, 48, 3 s |
  | `bf16.{add,sub}` | the same against the op in `(_ FloatingPoint 8 8)` flushed when subnormal: Z3 has no addition with more exponent than significand bits, and an underflowing sum is exact | 5, 4 s |
  | `bf16.from_f64` | the round-to-odd path equals the flushed rounding from Float64 | 2 s |
  | `f32.{add,sub}` | a Float32 sum or difference of DAZ operands that is subnormal is exact (rounding up and down agree): then it is the exact value, below `2^emin` with at most 24 bits, so flushing it when subnormal is the rule, and any other result is the rounding with an unbounded exponent (stating the whole op against `(_ FloatingPoint 10 24)` also holds, in 46 s) | 1 s |
  | `f32.sqrt` | the Float32 square root of a DAZ operand is never subnormal | 1 s |
  | `f32.{mul,div}` | for normal DAZ operands and every exact value `v` in the range their exponents give (`[2^(e_x+e_y), 2^(e_x+e_y+2))`, `(2^(e_x−e_y−1), 2^(e_x−e_y+1))`, the operands' sign), `scaled_op` with the Float32 operation replaced by its definition (the rounding of `v`, scaled) equals `v` rounded in `(_ FloatingPoint 10 24)`, flushed and narrowed. `v` ranges over Float64: both sides are constant between dyadic breakpoints of at most 26 significant bits, all Float64 values, so this covers every real. Zero, infinite and NaN operands give zero, infinity or NaN, never flushed (smoke cases) | 1, 1 s |
  | `f32.from_f64` | `native_from_f64(x)` is `x` rounded in `(_ FloatingPoint 10 24)`, flushed and narrowed, for every Float64 `x` (the f64 to f32 conversion) | 1 s |
  | `f8e4m3fn.decode` | agrees with `(_ FloatingPoint 4 4)` on subnormals and `(_ FloatingPoint 5 4)` on normals | 2 s |
  | `f8e4m3fn.round` | agrees with `(_ FloatingPoint 4 4)` up to 240, with `(_ FloatingPoint 5 4)` from 2^-6, and gives the NaN pattern past 464 and for NaN and infinities | 2 s |
  | `f8e4m3fn.saturate` | as `round`, but the largest finite pattern of the sign past 464 and for infinities | 2 s |
  | `f8e5m2.{decode,round,saturate}` | as bf16's (gradual), `saturate` with infinities replaced by ±57344 | 2 s |

  E4M3FN is no SMT-LIB sort (its top binade is finite), so its lemmas are piecewise against the two IEEE
  sorts whose grids it shares. An encoding without a lemma (an arithmetic op in an FN format, say) is an
  error, never trusted. The f32 multiply and divide lemmas leave the multiplier out (the operation is the
  rounding of the exact value by the theory's definition): stated against the operation in `(_
  FloatingPoint 10 24)`, Z3 relates two multipliers (or dividers) and does not decide it in 15 minutes.
  Their negative controls (the flush limit one binade off) are `sat` (smoke tests). Measured in the
  image (Z3 5.3.3, iacoma227, 2026-11-05), each lemma alone.

## Verilog to SMT-LIB

What chiplet-compiler writes in Verilog (the float glue of design §4.12) is proven on the Verilog
itself: Yosys (`read_verilog -sv`, `hierarchy -check`, `proc`) reads the module with blackbox stubs in
place of the third-party cores, `cutpoint <module>/<cell>` replaces each core instance's outputs by free
values, `check -assert` rejects an unclean design, and `write_smt2 -stbv -wires` writes it: the state
(inputs and free values) is one bit-vector sort `|<m>_s|`, every named wire a function `|<m>_n
<wire>|` of it (1-bit wires as `Bool`). The query declares one state `rtl.state`, binds a variable
`rtl.<wire>` to each port and core wire, and adds the term DAG's assertions over those variables: the
cores' contracts and the negated claim against `ops::apply` (docs/framework/float.md, section 6).
Yosys 0.33 has no `cutpoint -blackbox`, so the core cells are named. Both solvers accept the format
(checked with Bitwuzla 0.9.1 and Z3 5.3.3 in image 0.5.0).

## Op semantics (`ops.rs`)

`apply(ctx, op, attrs, args, ret)` is the value of a StableHLO element-wise op (without its prefix) on
scalar values `Val` (one term, or two for a complex number) of scalar types `Ty` (`Bool`, `Int { signed,
w }`, `Float(Format)`, `Complex(Format)`, from `cc_ir::Elem`). `convert`, `zero`, `input` (a free
variable of a type) and `same` (equality of results: bit-exact, all NaNs equal) complete it. The source is
the StableHLO specification (openxla/stablehlo `docs/spec.md`); where it leaves a result open, the
encoding follows XLA's element-wise emitter (`xla/service/elemental_ir_emitter.cc`), which is what the
exported models compute. Floats follow the numerics of design §6.4 ("Floats" above): in f32 and bf16 every
operator reads a subnormal operand as zero of its sign; the uninterpreted functions take their operands as
they are, so a proof holds whether a core flushes its inputs or not.

| op | encoding | source |
|---|---|---|
| `add`, `subtract`, `multiply` (int) | two's complement, wrapping | spec |
| `divide`, `remainder` (int) | truncating; `x / 0 = -1`, `MIN / -1 = MIN`, `x % 0 = x`, `MIN % -1 = 0`; the remainder has the dividend's sign | spec (open cases implementation-defined), XLA `EmitIntegerDivide` / `EmitIntegerRemainder` |
| `abs`, `negate`, `sign` (int) | wrapping (`abs(MIN) = MIN`); sign -1, 0, 1 | spec |
| `maximum`, `minimum` | integers by signedness; floats IEEE 754-2019 `maximum`/`minimum`: NaN if either is NaN, -0 < +0; booleans OR/AND | spec |
| `and`, `or`, `xor`, `not` | bitwise; on booleans logical | spec |
| `compare` | `SIGNED`/`UNSIGNED` on integers; `FLOAT`: the IEEE quiet predicates (false on NaN except `NE`); `GT`/`GE` are `LT`/`LE` with exchanged operands (IEEE compareQuietGreater(a, b) = compareQuietLess(b, a)); complex lexicographic; `TOTALORDER` not encoded (none in use) | spec |
| `select`, `clamp` | `ite`; `min(max(x, lo), hi)` | spec |
| `add`, `subtract`, `multiply`, `divide`, `sqrt` (float) | RNE (`float::arith`); f32 and bf16 flush (lemmas) | spec (IEEE 754), design §6.4 |
| `remainder` (float) | C `fmod` from the IEEE remainder, corrected by ±\|rhs\| when the signs differ; flushed in f32 and bf16 | spec (`d` rounded toward zero) |
| `round_nearest_even`, `floor`, `ceil`, `is_finite`, `sign`, `abs`, `negate` (float) | the theory's `roundToIntegral`, predicates, sign bit | spec |
| `reduce_precision` | round the significand to `mantissa_bits` (ties to even), then overflow to infinity / underflow to zero for `exponent_bits` (native formats) | spec |
| `exponential`, `log`, `log_plus_one`, `tanh`, `rsqrt`, `sine`, `cosine`, `power` (float), complex `multiply`/`divide` | uninterpreted functions per op and format (complex: one for each part) | the spec gives no rounding; a proof then holds for any implementation |
| `convert` int→int | truncation, or extension by the source's signedness | spec (inexact: TBD #180), XLA |
| `convert` int→float | round to nearest even (wide integers to carriers: round to odd first); into FP8 saturating | spec TBD, XLA `sitofp`/`uitofp`, design §6.4 |
| `convert` float→int | toward zero; out of range saturates at the bounds as the source type rounds them; NaN → 0 | spec (truncation), XLA |
| `convert` float→float | exact widening (a flushing source reads subnormals as zero); narrowing rounds to nearest even once with the target's subnormal rule; into FP8 saturating: overflow and infinities give ±max, NaN stays NaN | spec TBD, design §6.4 (OCP OFP8 saturating mode) |
| `convert` bool, complex | 0/1, non-zero is true (NaN included); complex keeps or zeroes the imaginary part | spec |

**`rescale`** is the one op that is not StableHLO's: `merge` puts it in place of a rescaling chain
`round_nearest_even(convert(x) · s)` for its `fixed` requantization (design §6.3, docs/passes/merge.md
§4.11). `rescale(x, s)` with `x` a signed integer of at most 32 bits and `s` an f32: `scale_parts`
decodes the significand `M` (24 bits, the implicit one when the exponent field `e` is not zero) and
`k = 150 − max(e, 1)`, so `|s| = M·2^-k` exactly; `rescale_product` forms `P = ±x·M` in `n + 25` bits;
`rescale_shift` gives `(P + 2^(k−1)) >> k` (arithmetic, rounding half up; shifts past the product's
width give 0) for `k ≥ 1`, `P << −k` for `−7 ≤ k ≤ 0` and saturation beyond, then saturates to i32 and
converts to f32 (round to nearest even). `rescale_reference` is the same function in plain integer
arithmetic; the RTL (`characterize`'s `rescale@i<n>`) is checked against it.

`tests/smoke/smt.rs` checks the encodings against concrete cases in both solvers: the spec's examples
(`remainder`, `round_nearest_even`), every open case above, ties and saturation at the OCP FP8 boundaries
(448, the tie at 464, 470, infinities, NaN, E5M2's 57344), subnormal ties, the flush rule (f32 and bf16
products and quotients below 2^-126, the sign kept, the case IEEE rounds up to 2^-126 and the rule
flushes, one it keeps, subnormal operands of add, negate, compare and convert read as zero), the whole
E4M3FN decoding table against an independent formula, `rescale` against its reference (ties, left
shifts, saturation, zero, subnormal and infinite scales, three widths), and four negative controls that
must be `sat`: exchanged `subtract` operands, a bf16 truncation instead of a rounding, an IEEE f32
multiplier (gradual underflow) against the f32 spec, and a bf16 multiply on an IEEE f32 multiplier
against the bf16 spec. It also runs Z3 on every encoding lemma it decides in seconds (all but bf16's multiply
and divide, about 45 s each), formats no model uses included.

## Reduction order and its error bound

A float sum is not associative, so the order of a reduction is part of what it computes (design §6.4).
`normalize` fixes ω per reduction, `datapaths` proves one ω step and asks whether the combiner is
commutative and associative (`<t>.order.smt2`). When it is not (ω binds), the datapath records the order
the hardware computes, which is ω (`order[].hardware`), and `datapaths`' `verify.json` states, per
accumulator that is a float sum, how far another order of the same reduction (the model's: XLA leaves it
open) can be:

`|hw − other| ≤ (γ(d) + γ(n)) · S + n · 2^emin · (2 + γ(d) + γ(n))`

- `n`: the reduced elements (the product of ω's dim extents, symbolic); the init value is one more term;
- `S = |init| + Σ|x_i|` over the values the operators read (subnormals of f32 and bf16 as zero);
- `u = 2^-p` (`p` significand bits, the hidden bit included), `γ(k) = k·u / (1 − k·u)`;
- `d`: the additions on the longest path of the hardware's schedule, `n` for `seq` and
  `(b − 1)·⌈log_b n⌉ + 1` for `tree:<b>`; any order has at most `n`.

Every order computes the exact sum within `γ(depth) · S` (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., SIAM 2002, §4.2, eq. 4.4, from `fl(a + b) = (a + b)(1 + δ)`, `|δ| ≤ u`); the bound
adds both sides' errors. A flushing format adds an absolute error below `2^emin` per flushed addition,
carried through later roundings by at most `1 + γ` (Higham §2.1's underflow model); in a gradual format
an underflowing sum is exact (Hauser 1996), so the term is absent there. Assumptions: round to nearest,
no partial sum overflows, `n·u < 1`. Other bound combiners (a float product, an arg-max) are listed with
no bound. The bound documents the difference; the proofs stay bit-exact against ω.

## Running

- `run_batch(runner, location, solver, dir, files, timeout_s)` runs `solver` on every file in `dir` (names
  may contain subdirectories) in one process (`sh -c`: a container start per query would dominate), each
  under `timeout <timeout_s>`, then `<solver> --version`. `location` is `Image` (the solvers on the
  toolchain image's `PATH`, `dir` mounted read-only at `/work/smt`, with a Docker runner) or `Host`
  (binary paths, with a host runner).
- `run_sharded(…, shards)` splits the files round-robin into `shards` batches run at once, each with an
  equal share of the runner's memory and CPU caps (so together they stay within the pass's
  declaration), and merges the answers; every batch must report the same version.
- `solver_version` runs `<solver> --version`.
- `Cache` stores verdicts by (solver, version, query hash) in a directory, one file per query; only
  `unsat` and `sat` are stored, so a timeout or an error is always asked again. Writes go to a unique
  temporary file and are renamed, so processes may share the directory.
- `solve_cached` (files with their texts) and `run_cached` (files read from `dir`; the result shape of
  `run_sharded`) answer what the cache holds and run the rest sharded, storing the new answers.
- **One cache for every pass and both sides.** `normalize`, `iteration_space`, `datapaths` and `merge`
  use it in `run` and in `verify`, in the directory `paths.smt_cache` (default `<build_dir>/.smt-cache`;
  `scripts/build-area` points every worktree of a build host at `/scratch/$USER/chiplet-compiler/
  smt-cache`). `verify` still writes every query from its own inputs and compares it byte for byte with
  the stored one; only an identical query already answered by the same solver and version is not run
  again. Constant queries (rewrite rules, encoding lemmas, the `rescale` lemmas) are therefore solved once
  per solver version.
- `query_hash` is the blake3 hex of a query's text.

The result of one query is a `Verdict`:

| Verdict | From |
|---|---|
| `unsat` | first output line `unsat` (Z3 then prints an error for `get-model`; it is ignored) |
| `sat { model }` | first line `sat`; the model is the rest of the output |
| `unknown` | first line `unknown` |
| `timeout` | exit status 124 of `timeout` |
| `error { message }` | anything else |

Only `unsat` proves an equivalence (its negation is unsatisfiable); `unknown`, `timeout` and `error` never
count as a proof.

## Configuration

`[tools.smt]`: `solver` (`bitwuzla` or `z3`: the one that proves; the other re-checks), `timeout_s` (per
query), `runner` (`docker` or `host`), `z3` and `bitwuzla` (host binaries, resolved on `PATH` unless
absolute). The image's binaries come from `[tools.docker.build]` (`z3_version`, `bitwuzla_version`, with
sha256); CI downloads the same releases for the smoke tests (`CC_SMT_RUNNER=host`). The z3 release build
needs glibc 2.39, so on the RHEL 8 build host the solvers run in the image. Bitwuzla is the default for
both QF_BV and QF_FP (word-level rewriting and SymFPU bit-blasting: Niemetz, Preiner, CAV 2023); the
encoding lemmas are Z3's, as only Z3 has the native sorts they compare with.

The solvers are adapters (`SmtSolver`, docs/framework/adapters.md): `cc_smt::SOLVERS` lists `Bitwuzla`
and `Z3`, each declaring its capabilities (bit-vectors, uninterpreted functions, the interchange float
sorts, float sorts of any width, the lemmas) and its program; `Solver::adapter()` maps the identifier
stored in certificates to its implementation. The encoding lemmas of `datapaths` and the composed array
steps of `merge` go to `lemma_solver()`, the first solver that declares `lemmas` (Z3). The smoke test
`adapters::smt_solvers_conform` gives every solver the same queries and checks the declarations: a
`(_ FloatingPoint 8 8)` query is proven by Z3 and refused by Bitwuzla.
