# `cc-smt`: SMT-LIB terms, op semantics and the solver runner

The passes state proof obligations as SMT-LIB 2 files (Barrett, Fontaine, Tinelli, *The SMT-LIB Standard:
Version 2.6*) and run an external solver on them out of process, through a capped tool runner
(docs/framework/core.md). The file is the certificate: any solver can re-check it. `normalize` (rule
proofs) and `iteration_space` (access and mapping proofs) write their queries with the string helpers;
`datapaths` uses the typed term builder and the op semantics; `merge` will reuse both.

## Scripts

`Script` collects comments, declarations (`constant`, `fun`, `define`) and assertions and renders one
query: `(set-logic ALL)`, `(set-option :produce-models true)`, the declarations, the assertions,
`(check-sat)`, `(get-model)`. Logic `ALL` is the one both solvers accept for any mix of uninterpreted
functions, bit-vectors, floating point and arrays (checked with Bitwuzla 0.9.1 and Z3 5.3.3 in the
toolchain image). String helpers build terms: `bv(w)`, `bv_lit(v, w)`, `and`, `or`, `distinct`, `app`.

What both solvers accept (observed in the image, 2026-11-02): bit-vectors, the IEEE interchange float
sorts (Float16, Float32, Float64, Float128), uninterpreted functions over bit-vectors and over those float
sorts. Bitwuzla rejects other float sorts (they need its `--fpexp` build option), equalities over
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

Bitwuzla has only the interchange sorts, and both solvers must answer every obligation. So:

- **Native formats** (f16, f32, f64) are FP terms with the theory's operations (Brain, Tinelli, Rümmer,
  Wahl, "An automatable formal semantics for IEEE-754 floating-point arithmetic", ARITH 2015), round to
  nearest even, no flush to zero.
- **Carrier formats** (bf16, the f8 formats) are their bit patterns. `to_f32` decodes a pattern exactly
  into Float32 (every value of such a format is a Float32); an operation runs in Float32; `from_f32`
  rounds back to nearest even on the Float32 bits: the significand is shifted by the exponent distance
  (more for results in the format's subnormal range), the guard and sticky bits decide the increment, and
  the packed magnitude `((exponent - 1) << m) + N` carries into the exponent. IEEE formats overflow to
  infinity; finite-NaN (`FN`) formats overflow to their NaN pattern, as ml_dtypes' default conversion
  does (`float8.h`: `ConvertFrom<kSaturate = false, kTruncate = false>`, overflow gives `infinity()`,
  which for `float8_e4m3fn` is the NaN pattern); infinities and NaN become NaN. Saturation exists only as
  explicit `clamp` ops in the graph.
- **Wide sources** (integers of more than 24 bits, f64) to a carrier format go exactly to Float64, then
  to Float32 rounded to odd (truncate, set the last bit if inexact), then to the format: rounding to odd
  first and to nearest second equals one rounding to nearest when the intermediate has at least two more
  bits (Boldo, Melquiond, "Emulation of FMA and correctly rounded sums: proved algorithms using rounding
  to odd", IEEE TC 57(4), 2008). Through Float32 rounded to nearest the lemma below is `sat`: some 32-bit
  integers round differently.
- **Lemmas:** `lemmas(used)` writes one query per encoding used, each `unsat` in Z3, which has the
  native sorts:

  | lemma | statement |
  |---|---|
  | `bf16.decode` | `to_f32(b)` is the exact widening of `(_ FloatingPoint 8 8)` |
  | `bf16.round` | `from_f32(x)` as `(_ FloatingPoint 8 8)` is `((_ to_fp 8 8) RNE x)` |
  | `bf16.{add,sub,mul,div,sqrt}` | the op in Float32, rounded, equals the native bf16 op (double rounding is harmless with 24 ≥ 2·8 + 2 bits; the lemma decides it) |
  | `bf16.from_f64` | the round-to-odd path equals `((_ to_fp 8 8) RNE x)` |
  | `f8e4m3fn.decode` | agrees with `(_ FloatingPoint 4 4)` on subnormals and `(_ FloatingPoint 5 4)` on normals |
  | `f8e4m3fn.round` | agrees with `(_ FloatingPoint 4 4)` up to 240, with `(_ FloatingPoint 5 4)` from 2^-6, and gives the NaN pattern past 464 and for NaN and infinities |

  E4M3FN is no SMT-LIB sort (its top binade is finite), so its lemmas are piecewise against the two IEEE
  sorts whose grids it shares. An encoding without a lemma (an arithmetic op in an FN format, say) is an
  error, never trusted. Measured in the image (2026-11-02, Z3 5.3.3): decodes and roundings under 1 s,
  add 11 s, sub 15 s, mul 66 s, div 98 s.

## Op semantics (`ops.rs`)

`apply(ctx, op, attrs, args, ret)` is the value of a StableHLO element-wise op (without its prefix) on
scalar values `Val` (one term, or two for a complex number) of scalar types `Ty` (`Bool`, `Int { signed,
w }`, `Float(Format)`, `Complex(Format)`, from `cc_ir::Elem`). `convert`, `zero`, `input` (a free
variable of a type) and `same` (equality of results: bit-exact, all NaNs equal) complete it. The source is
the StableHLO specification (openxla/stablehlo `docs/spec.md`); where it leaves a result open, the
encoding follows XLA's element-wise emitter (`xla/service/elemental_ir_emitter.cc`), which is what the
exported models compute:

| op | encoding | source |
|---|---|---|
| `add`, `subtract`, `multiply` (int) | two's complement, wrapping | spec |
| `divide`, `remainder` (int) | truncating; `x / 0 = -1`, `MIN / -1 = MIN`, `x % 0 = x`, `MIN % -1 = 0`; the remainder has the dividend's sign | spec (open cases implementation-defined), XLA `EmitIntegerDivide` / `EmitIntegerRemainder` |
| `abs`, `negate`, `sign` (int) | wrapping (`abs(MIN) = MIN`); sign -1, 0, 1 | spec |
| `maximum`, `minimum` | integers by signedness; floats IEEE 754-2019 `maximum`/`minimum`: NaN if either is NaN, -0 < +0; booleans OR/AND | spec |
| `and`, `or`, `xor`, `not` | bitwise; on booleans logical | spec |
| `compare` | `SIGNED`/`UNSIGNED` on integers; `FLOAT`: the IEEE quiet predicates (false on NaN except `NE`); `GT`/`GE` are `LT`/`LE` with exchanged operands (IEEE compareQuietGreater(a, b) = compareQuietLess(b, a)); complex lexicographic; `TOTALORDER` not encoded (none in use) | spec |
| `select`, `clamp` | `ite`; `min(max(x, lo), hi)` | spec |
| `add`, `subtract`, `multiply`, `divide`, `sqrt` (float) | RNE in the theory (native) or in the carrier (lemmas) | spec (IEEE 754) |
| `remainder` (float) | C `fmod` from the IEEE remainder, corrected by ±\|rhs\| when the signs differ | spec (`d` rounded toward zero) |
| `round_nearest_even`, `floor`, `ceil`, `is_finite`, `sign`, `abs`, `negate` (float) | the theory's `roundToIntegral`, predicates, sign bit | spec |
| `reduce_precision` | round the significand to `mantissa_bits` (ties to even), then overflow to infinity / underflow to zero for `exponent_bits` (native formats) | spec |
| `exponential`, `log`, `log_plus_one`, `tanh`, `rsqrt`, `sine`, `cosine`, `power` (float), complex `multiply`/`divide` | uninterpreted functions per op and format (complex: one for each part) | the spec gives no rounding; a proof then holds for any implementation |
| `convert` int→int | truncation, or extension by the source's signedness | spec (inexact: TBD #180), XLA |
| `convert` int→float | round to nearest even (wide integers to carriers: round to odd first) | spec TBD, XLA `sitofp`/`uitofp` |
| `convert` float→int | toward zero; out of range saturates at the bounds as the source type rounds them; NaN → 0 | spec (truncation), XLA |
| `convert` float→float | exact widening; narrowing rounds to nearest even once (E4M3FN overflow → NaN) | spec TBD, ml_dtypes |
| `convert` bool, complex | 0/1, non-zero is true (NaN included); complex keeps or zeroes the imaginary part | spec |

`tests/smoke/smt.rs` checks the encodings against concrete cases in both solvers: the spec's examples
(`remainder`, `round_nearest_even`), every open case above, ties and overflow at the OCP FP8 boundaries
(448, the tie at 464, 470), subnormal ties, the whole E4M3FN decoding table against an independent
formula, and two negative controls (exchanged `subtract` operands, a bf16 truncation instead of a
rounding) that must be `sat`.

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
  `unsat` and `sat` are stored. `datapaths` keeps it under `build/.datapaths/cache/` (outside `out/`, so
  a rerun keeps it and `chipc clean` removes it) and uses it in the pass only; its verify always asks
  the solver.
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
