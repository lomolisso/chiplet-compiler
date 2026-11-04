# `softfloat`

1. **Purpose.** The float primitives of the functional units as integer-only hardware: every float op
   of the primitive catalog (design §4.9) is an Allo function over bit patterns (`UInt(n)`), so Allo's
   `llvm` simulator and its XLS backend, which reject float types, accept every unit (design §4.12, §6.4).
   Each primitive is simulated against a reference, taken through Allo's XLS backend, xlscc, `opt_main`
   and `codegen_main` to Verilog, and evaluated on its optimized XLS IR. An exactly specified op is
   proven equal to its semantics in `cc_smt::ops` (the semantics every datapath proof uses) where the
   solver finishes; an approximate op (exponential, log, tanh, ...), whose result the datapath proofs
   leave open, is within one ulp of the exact value (correctly rounded where that is cheap).
   Position: a parallel branch that depends only on the environment; `allo_emit` calls the library and
   `characterize` prices its Verilog.

2. **Background.**
   - Design §4.9, §4.12, §6.4; the scalar op semantics (docs/framework/smt.md); Allo and XLS
     (docs/tools/allo.md, docs/tools/xls.md).
   - IEEE 754-2019 (binary formats, round to nearest ties to even, gradual underflow, `maximum` and
     `minimum` of §9.6); the OCP 8-bit floating point specification (OFP8) Rev. 1.1 for E4M3FN (no
     infinities, NaN `S.1111.111`, largest finite 448, the overflow boundary of Appendix A);
     Micikevicius et al., "FP8 formats for deep learning", arXiv:2209.05433, 2022.
   - Algorithms (Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., 2018; Goldberg, ACM
     Computing Surveys 23(1), 1991): alignment with guard and sticky bits, leading-zero normalization,
     restoring division and digit-by-digit square root with the remainder as sticky bit; Hauser's Berkeley
     SoftFloat for the structure of an integer-only implementation.
   - The reference: float64 holds every value of these formats exactly, and one float64 operation then
     one rounding is the correctly rounded result for `+ - * /` and `sqrt` because 53 >= 2p + 2 for every
     p <= 24 (Figueroa, "When is double rounding innocuous?", SIGNUM Newsletter 30(3), 1995).
   - The SMT-LIB floating-point theory (Brain, Tinelli, Rümmer, Wahl, ARITH 2015) defines `fp.mul` as the
     exact product rounded once; that is the multiply proofs' spec (§4).
   - Approximate ops: Tang, "Table-driven implementation of the exponential function in IEEE
     floating-point arithmetic", ACM TOMS 15(2), 1989, and "Table-driven implementation of the logarithm
     function", ACM TOMS 16(4), 1990 (argument reduction by a table); Muller, *Elementary Functions:
     Algorithms and Implementation*, 3rd ed., 2016 (range reduction, faithful rounding, the ulp); the
     special values of ISO C11 Annex F (`exp(-inf) = +0`, `log(0) = -inf`, `log1p(-1) = -inf`,
     `tanh(+-inf) = +-1`, `sin(+-inf)` and `cos(+-inf)` NaN, `pow` as F.10.4.4, `rsqrt` as `1/sqrt`).
   - Sine and cosine: Payne and Hanek, "Radian reduction for trigonometric functions", ACM SIGNUM
     Newsletter 18(1), 1983, and Ng, "Argument reduction for huge arguments: good to the last bit", Sun
     Microsystems, 1992 (reduction by a window of the bits of 2/pi); Muller, *Elementary Functions*, ch. 11
     (the worst cases of the reduction, found with Kahan's method; f32: `16367173 * 2^72`). pi from
     Machin's formula `pi = 16 atan(1/5) - 4 atan(1/239)` in integer arithmetic.

3. **Inputs and outputs.**
   - Inputs (not artifacts): the library `python/cc_softfloat` and the toolchain image (Allo, XLS); their
     hashes are in the cache key (`fingerprints`). The primitive list comes from `characterize`'s catalog
     (`catalog::catalog` on `float_formats`, `int_widths`, `ops`).
   - `out/softfloat/ops/<id>/` per primitive: `kernel.py` (the generated Allo function), `kernel.mlir`
     (Allo's module after inlining), `xls/test_block.cpp` (Allo's XLS C++), `fu.ir` (xlscc), `fu.opt.ir`
     (`opt_main`), `rtl.v` and `rtl.sig.textproto` (`codegen_main --generator=combinational`, module
     `<id>`), `vectors.npz` (operands, reference, `llvm` and IR results), `result.json` (schema
     `softfloat_result`), and for an exactly specified op `proof.smt2` and `proof.result` (schema
     `datapaths_result`).
   - `out/softfloat/coverage.json` (schema `softfloat_coverage`): per primitive its key, id, op, types,
     evidence (`proven`: the IR equals the semantics for every input; `exhaustive`: every input evaluated
     on the IR and on `llvm` matched the reference; `tested`: edge cases and random inputs matched),
     accuracy (`exact`, `correctly_rounded`: within 1/2 ulp, `faithful`: within 1 ulp), for an approximate
     op its design's error bound and its largest measured error in ulps, vector count, IR size and proof
     verdict; and the keys of any approximate op of the catalog the library does not build (none today).
   - `out/softfloat/summary.json` (schema `softfloat_summary`): counts, vectors, the library hash (the
     key `allo_emit` caches on), the image and the solver; `views/softfloat.md`.

4. **Algorithm.**
   1. *Primitives.* The catalog's float op primitives (every element-wise op that `cc_smt::ops` defines
      on a float type, conversions to and from every integer width and bool). Ops whose result the spec
      leaves to an approximate function (uninterpreted in the datapath proofs) have an accuracy of this
      library's choice (§4.6): `exponential`, `log`, `log_plus_one`, `tanh`, `rsqrt`, `sine`, `cosine` and
      `power`, every one the catalog has.
   2. *Generation.* `cc_softfloat.ops` writes one flat Allo function per primitive from shared blocks
      (`cc_softfloat.emit`): decode (sign, significand with its hidden bit, exponent, NaN/infinity/zero
      flags; subnormals get exponent 1), log-depth normalization (one conditional shift per power of
      two, optionally capped at the subnormal boundary), right shift with sticky, and `round_pack`: one
      rounding to nearest, ties to even, from guard and sticky bits, with the value first shifted into
      the subnormal position when it is below the normal range, and overflow to infinity (to NaN in
      E4M3FN, as `cc_smt` and ml_dtypes' non-saturating conversion do; saturation is an explicit
      `clamp` in the graph). The ops:
      - `add`, `subtract`: order by magnitude, align with sticky, add or subtract, normalize (stopping
        at the subnormal exponent, so no second shifter), round; an exact zero is `+0` unless both
        operands are `-0`.
      - `multiply`: exact significand product, normalized, rounded once.
      - `divide`: restoring radix-2 division of the normalized significands, `m + 3` quotient bits and
        the remainder as sticky bit.
      - `sqrt`: digit-by-digit square root of the normalized significand with the exponent made even,
        `m + 3` root bits, the remainder as sticky bit (never subnormal, never overflows).
      - `remainder` (C `fmod`, exact): the divisor's significand normalized, then one compare-subtract
        per binade of the exponent difference; the result keeps the dividend's sign.
      - `compare` (`EQ`, `NE`, `LT`, `LE`; IEEE quiet predicates, `-0 == +0`), `maximum` and `minimum`
        (NaN if either is NaN, `-0 < +0`), `clamp` (`minimum(maximum(x, lo), hi)`), by an unsigned key
        with the order of the values.
      - `negate`, `abs` on the sign bit; `sign`; `is_finite`; `select`.
      - `round_nearest_even`, `floor`, `ceil`: the magnitude's fraction bits rounded on the pattern (a
        carry into the exponent is exact).
      - `reduce_precision` (XLA): the significand rounded on the pattern, then overflow to infinity and
        underflow below the reduced normal range to zero.
      - `convert`: integer to float rounds to nearest even; float to integer truncates, saturates and
        maps NaN to 0 (XLA); float to bool is "not zero"; float to float is exact when widening and one
        rounding otherwise.
      Every NaN result is the format's canonical NaN; the proofs and the checks compare NaN as one class.
   3. *Allo constraints the generator keeps* (found building the library; docs/tools/allo.md): bits by
      shifts and masks (slice syntax emits Vivado `ap_int` code xlscc rejects); helper calls inlined by
      MLIR's `inline` pass before the XLS backend (it emits calls through output pointers against
      by-value signatures); no 64-bit types (`UInt(64) + 1` becomes a 66-bit value xlscc cannot
      convert back); literals below 2^31 and wider constants built in a variable (Allo types literals as
      32-bit signed); `~` written as XOR with all ones (Allo's `~` returns its operand); a spare top bit
      on every variable compared with `<`/`>=` (Allo's `llvm` lowering compares unsigned variables as
      signed); every variable shift amount below the operand width (LLVM makes a larger shift poison).
      Allo widens a product to the sum of its operand widths and emits 8-, 16-, 32- and 64-bit variables
      as native C++ integers, which xlscc cannot assign from a value wider than 64 bits: the approximate
      ops, whose fixed-point products reach about 100 bits, widen every such variable by one bit. A
      table is a module-level `uint32` NumPy array per 32-bit slice, read in the function (a constant
      array in MLIR, an array literal in XLS), indexed by a variable with a spare top bit (Allo's `llvm`
      lowering sign-extends an index). The C++ includes `xls_builtin.h`, which nothing ships: an empty
      header in `python/cc_softfloat/include`.
   4. *Build* (`python -m cc_softfloat build`, `batch` primitives per container): vectors (every input
      combination when the operands have at most `exhaustive_bits` bits, otherwise the cross product of
      each type's edge cases plus `random_vectors` seeded random operand sets, half of them for binary
      float ops with nearby exponents); the reference on them; Allo `llvm` on them; Allo's XLS C++,
      xlscc, `opt_main`, `codegen_main`; `eval_ir_main` (LLVM JIT) on every vector. A primitive passes
      when `llvm` equals the reference on every vector (NaN as a class; for an approximate op, within its
      accuracy limit, §4.6) and the IR equals `llvm` bit for bit.
   5. *Proof.* `cc_smt::xls` translates `fu.opt.ir` into SMT terms (docs/framework/smt.md); the query
      asserts that the circuit differs from `cc_smt::ops::apply` for some input. For `multiply` in IEEE
      formats a direct query is a multiplier-equivalence problem bit-blasting does not finish (f16 and
      f32 ran past 25 minutes in both solvers); the circuit's product is then an uninterpreted function,
      the spec the exact product of the same function of the operands' significands, normalized exactly
      into a Float64 value and rounded once by `to_fp RNE`, with the one true fact that the product is
      zero exactly when a factor is. A proof for every such function is a proof for the product, and the
      spec is `fp.mul` by the SMT-LIB definition (f32: Bitwuzla seconds, Z3 minutes). The solver is
      `tools.smt.solver` (Bitwuzla) with `proof_timeout_s`; `unsat` gives `proven`, a timeout leaves
      `exhaustive` or `tested`, and `sat` fails the pass with the counterexample.
   6. *Approximate ops* (`cc_softfloat.approx`). Accuracy: within one ulp of the exact value in every
      format, IEEE subnormals, the special values of C11 Annex F, and in E4M3FN NaN wherever the value
      is infinite or overflows (it has no infinities). The ulp of a value is the spacing of the format in
      its binade (the subnormal spacing below the normal range; Goldberg's definition).
      - 8-bit formats, unary ops: a table of the correctly rounded result of each of the 256 inputs, computed from
        NumPy's float64 value rounded once (§4.6 checks that no entry lies within 2^-30 ulp of a rounding
        boundary, so the float64 value's own error cannot change it).
      - `rsqrt`, correctly rounded in every format: with `x = m * 2^(2e)`, `m` in [1, 4) and integer
        significand `M = m * 2^fm`, an exact restoring division gives `D = floor(2^(2N + fm) / M)` and an
        exact integer square root `S = floor(sqrt(D))`; since `floor(sqrt(floor(y))) = floor(sqrt(y))`,
        `S = floor(2^N / sqrt(m))`, and the value is inexact exactly when either step leaves a
        remainder. With `N = p + 1` that is a round bit and a sticky bit: rounding to nearest even is
        exact. `rsqrt(+-0) = +-inf`, `rsqrt(+inf) = +0`, negative inputs give NaN.
      - `exponential` (Tang): `|x|` as fixed point with `F = p + 8` fraction bits; `n = floor(x * 2^t /
        ln 2)` from an estimate corrected by at most two subtractions, `r = x - n ln2 / 2^t` exact in
        `[0, ln2 / 2^t)` against a rounded-down `ln2 / 2^t` with `I + t + 3` extra bits (`|x| < 2^I`,
        beyond which the result is `+inf` or `+0`), `k = floor(n / 2^t)`, `j = n mod 2^t`; then
        `exp(x) = 2^k * T[j] * P(r)` with the table `T[j] = 2^(j / 2^t)` and the Taylor polynomial `P` of
        degree `d` in Horner form, all fixed point, rounded once at exponent `k` (so overflow and
        subnormal results come out of the same rounding). f32 `t = 6, d = 4`; bf16 `t = 4, d = 3`; f16
        `t = 5, d = 3`.
      - `log` (Tang): `x = m * 2^e`; `i` = the top `t + 1` fraction bits of `m` rounded to `t` bits; a
        table gives `c_i ~ 1 / (1 + i / 2^t)` with `t + 5` fraction bits (exactly 1 at `i = 0` and 1/2 at
        `i = 2^t`) and `L_i = -ln c_i`, so `r = m c_i - 1` is exact with `|r| <= 2^-(t+1) * 1.125`, and
        `log(x) = e ln2 + L_i + r Q(r)`, `Q(r) = log1p(r) / r` by Taylor in Horner form on `|r|` with the
        sign of each step from the sign of `r`. Near `x = 1` (`e = 0, i = 0` or `e = -1, i = 2^t`) the
        first two terms are zero and `r Q(r)` is normalized alone, so the relative accuracy holds for
        every `r`; elsewhere `|log x| >= 2^-(t+2)` and the sum has `p + t + 9` fraction bits. f32
        `t = 7, d = 4`; bf16 `t = 4, d = 3`; f16 `t = 5, d = 3`.
      - `log_plus_one`: `|x| < 2^-(t+2)`: `x Q(x)`. Otherwise `u = 1 + x` exact in fixed point below
        `x = 2^(p+7)` (above it `log(1 + x) - log(x) < 2^-(p+7)`, so `u = x`), cut to `p + t + 10`
        significant bits, into the log core. `log1p(+-0) = +-0`, `log1p(-1) = -inf`, below `-1` NaN.
      - `tanh`: odd; `|x| < 2^-ceil(p/2)`: `x` (the error `|x|^3 / 3` is below a third of an ulp);
        `|x| >= (p + 2) ln2 / 2`: `+-1` (`1 - tanh` is below half an ulp of 1); otherwise
        `N = 1 - exp(-2|x|)` from the exp core with `p + ceil(p/2) + 8` fraction bits (enough for the
        cancellation at the smallest `|x|`) and `tanh = N / (2 - N)` by restoring division with
        `p + 6` quotient bits and a sticky bit.
      - `sine`, `cosine`: one circuit per op (shared generator). `|x| < 1/2`: `r = |x|` exactly, quadrant
        0. Otherwise Payne-Hanek: with `|x| = M * 2^E` (`M` the `p`-bit significand), a table indexed by
        the exponent gives the bits `b_i` of `2/pi = sum b_i 2^-i` for `E - 1 <= i <= E + F` (lower bits
        add multiples of 4 to `x * 2/pi`, which leave the quadrant and the reduced value unchanged), so
        the exact product `M * window` holds `x * 2/pi mod 4` with `F` fraction bits; rounding it to the
        nearest integer `q` leaves `f` in `[-1/2, 1/2]`, normalized and multiplied by `pi/2`: `r = f pi/2`,
        `|r| <= pi/4`. The fraction keeps `F = 2p + K + G + 6` bits, where every float of the format with
        `|x| >= 1/2` has `||x * 2/pi|| > 2^-K` (`approx.TRIG_WORST`, from enumerating every input with
        `consts.trig_worst_case`: K = 30 for f32, 17 for f16, 14 for bf16), so `r` has its relative
        accuracy even at the worst case. Then `sin r = r S(r^2)` and `cos r = C(r^2)` (Taylor in Horner
        form on `z = r^2`, one multiplier chain with the coefficients of the function the quadrant picks:
        `k = q` for sine, `q + 1` for cosine, odd `k` uses `C`, `k >= 2` negates), rounded once. Sine is
        odd, cosine even; `sin(+-0) = +-0`, `cos(+-0) = 1`, infinities and NaN give NaN. Degree 6 for f32,
        4 for bf16 and f16.
      - `power`: `x^y = exp(y ln|x|)`. `ln|x|` comes from the log core with `ib + 3` more fraction bits
        (`|y ln|x|| < 2^ib` below overflow and underflow, so a relative error `rho` of the log moves the
        result by at most `2^ib rho`; the near-1 path keeps the relative accuracy for `x` close to 1), the
        product `z = y ln|x|` is exact, then fixed point into the exp core, rounded once; `|z| >= 2^ib`
        overflows or underflows by its sign. The result is negative when `x < 0` and `y` is an odd
        integer, read from `y`'s bits (its exponent against its fraction bits). Special values as C11
        F.10.4.4: `pow(x, +-0) = 1` and `pow(+1, y) = 1` even for NaN, `pow(-1, +-inf) = 1`, zeros and
        infinities of `x` by the sign of `y` and whether it is an odd integer, `pow(x, +-inf)` by `|x|`
        against 1, a negative finite `x` with a non-integer `y` NaN. FP8 is computed too (a table would
        need 2^16 entries); log and exp tables of 2^8 and 2^6 entries in f32.
      Error bounds (`approx.design`): each design adds, as a relative error, the argument reduction,
      every table and coefficient rounding, the Taylor remainder (`r^(d+1) / (d+1)! * e^r` for exp,
      `r^d / (d+1) / (1 - r)` for `Q`), and every truncated product; an ulp is at least `2^-p` of the
      value, so the bound in ulps is that sum times `2^p`, plus 1/2 for the final rounding. Every design
      is below 0.6 ulp (`coverage.json` records it per primitive).
      Reference and check: NumPy's float64 `exp`, `log`, `log1p`, `tanh` and `1 / sqrt`, whose own error
      is a few float64 ulps; a result is accepted when its distance from that value, in ulps of the
      value, is at most the limit plus `2^-48` of the value (sixteen float64 ulps, below `2^-24` ulp in
      every format here). An exactly zero value needs that zero with its sign; an infinite value needs
      the rounded infinity; beyond the largest finite value, overflow is accepted from `max + ulp/2` on.
      Vectors: as in §4.4, plus every input within 3 ulps of each boundary of the design (overflow,
      underflow, the tiny and saturated paths, the table intervals near 1), and half of the random inputs
      drawn where the op does real work (exponential: `2^-(p+2) <= |x| <= ln(max) + 1`; log and rsqrt: the
      whole range and near 1; log_plus_one: near 0, up to 2^40 and near -1; tanh: up to the saturation
      point; sine and cosine: near multiples of pi/2 and over the whole range, plus the worst case of the
      reduction). `power` gets every pair of a list of special operands, `x` near 1 with large `|y|`,
      negative `x` with integer `y`, pairs whose result is near the overflow and underflow thresholds,
      and pairs over both ranges.

   Worked example (`add@f8e4m3fn`, 1.0 + 0.0625, OFP8 E4M3): `1.0 = 0.0111.000`, `0.0625 = 0.0011.000`;
   the exponent difference is 4, so the smaller significand `1.000` shifted right by 4 with three guard
   bits is `0.000|100` (guard 1, sticky 0); the sum `1.000|100` is a tie and rounds to the even `1.000`:
   the result is 1.0. With `0.078125 = 0.0011.010` the shifted bits are `0.000|101` (sticky 1) and the
   sum rounds up to `1.001` = 1.125.

5. **Config keys** (`[passes.softfloat]`): `float_formats` (as `characterize`'s), `int_widths` (the
   integer types conversions reach), `ops` (empty: every float op), `exhaustive_bits` (16),
   `random_vectors` (100000), `batch` (8 primitives per container), `proof_timeout_s` (600, Bitwuzla per
   proof), `recheck_timeout_s` (7200, Z3 per proof in `verify`; `divide@f16` takes 48 min),
   `recheck_sample` (2000 vectors per primitive whose reference `verify` recomputes), `docker_python`,
   `views`. The approximate ops have no keys of their own: their parameters come from `approx.design`
   and enter the cache key through the library hash. Resources:
   `[passes.softfloat.resources]` (threads 16, memory 64 GB, 180 min: up to 16 containers at 4 GB).
   Subnormals are IEEE everywhere: the datapath semantics have no flush mode.

6. **Verification** (`verify.rs`, `verify.json`; reads only `out/softfloat/` and the configuration):
   - `primitives`: the catalog's float primitives, regenerated, are exactly the entries of
     `coverage.json` (an approximate op the library did not build would be listed apart).
   - `files`: every primitive has its eight files, and an exactly specified op its two proof files.
   - `builds`: each `result.json` has no reference mismatch and an IR equal to `llvm` on every vector.
   - `integer_only`: no `kernel.mlir` names a float type.
   - `rtl`: each `rtl.v` defines the primitive's module.
   - `exhaustive`: an exhaustive entry has one vector per input combination.
   - `reference`: the reference recomputed in the image on `recheck_sample` vectors per primitive
     equals the recorded reference, `llvm` and IR results.
   - `accuracy`: every approximate op's largest measured error and its design's bound are within its
     limit (1/2 ulp correctly rounded, 1 ulp faithful) and agree with its `result.json`; exact ops are
     `exact` with neither.
   - `queries`: each `proof.smt2` equals the query regenerated from `fu.opt.ir`, byte for byte.
   - `verdicts`: each `proof.result` matches its query's hash and the evidence (`proven` iff `unsat`).
   - `recheck`: every proven query is `unsat` in the other solver (Z3) within `recheck_timeout_s`.
   - `summary`: the counts and the library hash agree with `coverage.json` and `python/cc_softfloat`.
   Negative tests (deep): an edited `fu.opt.ir` or proof result, a changed reference value, a float type
   in a module, an approximate op's error beyond its limit, a missing primitive each fail `verify`.

7. **Resource behaviour.** Each container gets one CPU and `memory_gb / threads`; `threads` containers
   run at a time. The vectors are at most 2^16 per primitive by default (about 1 MB compressed). Time is
   the proofs: most finish in seconds, the f32 adder in about two minutes and the f32 multiply query in
   seconds (Bitwuzla) to about twelve minutes (Z3 in `verify`); a query past the limit costs the limit.
   The `divide@f16` query takes Bitwuzla about 8 minutes and Z3 48 minutes, which sets
   `recheck_timeout_s`. The hardest Z3 query (`remainder@f8e4m3fn`) peaks at 1.1 GB. Answers are cached
   in `build/.softfloat/cache/` by query hash, so a rerun repeats only the queries that changed or
   timed out. The full pass (218 primitives, 13.7 M vectors, proofs from the cache) took 10.6 minutes
   on 2026-11-04 (iacoma227). The approximate ops build in 1 to 3 seconds each; their IR has 2 to 4
   nodes for a table, 220 to 550 for the computed ones (sine and cosine 231 to 313, power 449 to 548),
   and 625 for `rsqrt` at f32.
   Every input with 32 bits in all (the f32 unary ops, `power` at bf16 and f16) was checked once with
   `python -m cc_softfloat sweep` (16 one-CPU containers per op, 33 to 91 seconds each for its sixteenth
   of the 2^32 inputs or operand pairs; Allo `llvm` against the reference, nothing stored): no input
   outside its limit, and the largest errors, in ulps, exponential 0.5098, log 0.5056, log_plus_one
   0.5083, tanh 0.5023, rsqrt 0.5000 (correctly rounded), sine 0.5025, cosine 0.5025, power 0.5028 at
   bf16 and 0.5028 at f16.

8. **Known limitations.**
   - The approximate ops' check is against NumPy's float64 functions with an allowance for their error,
     not a correctly rounded reference; the claim is "within the limit plus `2^-48` of the value".
   - f32 approximate ops and 16-bit `power` are checked on edge and random vectors in the pass; every
     input was checked once (§7), not on every run. `power@f32` (2^64 pairs) is never checked
     exhaustively.
   - Sine and cosine evaluate one Taylor polynomial on `[-pi/4, pi/4]` without a table (degree 6 in f32);
     a table of `sin` and `cos` at breakpoints would shorten it at the cost of more multipliers.
   - `tanh` in f32 runs the exp core with 44 fraction bits for the cancellation of `1 - exp(-2|x|)` at
     small `|x|`; a separate series path there would make it smaller.
   - The polynomials are Taylor, with closed-form remainders; minimax polynomials (Remez) would need a
     degree less in some designs.
   - A primitive whose proof runs out of time is checked by simulation only (`exhaustive` or `tested`).
   - The multiply proofs' spec is IEEE's definition (exact product, one rounding) rather than the
     solver's `fp.mul` term; the two are equal by the SMT-LIB definition, not by a separate query.
   - `remainder` reduces one binade per step: its circuit grows with the exponent range (277 steps in
     f32).
   - Allo is used from its pinned commit as is; the constraints of §4.3 are kept by the generator, not
     fixed upstream.
