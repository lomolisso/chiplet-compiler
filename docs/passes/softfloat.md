# `softfloat`

1. **Purpose.** The float primitives of the functional units as integer-only hardware: every float op
   of the primitive catalog (design §4.9) is an Allo function over bit patterns (`UInt(n)`), so Allo's
   `llvm` simulator and its XLS backend, which reject float types, accept every unit (design §4.12, §6.4).
   Each primitive is simulated against a reference, taken through Allo's XLS backend, xlscc, `opt_main`
   and `codegen_main` to Verilog, evaluated on its optimized XLS IR, and proven equal to the op's
   semantics in `cc_smt::ops` (the semantics every datapath proof uses) where the solver finishes.
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

3. **Inputs and outputs.**
   - Inputs (not artifacts): the library `python/cc_softfloat` and the toolchain image (Allo, XLS); their
     hashes are in the cache key (`fingerprints`). The primitive list comes from `characterize`'s catalog
     (`catalog::catalog` on `float_formats`, `int_widths`, `ops`).
   - `out/softfloat/ops/<id>/` per primitive: `kernel.py` (the generated Allo function), `kernel.mlir`
     (Allo's module after inlining), `xls/test_block.cpp` (Allo's XLS C++), `fu.ir` (xlscc), `fu.opt.ir`
     (`opt_main`), `rtl.v` and `rtl.sig.textproto` (`codegen_main --generator=combinational`, module
     `<id>`), `vectors.npz` (operands, reference, `llvm` and IR results), `result.json` (schema
     `softfloat_result`), `proof.smt2` and `proof.result` (schema `datapaths_result`).
   - `out/softfloat/coverage.json` (schema `softfloat_coverage`): per primitive its key, id, op, types,
     evidence (`proven`: the IR equals the semantics for every input; `exhaustive`: every input evaluated
     on the IR and on `llvm` matched the reference; `tested`: edge cases and random inputs matched), vector
     count, IR size and proof verdict; and the keys of the approximate ops (not built here).
   - `out/softfloat/summary.json` (schema `softfloat_summary`): counts, vectors, the library hash (the
     key `allo_emit` caches on), the image and the solver; `views/softfloat.md`.

4. **Algorithm.**
   1. *Primitives.* The catalog's float op primitives (every element-wise op that `cc_smt::ops` defines
      on a float type, conversions to and from every integer width and bool). Ops whose result the spec
      leaves to an approximate function (`exponential`, `log`, `tanh`, `rsqrt`, `sine`, ...; uninterpreted
      in the datapath proofs) are listed apart: their accuracy is this library's choice and they are
      built separately.
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
      The C++ includes `xls_builtin.h`, which nothing ships: an empty header in
      `python/cc_softfloat/include`.
   4. *Build* (`python -m cc_softfloat build`, `batch` primitives per container): vectors (every input
      combination when the operands have at most `exhaustive_bits` bits, otherwise the cross product of
      each type's edge cases plus `random_vectors` seeded random operand sets, half of them for binary
      float ops with nearby exponents); the reference on them; Allo `llvm` on them; Allo's XLS C++,
      xlscc, `opt_main`, `codegen_main`; `eval_ir_main` (LLVM JIT) on every vector. A primitive passes
      when `llvm` equals the reference on every vector (NaN as a class) and the IR equals `llvm` bit for
      bit.
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

   Worked example (`add@f8e4m3fn`, 1.0 + 0.0625, OFP8 E4M3): `1.0 = 0.0111.000`, `0.0625 = 0.0011.000`;
   the exponent difference is 4, so the smaller significand `1.000` shifted right by 4 with three guard
   bits is `0.000|100` (guard 1, sticky 0); the sum `1.000|100` is a tie and rounds to the even `1.000`:
   the result is 1.0. With `0.078125 = 0.0011.010` the shifted bits are `0.000|101` (sticky 1) and the
   sum rounds up to `1.001` = 1.125.

5. **Config keys** (`[passes.softfloat]`): `float_formats` (as `characterize`'s), `int_widths` (the
   integer types conversions reach), `ops` (empty: every float op), `exhaustive_bits` (16),
   `random_vectors` (100000), `batch` (8 primitives per container), `proof_timeout_s` (600, Bitwuzla per
   proof), `recheck_timeout_s` (1800, Z3 per proof in `verify`), `recheck_sample` (2000 vectors per
   primitive whose reference `verify` recomputes), `docker_python`, `views`. Resources:
   `[passes.softfloat.resources]` (threads 8, memory 4 GB, 120 min). Subnormals are IEEE everywhere: the
   datapath semantics have no flush mode.

6. **Verification** (`verify.rs`, `verify.json`; reads only `out/softfloat/` and the configuration):
   - `primitives`: the catalog's float primitives, regenerated, are exactly the entries of
     `coverage.json`, with the approximate ops listed apart.
   - `files`: every primitive has its ten files.
   - `builds`: each `result.json` has no reference mismatch and an IR equal to `llvm` on every vector.
   - `integer_only`: no `kernel.mlir` names a float type.
   - `rtl`: each `rtl.v` defines the primitive's module.
   - `exhaustive`: an exhaustive entry has one vector per input combination.
   - `reference`: the reference recomputed in the image on `recheck_sample` vectors per primitive
     equals the recorded reference, `llvm` and IR results.
   - `queries`: each `proof.smt2` equals the query regenerated from `fu.opt.ir`, byte for byte.
   - `verdicts`: each `proof.result` matches its query's hash and the evidence (`proven` iff `unsat`).
   - `recheck`: every proven query is `unsat` in the other solver (Z3) within `recheck_timeout_s`.
   - `summary`: the counts and the library hash agree with `coverage.json` and `python/cc_softfloat`.
   Negative tests (smoke): an edited `fu.opt.ir` or proof result, a changed reference value, a float type
   in a module, a missing primitive each fail `verify`.

7. **Resource behaviour.** Each container gets one CPU and `memory_gb / threads`; `threads` containers
   run at a time. The vectors are at most 2^16 per primitive by default (about 1 MB compressed). Time is
   the proofs: most finish in seconds, the f32 adder in about two minutes and the f32 multiply query in
   seconds (Bitwuzla) to about twelve minutes (Z3 in `verify`); a query past the limit costs the limit.

8. **Known limitations.**
   - The approximate ops (exp, log, log1p, tanh, rsqrt, sin, cos, power) are not built yet: they are
     listed in `coverage.json` and their `characterize` rows stay pending.
   - A primitive whose proof runs out of time is checked by simulation only (`exhaustive` or `tested`).
   - The multiply proofs' spec is IEEE's definition (exact product, one rounding) rather than the
     solver's `fp.mul` term; the two are equal by the SMT-LIB definition, not by a separate query.
   - `remainder` reduces one binade per step: its circuit grows with the exponent range (277 steps in
     f32).
   - Allo is used from its pinned commit as is; the constraints of §4.3 are kept by the generator, not
     fixed upstream.
