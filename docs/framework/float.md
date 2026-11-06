# `cc-float`: glue around float cores, its proofs, compositions

## 1. Purpose and position

Float operators are instances of third-party IP cores (design §4.12). What chiplet-compiler builds
around them is small: wrappers that adapt a core to the numerics of design §6.4 (subnormal rule,
FP8 encodings, NaN and signed-zero cases) and compositions of cores for operators that have no core.
`cc-float` generates the Verilog of both, proves the glue bit-exact against `cc_smt`'s specification
through Yosys, sizes compositions to the error budget, and records the evidence. `characterize` uses
it to build one wrapper module per float primitive around the cores of the flow's providers; it takes
`CoreInstance` and `Contract` values (`cc_core::adapters::contract`) and knows no provider.

| Module | Role |
|---|---|
| `glue.rs` | `Glue` (one wrapper: op and format), `CoreNeed` (the cores it needs: role, format, arity), `Glue::build` (Verilog around given instances), `standard` (the wrapper set per format), `stand_in` (placeholder instances for tests) |
| `rtl.rs` | the Verilog writer and the bit-level glue: DAZ, exact widening, flush-and-narrow, the FP8 encoder, saturation; `Mutation` (deliberate faults for negative controls) |
| `recipe.rs` | the compositions' Verilog (power, tanh, log1p, rsqrt) and their special values |
| `compose.rs` | `Budget` (default: the CUDA table), the recipes' error constants, `size` (extra bits `k` from the budget and the cores' accuracy) |
| `proof.rs` | `yosys_script`, `query` (Yosys' SMT-LIB of a wrapper plus contracts and the specification), `Prover` (Yosys and the solvers in the toolchain image, the shared answer cache) |
| `python/cc_float/` | the composition-error sampler and the sine/cosine measurement (section 6) |

## 2. Background

- **Numerics** (design §6.4, docs/framework/smt.md "Floats"): f32 and bf16 read subnormal operands as
  zero (DAZ) and flush a result whose value rounded with an unbounded exponent is below `2^emin`,
  keeping its sign (FTZ); f16 and FP8 keep gradual underflow; round to nearest even; an E4M3FN
  arithmetic result beyond 448 is NaN, a conversion into FP8 saturates (OCP OFP8 Rev 1.1, saturating
  mode).
- **Cores are trusted** (design §5): a core instance is an opaque function bound by its documented
  contract: `Accuracy` (correctly rounded, faithful, `n` ulps, an absolute bound, unbounded),
  IEEE special values, `Subnormals` (IEEE or flushing).
- **Rounding with an unbounded exponent**: a correctly rounded core of a format with one more exponent
  bit than the target rounds a product, quotient or root of two target values exactly as an unbounded
  exponent would (the result stays normal there), so flushing below `2^emin` and narrowing the exponent
  field afterwards is design §6.4's rule with no further rounding (numerics rule D2). A sum or difference
  that underflows is exact (J. R. Hauser, "Handling floating-point exceptions in numeric programs", ACM
  TOPLAS 18(2), 1996), so a core of the format's own width rounds sums alike and the glue flushes a
  subnormal result.
- **Double rounding** (S. A. Figueroa del Cid, "When is double rounding innocuous?", ACM SIGNUM 30(3),
  1995): rounding a sum, product, quotient or root of `p`-bit values to `q >= 2p + 2` bits and then to
  `p` equals one rounding; E4M3FN (`p = 4`) on f16 cores (`q = 11`) is correctly rounded.
- **Error of a composition** (N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM 2002, §2.2): cores at precision `q = p + k` with errors `e` (ulps of `q`) give a result within
  `1/2 + C 2^-k` ulps of the format after one final rounding, `C` the recipe's error constant in units of
  `2^-q`; a budget of `T` ulps needs `k >= ceil(log2(C / (T - 1/2)))`.
  - power `exp2(y log2 x)`: `C = ln 2 · tau · (2 e_log + 1) + 2 e_exp`, `tau` the largest `|y log2 x|`
    with a finite result (128 for f32 and bf16, 16 for f16, 9 for E4M3FN); FloPoCo's `FPPow` computes its
    log with the same number of extra bits for f32 (F. de Dinechin, gitlab.com/flopoco/flopoco,
    `FPPow.cpp`);
  - tanh `(e^2x - 1)/(e^2x + 1)` with `x - x^3/3` below `x1 = (7.5 · 2^-(p+2))^(1/4)` and 1 above
    `(p + 2) ln 2 / 2`: `C = (2 e_exp + 2) / (2 x1)` (the cancellation in `e^2x - 1`; without the
    branch no `k` suffices);
  - log1p by D. Goldberg, "What every computer scientist should know about floating-point arithmetic",
    ACM Computing Surveys 23(1), 1991, Theorem 4 (`x ln(w) / (w - 1)`, `w = fl(1 + x)`): `C = 2 e_log + 3`;
  - rsqrt `sqrt(RN(1/x))`: `C = 2 e_sqrt + 1/2`;
  - a core of a wider format (f32 for bf16, f16 and FP8): `k = 24 - p`, `C = e`.
- **Yosys** (C. Wolf, yosyshq.net) reads the wrapper, `cutpoint` replaces each core instance's outputs by
  free values, `write_smt2 -stbv -wires` writes the design as SMT-LIB with the state and the free values
  as one bit-vector and every named wire as a function of it (docs/framework/smt.md, "Verilog to
  SMT-LIB").

## 3. Inputs and outputs

- **In:** a `Glue` (op, formats; for a composition its recipe and inner precision `q`) and one
  `CoreInstance` per `CoreNeed`. A need names the core's role and the float format it computes in: the
  operands and result of an arithmetic or approximate core, the result of a conversion core, the
  operands of a less-than or float-to-integer core. `CoreInstance::exp_width`/`sig_width` must be that
  format; its ports map onto the operands `a0, a1`, the result `z`, constants, or nothing. Core ports
  carry IEEE 754 interchange encodings of their widths.
- **Out:** `Built { rtl, needs, .. }`: the module `glue_<op>_<format>` (ports `a0`, `a1`, `z`), with every
  core's operands and result on kept wires `c<k>_a<j>`, `c<k>_z`, and blackbox stubs of the core modules
  (the proofs read them in place of the cores). `Prover::prove` writes per wrapper `<name>/{glue.v,
  stubs.v, run.ys, glue.smt2, yosys.log}` and `smt/<name>.smt2`, and returns one `ProofRecord` per
  wrapper (module, mutation, expected answer, solver and version, verdict, seconds, cached, query hash).
- **Sampler** (`python -m cc_float.sampler --out <file>`): JSON rows per operator, format and core class:
  recipe, core errors, `k`, `q`, the first-order bound, the budget, inputs and method (exhaustive or
  stratified), the measured maximum in ulps, the worst input, `within_budget`; and per FP8 format the
  number of `+ - * / sqrt` results through f16 cores that are not correctly rounded.

## 4. Algorithm

**Wrappers** (`Glue::build`), per format `F`:

| wrapper | cores | glue |
|---|---|---|
| add, subtract (f32, bf16) | one at `F` | DAZ the operands; flush a subnormal result |
| multiply, divide, sqrt (f32, bf16) | one at `(e + 1, m)` | DAZ, widen exactly; flush below `2^emin` keeping the sign, narrow the exponent (beyond `F`'s range infinity; NaN and infinities kept) |
| arithmetic (f16, E5M2) | one at `F` | none |
| arithmetic (E4M3FN) | one at f16 | decode to f16 (exact; the NaN pattern to f16's NaN), encode back (below) |
| approximate operator | one at `F`, at `(e + 1, m)` (f32, bf16), or at f32 (16-bit, FP8) | DAZ, widen; flush-and-narrow, the encoder (E4M3FN), or a conversion core (f16, bf16, E5M2 from f32) |
| convert, widening | none | DAZ of a flushing source, exact widening (subnormals normalized) |
| convert to E4M3FN | none | the encoder, saturating |
| convert to E5M2 | one conversion core | infinities to the largest finite value of their sign |
| convert to bf16, f16 (narrowing) | one conversion core (result at `(e + 1, m)` for bf16) | DAZ of the source; flush-and-narrow for bf16 |
| maximum, minimum | one less-than core (E4M3FN at f16) | DAZ; NaN if either operand is NaN; among zeros `-0 < +0`; else the core's order |
| compare EQ, NE, LT, LE | one or two less-than cores | DAZ; quiet predicates from `a < b`, `b < a` and the NaN tests |
| abs, negate, sign | none | DAZ; the sign bit; `sign`: NaN and zeros kept, else `±1` |
| convert to integer | one float-to-integer core (rounding toward zero, saturating) | DAZ; NaN gives 0 (the core's NaN result is not used) |

The **FP8 encoder** rounds an IEEE value of a wider format to E4M3FN by `cc_smt::float::from_f32`'s steps
on the source's bits: the significand shifted by the exponent distance (more in the subnormal range),
guard and sticky bits, the increment, the packed magnitude carrying into the exponent; beyond 448 NaN
(arithmetic) or ±448 (conversion); NaN kept. It is the only rounding the glue does (design §4.12).

**Compositions** (`recipe.rs`) compute in cores of the inner format `Q = (e + 2, q - 1)`: two more
exponent bits keep every intermediate of a finite result in range. One conversion core rounds back to
`F` (`(e + 1, m)` and flush-and-narrow for f32 and bf16) or the encoder rounds to E4M3FN. The special
values are glue on `F`'s bits: power follows C11 Annex F.10.4.4 (`y = ±0` and `x = +1` give 1, NaN,
infinite `y` by `|x|` against 1, zeros and infinities of `x` by the sign and parity of `y`, a negative
finite `x` with a non-integer `y` is NaN, with an integer `y` the core path on `|x|` with the sign of
`x^y`); tanh is odd, NaN stays NaN, the clamp gives `±1` also for infinities; log1p gives NaN below `-1`,
`-inf` at `-1`, `+inf` at `+inf`, and keeps zeros; rsqrt gives NaN below zero, `±inf` at `±0`, `+0` at
`+inf` (IEEE 754-2019 §9.2). An FP8 result that would be infinite is NaN.

**Sizing** (`compose::size`): the budget of the op and format, the cores' errors from their `Accuracy`
(`CorrectlyRounded` 1/2, `Faithful` 1, `Ulp { ulp }`), the recipe's constant, `k`. A core with an
absolute or no error bound is refused: the recipes need a relatively accurate core (a logarithm that is
only absolutely accurate near 1 makes power need `k >= p + log2 tau + 1`, and log1p no `k` at all).

## 5. Configuration

`cc-float` reads no section itself. Its inputs are the core instances and the budget: `Budget::cuda()`
is the default (f32: exp 2, exp2 2, log 1, log2 1, log1p 1, sin 2, cos 2, tanh 2, rsqrt 2, pow 4 ulp;
16-bit and 8-bit formats 1 ulp), and a caller passes the configured one. The prover takes the solver of
`[tools.smt] solver`, its `timeout_s`, and the answer cache `paths.smt_cache`.

## 6. Verification

**Glue proofs** (one per wrapper and format). `query` takes Yosys' SMT-LIB of the wrapper, every core
instance cut, and adds:

- each core's contract on its wires: a correctly rounded core's result is the theory's operation at the
  core's own widths (gradual underflow) on its operands; a conversion core's result is its operand
  rounded to nearest even; a less-than core's bit is `fp.lt`; a float-to-integer core's result is
  `cc_smt::ops::convert` unless the operand is NaN; an approximate core is free;
- the negated claim: the output differs from `cc_smt::ops::apply` (or `convert`) for the format; for an
  approximate operator, from the core's result rounded to the format by `cc_smt`'s rounding (an
  arithmetic result: flush, NaN on E4M3FN overflow); or the core's operands are not the format's DAZ
  operands exactly widened.

The f32 multiply, divide and square root leave the arithmetic out: core and specification are each the
rounding of one Float64 value `v`, the exact result, constrained to the binade range of the operands
(or to the IEEE special result), with `cc_smt::float::arith_exact` as the specification's encoding. A
query relating a 24-bit multiplier of one sort to one of another does not finish in Z3 (600 s
measured), these take seconds. The narrower formats keep both operations: their specification rounds
twice (to Float32, then to the format), which equals one rounding for actual products but not for every
`v`.

A composition's proof binds every core but the final conversion to a free function of its operands'
encodings (`core<k>`), so it covers the structure, the wiring, the special values and the final rounding
for any cores; the specification (`compose_spec`) states the recipe and C11 Annex F with the theory's
predicates (integer and odd tests by truncation, the half of `|y| >= 2` on its exponent field),
independently of the glue's bit tests.

Each query goes to the configured solver unless it names a float sort other than Float16, Float32,
Float64 and Float128 (as a sort or as `to_fp` indices; Bitwuzla's release parses the latter without its
sort check), then to the first solver declaring sorts of any width (Z3). Answers go through the shared
cache. Negative controls, each `sat`: an f32 multiply and a bf16 add without the flush, an encoder
rounding toward zero, an E4M3FN multiply without the overflow case, maximum without NaN propagation,
minimum without the signed-zero order, float to integer without NaN to 0, and the compositions without
their NaN case.

Measured (deep test, iacoma227, Z3 5.3.3 and Bitwuzla 0.9.1 in image 0.5.0, an empty answer cache,
2026-11-05): 80 wrappers of the standard set over f32, bf16, f16 and E4M3FN, three conversions into
E5M2 and 16 compositions (four recipes, four formats), every one `unsat`; the 11 negative controls
`sat`. Median 0.6 s per query including a container start; the slowest wrappers are bf16 divide (Z3,
33 s) and multiply (11 s), the slowest compositions E4M3FN power (106 s) and tanh (46 s); 345 s of
solver time in total, 130 s wall on 8 parallel calls.

**Compositions** are measured, not proven: `python -m cc_float.sampler` models each core at its contract
(a correctly rounded core rounds the float64 reference; an `e`-ulp core is moved `e - 1/2` ulps to one
side first, both sides and every combination for two cores) at precision `q` with an unbounded exponent,
over every value of the 16-bit and 8-bit formats (for power every positive `x` with every non-zero `y`
whose exact result lies in the format's range: a negative `x` is `|x|` and the sign glue, the rest
special values), and over every f32 binade with
seeded random significands plus each recipe's hard cases (pairs for power: log-uniform, `x` near 1 with
`|y log2 x|` near `tau`, integer `y`). Errors are taken before the flush of a flushing format (the flush
is glue). At the `k` the formula gives for correctly rounded, faithful and 2-ulp cores, every composition
is within the default budget (sampler run of 2026-11-05, iacoma227, 4 minutes on 12 processes):

| op | format | budget | k (CR / faithful / 2 ulp) | max error, ulps (CR / faithful / 2 ulp) | inputs |
|---|---|---|---|---|---|
| power | f32 | 4 | 6 / 7 / 7 | 2.111 / 1.871 / 3.221 | stratified, 514,796 |
| tanh | f32 | 2 | 6 / 7 / 7 | 0.732 / 0.740 / 0.981 | stratified, 139,258 |
| log_plus_one | f32 | 1 | 3 / 4 / 4 | 0.712 / 0.635 / 0.777 | stratified, 98,189 |
| rsqrt | f32 | 2 | 0 / 1 / 2 | 0.843 / 1.160 / 1.083 | stratified, 65,532 |
| power | bf16 | 1 | 9 / 10 / 10 | 0.719 / 0.681 / 0.842 | exhaustive, 1,060,599,911 |
| tanh | bf16 | 1 | 4 / 4 / 5 | 0.612 / 0.612 / 0.612 | exhaustive, 65,024 |
| log_plus_one | bf16 | 1 | 3 / 4 / 4 | 0.684 / 0.625 / 0.778 | exhaustive, 48,640 |
| rsqrt | bf16 | 1 | 2 / 3 / 4 | 0.614 / 0.637 / 0.621 | exhaustive, 32,512 |
| power | f16 | 1 | 6 / 7 / 7 | 0.735 / 0.696 / 0.863 | exhaustive, 1,089,723,731 |
| tanh | f16 | 1 | 5 / 5 / 6 | 0.642 / 0.642 / 0.642 | exhaustive, 63,486 |
| log_plus_one | f16 | 1 | 3 / 4 / 4 | 0.714 / 0.629 / 0.780 | exhaustive, 47,102 |
| rsqrt | f16 | 1 | 2 / 3 / 4 | 0.704 / 0.644 / 0.632 | exhaustive, 31,743 |
| power | f8e4m3fn | 1 | 5 / 6 / 7 | 0.655 / 0.637 / 0.672 | exhaustive, 17,005 |
| tanh | f8e4m3fn | 1 | 3 / 3 / 4 | 0.533 / 0.533 / 0.482 | exhaustive, 252 |
| log_plus_one | f8e4m3fn | 1 | 3 / 4 / 4 | 0.600 / 0.577 / 0.776 | exhaustive, 181 |
| rsqrt | f8e4m3fn | 1 | 2 / 3 / 4 | 0.552 / 0.552 / 0.552 | exhaustive, 126 |

Through one f32 core (`k = 24 - p`) exponential, log, log1p, tanh, rsqrt, sine and cosine stay within
0.5000 ulp (bf16), 0.5002 ulp (f16) and 0.4998 ulp (E4M3FN) for every core class; E4M3FN `+ - * / sqrt`
through f16 cores are correctly rounded on all 64,516 pairs.

**Sine and cosine** (`python -m cc_float.sincos`): XLS's `fp_sincos_32` IR (`ir_converter_main`,
`opt_main` from `third_party/xls_go_math/fp_sincos_32.x` in the image) evaluated by `eval_ir_main` on
every bf16 and f16 value widened to f32 (119,808 inputs, 3 s), rounded once to the format: bf16 sine and
cosine at most 0.500 ulp, every result correctly rounded; f16 sine 0.5054 ulp, cosine 0.5003 ulp, 8
results each not correctly rounded; below `2^-4` at most 0.4999 ulp (f16) and 0.165 ulp (bf16 sine). The
0.51-ulp statement of design §6.4 holds for every input, near zero included.

## 7. Resource behaviour

Wrappers are a few hundred lines of Verilog. A proof is one Yosys call (all wrappers in one container)
and one solver call per wrapper, in parallel within the prover's caps; the deep set takes about six
minutes of solver time when nothing is cached (two minutes wall on 8 calls), and none once the shared
cache holds the answers. The sampler is numpy on float64 carriers: power over every pair
of a 16-bit format is about 2 · 10^9 evaluations per core mode, in parallel processes (`--jobs`); the
whole run takes a few minutes on 12 processes.

## 8. Known limitations

- Core ports are taken as IEEE interchange encodings of the instance's widths; a core with another
  encoding (HardFloat's recoded format) comes with its conversions inside the instance.
- The compositions' accuracy is sampled, not proven: first-order bounds, cores modelled at their contract
  (a real core's errors need not be one-sided or uniform), float64 references (2^-29 ulp of f32), f32
  inputs sampled. A machine-checked bound (Gappa) would make them certificates.
- The composition proofs leave the cores' IEEE special behaviour to the cores (`Contract::ieee_specials`):
  a zero through tanh's cubic branch is zero because the cores compute zeros exactly.
- FP8 arithmetic through f16 cores is correctly rounded by Figueroa's condition, checked exhaustively by
  the sampler and proven per wrapper against `cc_smt`'s specification; FP8 transcendentals use f32 cores.
- E5M2 arithmetic runs on cores at its own width (IEEE); only its conversions are in the standard set.
