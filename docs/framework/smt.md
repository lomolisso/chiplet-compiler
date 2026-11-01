# `cc-smt`: SMT-LIB scripts and the solver runner

The passes state proof obligations as SMT-LIB 2 files (Barrett, Fontaine, Tinelli, *The SMT-LIB Standard:
Version 2.6*) and run an external solver on them out of process, through a capped tool runner
(docs/framework/core.md). The file is the certificate: any solver can re-check it. Used by `normalize`
(rule proofs, docs/passes/normalize.md); `datapaths` will extend it with scalar op semantics.

## Scripts

`Script` collects comments, declarations (`constant`, `fun`, `define`) and assertions and renders one
query: `(set-logic ALL)`, `(set-option :produce-models true)`, the declarations, the assertions,
`(check-sat)`, `(get-model)`. Logic `ALL` is the one both solvers accept for any mix of uninterpreted
functions, bit-vectors, floating point and arrays (checked with Bitwuzla 0.9.1 and Z3 5.3.3 in the
toolchain image). Helpers build terms: `bv(w)`, `bv_lit(v, w)`, `and`, `or`, `distinct`, `app`.

Bitwuzla supports only the IEEE interchange float formats (Float16, Float32, Float64, Float128) without
its experimental build option, and no equalities over uninterpreted sorts; queries that both solvers
must answer stay within bit-vectors, those formats and uninterpreted functions over bit-vectors.

## Running a batch

`run_batch(runner, location, solver, dir, files, timeout_s)` runs `solver` on every file in `dir` in one
process (`sh -c`: a container start per query would dominate), each under `timeout <timeout_s>`, then
`<solver> --version`. `location` is `Image` (the solvers on the toolchain image's `PATH`, `dir` mounted
read-only at `/work/smt`, with a Docker runner) or `Host` (binary paths, with a host runner). The result
is a `Batch`: the solver, the first line of its version, and a `Verdict` per file:

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
needs glibc 2.39, so on the RHEL 8 build host the solvers run in the image.
