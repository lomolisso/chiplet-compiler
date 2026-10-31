# cc-smt

Skeleton: SMT-LIB term generation, scalar op semantics (QF_BV / QF_FP) and
the solver runner. Solvers run out of process through a `ToolRunner` (crash isolation, memory caps,
replayable `.smt2` certificates; `docs/context/tooling/rust_stack.md` §4).
