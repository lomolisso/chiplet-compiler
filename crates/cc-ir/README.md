# cc-ir

Skeleton: IR types (`Module`, `Func`, `Op`, `Value`, `TensorType`, symbolic
`Dim = Const | Sym(expr)`) and the op-semantics table (StableHLO semantics only: arity, elementwise or
not, commutativity, iteration semantics), built from the measured op inventory (spec §4.2).
