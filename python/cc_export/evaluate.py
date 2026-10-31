"""Evaluate an exported phase at tiny sizes (spec §5.3: symbolic dims instantiated small, generated data).

`Exported.call` refines the symbolic shapes to the concrete argument shapes before compiling. XLA's shape
refinement (jaxlib 0.12) specializes `func.call` callees but not `stablehlo.composite` decompositions, so a
module with composites and symbolic shapes fails with "Error refining shapes". By the StableHLO spec a
composite "can be replaced with its decomposition without changing program semantics", so for evaluation
every composite is rewritten into a `func.call` of its decomposition (what StableHLO's
`stablehlo-legalize-composite-to-call` does; that pass is not registered in jaxlib's bundled bindings).
"""

from __future__ import annotations

import dataclasses

from jax import export
from jax._src.export import _export
from jax._src.interpreters import mlir as jmlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func


def composites_to_calls(module: ir.Module) -> int:
    ops = []

    def collect(op):
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    if o.operation.name == "stablehlo.composite":
                        ops.append(o.operation)
                    collect(o.operation)

    collect(module.operation)
    for op in ops:
        callee = ir.FlatSymbolRefAttr(op.attributes["decomposition"])
        with ir.InsertionPoint(op), op.location:
            call = func.CallOp([r.type for r in op.results], callee, list(op.operands))
        for old, new in zip(op.results, call.results):
            old.replace_all_uses_with(new)
        op.erase()
    return len(ops)


def evaluable(exp: export.Exported) -> export.Exported:
    """`exp` with its composites inlined as calls; `.call(*args)` then works for concrete tiny shapes."""
    with jmlir.make_ir_context():
        m = ir.Module.parse(exp.mlir_module())
        composites_to_calls(m)
        serialized = _export._module_to_bytecode(m)
    return dataclasses.replace(exp, mlir_module_serialized=serialized)


def load(path) -> export.Exported:
    with open(path, "rb") as f:
        return evaluable(export.deserialize(bytearray(f.read())))
