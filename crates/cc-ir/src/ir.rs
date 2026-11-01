//! The IR: an arena of functions, regions, ops and values addressed by typed `u32` indices.
//!
//! Every region has one block (StableHLO bodies are single-block). An op's operand port is its position
//! in `operands`. `func.call` never occurs (calls are inlined); a `stablehlo.composite` refers to its
//! decomposition through `callee`, which is a `Func` of kind `Decomposition`.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::attrs::Attrs;
use crate::sym::Constraint;
use crate::types::Type;

macro_rules! id {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
        )]
        pub struct $name(pub u32);
        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}
id!(FuncId);
id!(RegionId);
id!(OpId);
id!(ValueId);

pub const IR_SCHEMA: &str = "cc.ir.module/1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Module {
    pub schema: String,
    pub model: String,
    pub phase: String,
    /// `models.config` of the export (`full` / `reduced`).
    pub config: String,
    /// The size symbols of the phase, sorted.
    pub symbols: Vec<String>,
    /// The constraints of the symbolic scope.
    pub constraints: Vec<Constraint>,
    pub entry: FuncId,
    pub funcs: Vec<Func>,
    pub regions: Vec<Region>,
    pub ops: Vec<Op>,
    pub values: Vec<Value>,
    /// Interned source locations (`Provenance::loc` indexes this).
    pub locs: Vec<String>,
}

/// Why a function survives inlining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FuncKind {
    /// The phase entry (`main`).
    Entry,
    /// The decomposition of a component composite (`cc.*`).
    Decomposition,
    /// A computation referenced by a `custom_call` (`called_computations`, e.g. a reducer).
    Computation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Func {
    pub name: String,
    pub kind: FuncKind,
    pub body: RegionId,
    pub results: Vec<Type>,
}

/// What owns a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    Func(FuncId),
    Op(OpId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Region {
    pub owner: Owner,
    pub args: Vec<ValueId>,
    /// In program order; the last op is the terminator (`func.return` / `stablehlo.return`).
    pub ops: Vec<OpId>,
}

/// Where an op came from in the exported module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Provenance {
    /// Function of `graph.json` holding the source op.
    pub func: String,
    /// Op id in `graph.json`.
    pub op: u64,
    /// Index into `Module::locs`.
    pub loc: u32,
    /// The inlined call sites, outermost first.
    pub inlined_from: Vec<CallSite>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CallSite {
    /// The inlined op: `func.call`, or a composite that is not a component boundary.
    pub op: u64,
    pub callee: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Op {
    /// `stablehlo.add`, `func.return`, …
    pub name: String,
    /// `call_target_name` of a `stablehlo.custom_call`.
    pub target: Option<String>,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub attrs: Attrs,
    pub regions: Vec<RegionId>,
    /// The decomposition of a composite, or the computation of a custom call.
    pub callee: Option<FuncId>,
    /// The region holding this op.
    pub parent: RegionId,
    /// Size computation (design §4.2): the op computes dimension values that reach a shape operand or a
    /// shape assertion.
    pub size: bool,
    /// Operand ports where a dimension value enters compute (a size value at a non-shape port of a
    /// non-size op), ascending.
    pub dim_as_data: Vec<u32>,
    pub prov: Provenance,
}

/// How a value is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Def {
    Arg { region: RegionId, index: u32 },
    Result { op: OpId, index: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Value {
    pub ty: Type,
    pub def: Def,
    /// A dimension-variable argument (`jax.global_constant = "<symbol>"`).
    pub dim_var: Option<String>,
    /// The value id in `graph.json` it was created from.
    pub src: u64,
}

impl Module {
    pub fn op(&self, id: OpId) -> &Op {
        &self.ops[id.index()]
    }
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.index()]
    }
    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id.index()]
    }
    pub fn func(&self, id: FuncId) -> &Func {
        &self.funcs[id.index()]
    }

    /// The function a region belongs to (walking up through op regions).
    pub fn region_func(&self, mut r: RegionId) -> FuncId {
        loop {
            match self.region(r).owner {
                Owner::Func(f) => return f,
                Owner::Op(o) => r = self.op(o).parent,
            }
        }
    }

    /// Users of every value: `(op, port)` lists indexed by value.
    pub fn users(&self) -> Vec<Vec<(OpId, u32)>> {
        let mut u = vec![Vec::new(); self.values.len()];
        for (i, op) in self.ops.iter().enumerate() {
            for (p, v) in op.operands.iter().enumerate() {
                u[v.index()].push((OpId(i as u32), p as u32));
            }
        }
        u
    }

    /// Op-name multiset of the composite expansion from the entry: every op of the entry, and of each
    /// composite's decomposition at each composite (nested regions included), as `func.func`-free names.
    pub fn expanded_multiset(&self) -> BTreeMap<String, u64> {
        let mut out = BTreeMap::new();
        let mut stack = vec![self.func(self.entry).body];
        while let Some(r) = stack.pop() {
            for &o in &self.region(r).ops {
                let op = self.op(o);
                *out.entry(op.name.clone()).or_default() += 1;
                stack.extend(op.regions.iter().copied());
                if op.name == "stablehlo.composite"
                    && let Some(f) = op.callee
                {
                    stack.push(self.func(f).body);
                }
            }
        }
        out
    }
}
