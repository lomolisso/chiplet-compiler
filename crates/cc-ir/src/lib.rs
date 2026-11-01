//! The chiplet-compiler IR (`docs/framework/ir.md`): typed modules with symbolic sizes, and the
//! op-semantics table that encodes the StableHLO specification (design §4.2).

pub mod absgraph;
pub mod attrs;
pub mod hash;
pub mod ir;
pub mod ops;
pub mod sym;
pub mod types;

pub use attrs::{Attr, Attrs, AttrsExt, Scalar};
pub use ir::{
    CallSite, Def, Func, FuncId, FuncKind, Module, Op, OpId, Owner, Provenance, Region, RegionId, Value,
    ValueId,
};
pub use ops::{OpClass, OpSpec, lookup};
pub use sym::{Constraint, SymCtx, SymExpr};
pub use types::{Dim, Elem, TensorType, Type};
