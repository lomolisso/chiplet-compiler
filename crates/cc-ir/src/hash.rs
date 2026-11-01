//! An order-independent structural hash of a module (Merkle style, as in hash-consing).
//!
//! A value's hash covers how it is computed: an op's name, call target, attributes, result types, the
//! hashes of its operands in port order, its regions and its callee. A region's hash covers the multiset
//! of its ops' hashes, so neither the order of independent ops nor value numbering nor provenance enters
//! the hash.

use std::collections::HashMap;

use crate::ir::{Def, FuncId, Module, Owner, RegionId, ValueId};

type H = [u8; 32];

fn h(parts: &[&[u8]]) -> H {
    let mut x = blake3::Hasher::new();
    for p in parts {
        x.update(&(p.len() as u64).to_le_bytes());
        x.update(p);
    }
    *x.finalize().as_bytes()
}

fn bytes<T: serde::Serialize>(v: &T) -> Vec<u8> {
    postcard::to_stdvec(v).unwrap_or_default()
}

struct Hasher<'a> {
    m: &'a Module,
    values: Vec<Option<H>>,
    funcs: HashMap<FuncId, H>,
}

impl Hasher<'_> {
    fn value(&self, v: ValueId) -> H {
        self.values[v.index()].unwrap_or([0; 32])
    }

    fn region(&mut self, r: RegionId) -> H {
        let reg = self.m.region(r);
        let owner_tag: &[u8] = match reg.owner {
            Owner::Func(_) => b"func",
            Owner::Op(_) => b"op",
        };
        for (i, &a) in reg.args.iter().enumerate() {
            let val = self.m.value(a);
            self.values[a.index()] = Some(h(&[
                b"arg",
                owner_tag,
                &(i as u64).to_le_bytes(),
                &bytes(&val.ty),
                &bytes(&val.dim_var),
            ]));
        }
        let mut ops = Vec::with_capacity(reg.ops.len());
        for &o in &reg.ops {
            let op = self.m.op(o);
            let regions: Vec<H> = op.regions.clone().into_iter().map(|rr| self.region(rr)).collect();
            let callee = op.callee.map(|f| self.func(f)).unwrap_or([0; 32]);
            let operands: Vec<u8> = op.operands.iter().flat_map(|&v| self.value(v)).collect();
            let rtypes: Vec<_> = op.results.iter().map(|&v| &self.m.value(v).ty).collect();
            let regions: Vec<u8> = regions.concat();
            let oh = h(&[
                op.name.as_bytes(),
                &bytes(&op.target),
                &bytes(&op.attrs),
                &bytes(&rtypes),
                &operands,
                &regions,
                &callee,
            ]);
            for (i, &res) in op.results.iter().enumerate() {
                if let Def::Result { .. } = self.m.value(res).def {
                    self.values[res.index()] = Some(h(&[&oh, &(i as u64).to_le_bytes()]));
                }
            }
            ops.push(oh);
        }
        ops.sort_unstable();
        let args: Vec<_> = reg.args.iter().map(|&a| &self.m.value(a).ty).collect();
        h(&[b"region", &bytes(&args), &ops.concat()])
    }

    fn func(&mut self, f: FuncId) -> H {
        if let Some(x) = self.funcs.get(&f) {
            return *x;
        }
        let func = self.m.func(f);
        let body = self.region(func.body);
        let x = h(&[b"func", &bytes(&func.kind), &bytes(&func.results), &body]);
        self.funcs.insert(f, x);
        x
    }
}

/// The module's structural hash (hex).
pub fn module_hash(m: &Module) -> String {
    let mut s = Hasher {
        m,
        values: vec![None; m.values.len()],
        funcs: HashMap::new(),
    };
    let entry = s.func(m.entry);
    let cons: Vec<&str> = m.constraints.iter().map(|c| c.text.as_str()).collect();
    let x = h(&[b"module", &entry, &bytes(&m.symbols), &bytes(&cons)]);
    x.iter().map(|b| format!("{b:02x}")).collect()
}
