//! Facts recomputed from `graph.json` alone, independently of the Python exporter: the static op-name
//! multiset, and (following `func.call` and composite decompositions from the entry at every call site)
//! the GEMM/convolution census and the composite coverage.

use std::collections::{BTreeMap, HashMap};

use super::json::{Attr, GraphFile, Module, Op, Region, SigSummary, Type};

/// Ops that only delimit structure; coverage counts every other op (`python/cc_export/analysis.py`).
const STRUCTURAL: [&str; 4] = [
    "func.return",
    "stablehlo.return",
    "func.call",
    "stablehlo.composite",
];

/// Op names of every op in the module (nested regions included), plus one `func.func` per function.
pub fn multiset(m: &Module) -> BTreeMap<String, u64> {
    fn region(r: &Region, out: &mut BTreeMap<String, u64>) {
        for b in &r.blocks {
            for op in &b.ops {
                *out.entry(op.name.clone()).or_default() += 1;
                for rr in &op.regions {
                    region(rr, out);
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    for f in &m.funcs {
        *out.entry("func.func".to_string()).or_default() += 1;
        region(&f.body, &mut out);
    }
    for op in &m.other_ops {
        *out.entry(op.name.clone()).or_default() += 1;
    }
    out
}

/// The callee of a `func.call` or the decomposition of a `stablehlo.composite`.
pub fn callee(op: &Op) -> Option<&str> {
    match op.name.as_str() {
        "func.call" => op.attributes.get("callee").and_then(Attr::as_str),
        "stablehlo.composite" => op.attributes.get("decomposition").and_then(Attr::as_str),
        _ => None,
    }
}

/// What one expansion from the entry yields.
#[derive(Debug, Default)]
pub struct Expansion {
    pub census: BTreeMap<String, SigSummary>,
    pub ops: u64,
    pub in_composite: u64,
    pub in_leaf_composite: u64,
    pub anon: u64,
    pub instances: u64,
    pub leaf_instances: u64,
    /// Call targets that are not functions of the module.
    pub missing_callees: Vec<String>,
}

struct Walker<'a> {
    funcs: HashMap<&'a str, &'a Region>,
    types: HashMap<u64, &'a Type>,
    out: Expansion,
    /// Per instance: has a child instance.
    has_child: Vec<bool>,
    /// Per non-structural op: innermost instance.
    owner: Vec<Option<usize>>,
}

impl<'a> Walker<'a> {
    fn region(&mut self, r: &'a Region, stack: &mut Vec<usize>, depth: usize) {
        for b in &r.blocks {
            for op in &b.ops {
                self.op(op, stack, depth);
                for rr in &op.regions {
                    self.region(rr, stack, depth);
                }
            }
        }
    }

    fn op(&mut self, op: &'a Op, stack: &mut Vec<usize>, depth: usize) {
        if !STRUCTURAL.contains(&op.name.as_str()) {
            self.owner.push(stack.last().copied());
        }
        if let Some((key, flops)) = self.core(op) {
            let e = self.out.census.entry(key).or_insert(SigSummary {
                count: 0,
                flops: Some(0),
            });
            e.count += 1;
            e.flops = match (e.flops, flops) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
        }
        let Some(target) = callee(op) else { return };
        let Some(body) = self.funcs.get(target).copied() else {
            self.out.missing_callees.push(target.to_string());
            return;
        };
        if depth > 10_000 {
            self.out
                .missing_callees
                .push(format!("{target} (call depth over 10000)"));
            return;
        }
        if op.name == "stablehlo.composite" {
            let inst = self.has_child.len();
            if let Some(&parent) = stack.last() {
                self.has_child[parent] = true;
            }
            self.has_child.push(false);
            stack.push(inst);
            self.region(body, stack, depth + 1);
            stack.pop();
        } else {
            self.region(body, stack, depth + 1);
        }
    }

    fn shape(&self, v: u64) -> Option<(&'a [Option<u64>], &'a str)> {
        match self.types.get(&v)? {
            Type::Tensor { shape, elem } => Some((shape, elem)),
            _ => None,
        }
    }

    /// `("<kind> <sig>", flops)` for a dot_general / convolution, as `verify/hlo.py` defines them.
    fn core(&self, op: &Op) -> Option<(String, Option<u64>)> {
        let kind = match op.name.as_str() {
            "stablehlo.dot_general" => "dot_general",
            "stablehlo.convolution" | "stablehlo.dynamic_conv" => "convolution",
            _ => return None,
        };
        let (ls, lt) = self.shape(*op.operands.first()?)?;
        let (rs, rt) = self.shape(*op.operands.get(1)?)?;
        let (os, ot) = match &op.results.first()?.ty {
            Type::Tensor { shape, elem } => (shape.as_slice(), elem.as_str()),
            _ => return None,
        };
        let dn = op.attributes.get(if kind == "dot_general" {
            "dot_dimension_numbers"
        } else {
            "dimension_numbers"
        })?;
        let k = match (kind, dn) {
            (
                "dot_general",
                Attr::DotDimensionNumbers {
                    lhs_contracting_dimensions,
                    ..
                },
            ) => prod(
                lhs_contracting_dimensions
                    .iter()
                    .map(|&i| ls.get(i as usize).copied().flatten()),
            ),
            (
                "convolution",
                Attr::ConvDimensionNumbers {
                    kernel_output_feature_dimension,
                    ..
                },
            ) => prod(
                rs.iter()
                    .enumerate()
                    .filter(|(i, _)| *i as i64 != *kernel_output_feature_dimension)
                    .map(|(_, d)| *d),
            ),
            _ => return None,
        };
        let out_n = prod(os.iter().copied());
        let flops = match (out_n, k) {
            (Some(n), Some(k)) => Some(2 * n * k),
            _ => None,
        };
        Some((format!("{kind} {lt}x{rt}->{ot}"), flops))
    }
}

/// Product of dims; `None` if any dim is dynamic.
fn prod(mut it: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    it.try_fold(1u64, |a, d| d.map(|d| a * d))
}

pub fn expand(g: &GraphFile) -> Expansion {
    let mut types = HashMap::new();
    fn region<'a>(r: &'a Region, t: &mut HashMap<u64, &'a Type>) {
        for b in &r.blocks {
            for a in &b.args {
                t.insert(a.v, &a.ty);
            }
            for op in &b.ops {
                for res in &op.results {
                    t.insert(res.v, &res.ty);
                }
                for rr in &op.regions {
                    region(rr, t);
                }
            }
        }
    }
    for f in &g.module.funcs {
        region(&f.body, &mut types);
    }
    let mut w = Walker {
        funcs: g
            .module
            .funcs
            .iter()
            .map(|f| (f.name.as_str(), &f.body))
            .collect(),
        types,
        out: Expansion::default(),
        has_child: Vec::new(),
        owner: Vec::new(),
    };
    match w.funcs.get(g.entry.as_str()).copied() {
        Some(body) => w.region(body, &mut Vec::new(), 0),
        None => w.out.missing_callees.push(g.entry.clone()),
    }
    let mut out = w.out;
    out.ops = w.owner.len() as u64;
    out.anon = w.owner.iter().filter(|o| o.is_none()).count() as u64;
    out.in_composite = out.ops - out.anon;
    out.in_leaf_composite = w
        .owner
        .iter()
        .filter(|o| o.is_some_and(|i| !w.has_child[i]))
        .count() as u64;
    out.instances = w.has_child.len() as u64;
    out.leaf_instances = w.has_child.iter().filter(|c| !**c).count() as u64;
    out
}
