//! A functional unit under construction (docs/passes/merge.md §4.3): blocks, and per covered unit the
//! block of every node (μ) and the orientation of every commutative pair. Everything else follows:
//! a block port's sources are the images of the unit edges into it, two or more make a multiplexer,
//! and a configuration's selects are the positions of its own sources.

use std::collections::{BTreeMap, BTreeSet};

use super::library::{Lib, bits};
use super::unit::{UNode, Unit, result_type};
use crate::merge::json::{BlockKind, Price, Source};

/// Integer operators whose narrow result is the wide result truncated, with operands extended by
/// their signedness (two's-complement arithmetic and comparisons on extended values).
const WIDEN_OPS: &[&str] = &[
    "add",
    "subtract",
    "multiply",
    "negate",
    "and",
    "or",
    "xor",
    "not",
    "maximum",
    "minimum",
    "abs",
    "compare",
    "select",
    "divide",
    "remainder",
    "shift_left",
];

#[derive(Debug, Clone, PartialEq)]
pub struct FBlock {
    pub kind: BlockKind,
    pub op: String,
    pub attrs: BTreeMap<String, String>,
    pub label: String,
    /// The widest type served at every position.
    pub types: Vec<String>,
    /// Serves only literal constants of one value (no register).
    pub literal: bool,
    pub group: Option<u32>,
    pub member: u32,
}

/// One covered unit: its index in the family, the block of each node, and per node whether its two
/// commutative operands are exchanged on the block's ports.
#[derive(Debug, Clone, PartialEq)]
pub struct UCfg {
    pub unit: u32,
    pub map: Vec<u32>,
    pub swap: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fu {
    pub blocks: Vec<FBlock>,
    /// The pattern of each group.
    pub groups: Vec<String>,
    pub cfgs: Vec<UCfg>,
    /// The host class (every covered class's skeleton embeds into its skeleton).
    pub host: String,
}

/// Whether a block of this kind may serve narrower integers.
pub fn widen_ok(kind: BlockKind, op: &str, attrs: &BTreeMap<String, String>, types: &[String]) -> bool {
    match kind {
        BlockKind::Op => {
            WIDEN_OPS.contains(&op)
                && !(op == "compare"
                    && attrs.get("compare_type").map(String::as_str) == Some("UNSIGNED")
                    && types.iter().any(|t| t.starts_with('i') && t != "i1"))
        }
        _ => true,
    }
}

fn int(t: &str) -> Option<(bool, u32)> {
    if t == "i1" {
        return None;
    }
    if let Some(w) = t.strip_prefix("ui") {
        return w.parse().ok().map(|w| (false, w));
    }
    t.strip_prefix('i')
        .and_then(|w| w.parse().ok())
        .map(|w| (true, w))
}

/// The type serving both `a` and `b`: equal types, or the wider of two integers of one signedness.
pub fn join_type(a: &str, b: &str, widen: bool) -> Option<String> {
    if a == b {
        return Some(a.to_string());
    }
    if !widen {
        return None;
    }
    match (int(a), int(b)) {
        (Some((sa, wa)), Some((sb, wb))) if sa == sb => Some(if wa >= wb { a.into() } else { b.into() }),
        _ => None,
    }
}

pub fn join_sig(a: &[String], b: &[String], widen: bool) -> Option<Vec<String>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter().zip(b).map(|(x, y)| join_type(x, y, widen)).collect()
}

/// The join of a node's signatures over its tuples.
pub fn node_types(n: &UNode) -> Option<Vec<String>> {
    let widen = widen_ok(
        n.kind,
        &n.op,
        &n.attrs,
        n.sigs.first().map(Vec::as_slice).unwrap_or(&[]),
    );
    let mut it = n.sigs.iter();
    let mut acc = it.next()?.clone();
    for s in it {
        acc = join_sig(&acc, s, widen)?;
    }
    Some(acc)
}

/// The type at port `p` of a block (operators: operand `p`; an accumulator's ports and an output: its
/// value type).
pub fn port_type(b: &FBlock, p: usize) -> &str {
    match b.kind {
        BlockKind::Op => b.types.get(p).map(String::as_str).unwrap_or(""),
        _ => b.types.first().map(String::as_str).unwrap_or(""),
    }
}

/// The type of result `res` of a block.
pub fn out_type(b: &FBlock) -> &str {
    match b.kind {
        BlockKind::Op => result_type(&b.types).map(String::as_str).unwrap_or(""),
        _ => b.types.first().map(String::as_str).unwrap_or(""),
    }
}

impl Fu {
    /// The unit `u` (index `ui` in its family) as a unit of its own.
    pub fn single(u: &Unit, ui: u32) -> Result<Fu, String> {
        let mut blocks = Vec::new();
        let mut group_of = vec![None; u.nodes.len()];
        for (g, (_, members)) in u.groups.iter().enumerate() {
            for (m, &n) in members.iter().enumerate() {
                group_of[n as usize] = Some((g as u32, m as u32));
            }
        }
        for (i, n) in u.nodes.iter().enumerate() {
            let types =
                node_types(n).ok_or_else(|| format!("{} {}: tuples of different kinds", u.id, n.name))?;
            let (group, member) = match group_of[i] {
                Some((g, m)) => (Some(g), m),
                None => (None, 0),
            };
            blocks.push(FBlock {
                kind: n.kind,
                op: n.op.clone(),
                attrs: n.attrs.clone(),
                label: n.label(),
                types,
                literal: n.kind == BlockKind::Const && n.var.is_none(),
                group,
                member,
            });
        }
        Ok(Fu {
            blocks,
            groups: u.groups.iter().map(|(p, _)| p.clone()).collect(),
            cfgs: vec![UCfg {
                unit: ui,
                map: (0..u.nodes.len() as u32).collect(),
                swap: vec![false; u.nodes.len()],
            }],
            host: u.class.clone(),
        })
    }

    /// The block port that operand `i` of node `n` uses in configuration `c`.
    pub fn port(c: &UCfg, n: usize, i: usize, node: &UNode) -> usize {
        if node.commutative && c.swap[n] { 1 - i } else { i }
    }

    /// Per block, per port, the sorted set of sources.
    pub fn wiring(&self, units: &[Unit]) -> Vec<Vec<Vec<Source>>> {
        let mut w: Vec<Vec<BTreeSet<Source>>> = self
            .blocks
            .iter()
            .map(|b| vec![BTreeSet::new(); arity(b)])
            .collect();
        for c in &self.cfgs {
            let u = &units[c.unit as usize];
            for (n, node) in u.nodes.iter().enumerate() {
                let b = c.map[n] as usize;
                for (i, &(s, res)) in node.args.iter().enumerate() {
                    let p = Self::port(c, n, i, node);
                    if p >= w[b].len() {
                        w[b].resize(p + 1, BTreeSet::new());
                    }
                    w[b][p].insert(Source {
                        block: c.map[s as usize],
                        res,
                    });
                }
            }
        }
        w.into_iter()
            .map(|ps| ps.into_iter().map(|s| s.into_iter().collect()).collect())
            .collect()
    }
}

/// Operand ports of a block.
pub fn arity(b: &FBlock) -> usize {
    match b.kind {
        BlockKind::Op => b
            .types
            .iter()
            .filter(|t| !t.starts_with("rp:"))
            .count()
            .saturating_sub(1),
        BlockKind::Acc => 2,
        BlockKind::Output => 1,
        BlockKind::Const | BlockKind::Input => 0,
    }
}

/// A unit's prices and timing.
#[derive(Debug, Clone)]
pub struct Costed {
    pub wiring: Vec<Vec<Vec<Source>>>,
    pub blocks: Vec<Price>,
    /// `(block, port, inputs, width, price)`.
    pub muxes: Vec<(u32, u32, u32, u32, Price)>,
    pub area: f64,
    pub depth: f64,
}

pub fn cost(fu: &Fu, units: &[Unit], lib: &Lib) -> Costed {
    let wiring = fu.wiring(units);
    let blocks: Vec<Price> = fu
        .blocks
        .iter()
        .map(|b| lib.block(b.kind, &b.op, &b.attrs, &b.types, b.literal))
        .collect();
    let mut muxes = Vec::new();
    for (b, ps) in wiring.iter().enumerate() {
        for (p, s) in ps.iter().enumerate() {
            if s.len() >= 2 {
                let w = bits(port_type(&fu.blocks[b], p));
                muxes.push((b as u32, p as u32, s.len() as u32, w, lib.mux(s.len() as u32, w)));
            }
        }
    }
    let area =
        blocks.iter().map(|p| p.area_um2).sum::<f64>() + muxes.iter().map(|m| m.4.area_um2).sum::<f64>();
    let depth = depth(fu, &wiring, &blocks, &muxes);
    Costed {
        wiring,
        blocks,
        muxes,
        area,
        depth,
    }
}

/// The critical path: sources (inputs, constants, accumulator outputs) at 0, each operator adds its
/// delay, each multiplexer its delay. `None` entries of a cycle are ignored (callers check cycles).
fn depth(
    fu: &Fu,
    wiring: &[Vec<Vec<Source>>],
    blocks: &[Price],
    muxes: &[(u32, u32, u32, u32, Price)],
) -> f64 {
    let mux: BTreeMap<(u32, u32), f64> = muxes.iter().map(|m| ((m.0, m.1), m.4.delay_ps)).collect();
    let order = match topo(fu, wiring) {
        Ok(o) => o,
        Err(_) => return f64::INFINITY,
    };
    let mut at = vec![0.0f64; fu.blocks.len()];
    let mut worst = 0.0f64;
    for b in order {
        let blk = &fu.blocks[b];
        let mut t = 0.0f64;
        for (p, srcs) in wiring[b].iter().enumerate() {
            let m = mux.get(&(b as u32, p as u32)).copied().unwrap_or(0.0);
            for s in srcs {
                let src = s.block as usize;
                let a = if fu.blocks[src].kind == BlockKind::Acc {
                    0.0
                } else {
                    at[src]
                };
                t = t.max(a + m);
            }
        }
        let own = if blk.kind == BlockKind::Op {
            blocks[b].delay_ps
        } else {
            0.0
        };
        at[b] = t + own;
        worst = worst.max(at[b]);
    }
    worst
}

/// Blocks in dependence order; edges out of accumulator registers do not count. `Err` holds a cycle.
pub fn topo(fu: &Fu, wiring: &[Vec<Vec<Source>>]) -> Result<Vec<usize>, Vec<u32>> {
    let n = fu.blocks.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ps) in wiring.iter().enumerate() {
        for s in ps.iter().flatten() {
            if fu.blocks[s.block as usize].kind != BlockKind::Acc {
                succ[s.block as usize].push(b);
            }
        }
    }
    for s in &mut succ {
        s.sort_unstable();
        s.dedup();
    }
    // Iterative DFS with colours; a back edge gives the cycle.
    let mut colour = vec![0u8; n];
    let mut parent = vec![usize::MAX; n];
    let mut post = Vec::with_capacity(n);
    for root in 0..n {
        if colour[root] != 0 {
            continue;
        }
        let mut stack = vec![(root, 0usize)];
        colour[root] = 1;
        while let Some(&mut (v, ref mut i)) = stack.last_mut() {
            if *i < succ[v].len() {
                let w = succ[v][*i];
                *i += 1;
                match colour[w] {
                    0 => {
                        colour[w] = 1;
                        parent[w] = v;
                        stack.push((w, 0));
                    }
                    1 => {
                        let mut cyc = vec![w as u32];
                        let mut x = v;
                        while x != w && x != usize::MAX {
                            cyc.push(x as u32);
                            x = parent[x];
                        }
                        cyc.reverse();
                        return Err(cyc);
                    }
                    _ => {}
                }
            } else {
                colour[v] = 2;
                post.push(v);
                stack.pop();
            }
        }
    }
    post.reverse();
    Ok(post)
}
