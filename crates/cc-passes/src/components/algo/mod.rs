//! The components algorithm (docs/passes/components.md §4): per phase, instances and their abstract
//! graphs (`graph`, `classes`); across phases, deduplication (`dedup`).

pub mod classes;
pub mod dedup;
pub mod graph;

use std::collections::BTreeMap;

use cc_ir::absgraph::{Graph, Hash128, UnionFind, hash_bytes, wl};
use cc_ir::{Attr, Def, Dim, Elem, FuncId, Module, OpId, SymCtx};

use classes::Iface;
use graph::{ANode, Facts, NodeRef, Param};

/// The abstract graph of one decomposition or anon region of a phase, with the bindings of its own
/// concrete types (every call site of a shared decomposition has the same).
#[derive(Debug, Clone)]
pub struct PhaseGraph {
    pub graph: Graph<ANode>,
    pub refs: Vec<NodeRef>,
    pub params: Vec<Vec<Param>>,
    pub dims: Vec<Dim>,
    pub elems: Vec<Elem>,
    /// Stable WL colours.
    pub colors: Vec<Hash128>,
    pub hash: Hash128,
    /// Per node: the graph (index in the phase) of a call node's child.
    pub children: Vec<Option<u32>>,
    pub height: u32,
    pub iface: Iface,
}

/// An instance of a phase before deduplication.
#[derive(Debug, Clone)]
pub struct Draft {
    pub id: String,
    pub anon: bool,
    pub name: Option<String>,
    pub attrs: BTreeMap<String, Attr>,
    /// Composite-call op ids from the entry.
    pub path: Vec<u32>,
    pub parent: Option<String>,
    pub children: Vec<String>,
    /// The function whose ops it owns (`None`: entry ops of an anon region).
    pub func: Option<u32>,
    /// Owned ops at `path` (sorted); the call op itself is owned at the parent's path.
    pub ops: Vec<u32>,
    pub size_ops: u64,
    /// The graph in the phase (`None`: no compute op).
    pub graph: Option<u32>,
}

/// Everything one phase contributes.
#[derive(Debug, Clone)]
pub struct PhaseOut {
    pub model: String,
    pub phase: String,
    pub graphs: Vec<PhaseGraph>,
    pub instances: Vec<Draft>,
    /// The entry's terminator (not in any instance).
    pub root: Option<u32>,
    pub ops: u64,
}

pub fn instance_id(model: &str, phase: &str, path: &[u32]) -> String {
    let mut s = format!("{model}/{phase}");
    for p in path {
        s.push('/');
        s.push_str(&p.to_string());
    }
    s
}

pub fn anon_id(model: &str, phase: &str, ops: &[u32]) -> String {
    let mut x = blake3::Hasher::new();
    x.update(model.as_bytes());
    x.update(&[0]);
    x.update(phase.as_bytes());
    x.update(&[0]);
    for o in ops {
        x.update(&o.to_le_bytes());
    }
    let hex: String = x.finalize().as_bytes()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{model}/{phase}/anon:{hex}")
}

fn node_init(key: &[u8], child: Option<Hash128>) -> Hash128 {
    let mut b = key.to_vec();
    if let Some(c) = child {
        b.extend_from_slice(&c.to_le_bytes());
    }
    hash_bytes(&b)
}

#[derive(Debug)]
struct Phase<'m> {
    f: Facts<'m>,
    ctx: SymCtx,
    wl_rounds: u32,
    graphs: Vec<PhaseGraph>,
    of_func: BTreeMap<FuncId, u32>,
    func_ops: Vec<Vec<OpId>>,
}

impl Phase<'_> {
    /// Finish a built graph: classes, bindings, keys, WL.
    fn finish(&mut self, mut b: graph::Built) -> Result<u32, String> {
        let children: Vec<Option<u32>> = b.callee.iter().map(|c| c.map(|f| self.of_func[&f])).collect();
        let graphs = &self.graphs;
        let ifc = |n: u32| children[n as usize].map(|g| graphs[g as usize].iface.clone());
        let (nd, ne) = classes::assign(&mut b.graph, &ifc)?;
        let (dims, elems) = classes::bind(&b.graph, &b.types, nd, ne, &self.ctx)?;
        let keys: Vec<Vec<u8>> = b.graph.nodes.iter().map(ANode::key).collect();
        let child_hash: Vec<Option<Hash128>> = children
            .iter()
            .map(|c| c.map(|g| self.graphs[g as usize].hash))
            .collect();
        let idx: Vec<usize> = (0..b.graph.node_count()).collect();
        let ig = Graph {
            nodes: idx,
            out: b.graph.out.clone(),
            inc: b.graph.inc.clone(),
        };
        let w = wl(&ig, |&i| node_init(&keys[i], child_hash[i]), self.wl_rounds);
        let height = children
            .iter()
            .flatten()
            .map(|&g| self.graphs[g as usize].height + 1)
            .max()
            .unwrap_or(0);
        let iface = classes::iface(&b.graph);
        self.graphs.push(PhaseGraph {
            graph: b.graph,
            refs: b.refs,
            params: b.params,
            dims,
            elems,
            colors: w.colors,
            hash: w.hash,
            children,
            height,
            iface,
        });
        Ok((self.graphs.len() - 1) as u32)
    }

    /// The graph of a decomposition, built once, children first.
    fn func_graph(&mut self, func: FuncId) -> Result<u32, String> {
        if let Some(&g) = self.of_func.get(&func) {
            return Ok(g);
        }
        let ops = self.func_ops[func.index()].clone();
        for &o in &ops {
            if self.f.is_call(o)
                && let Some(c) = self.f.m.op(o).callee
            {
                self.func_graph(c)?;
            }
        }
        let b = graph::build(&self.f, &ops, Some(func))
            .map_err(|e| format!("{}: {e}", self.f.m.func(func).name))?;
        let g = self
            .finish(b)
            .map_err(|e| format!("{}: {e}", self.f.m.func(func).name))?;
        self.of_func.insert(func, g);
        Ok(g)
    }
}

/// Instances and graphs of one phase.
pub fn phase(m: &Module, wl_rounds: u32) -> Result<PhaseOut, String> {
    let f = Facts::new(m);
    let mut func_ops: Vec<Vec<OpId>> = vec![Vec::new(); m.funcs.len()];
    for i in 0..m.ops.len() as u32 {
        func_ops[f.op_func(OpId(i)).index()].push(OpId(i));
    }
    let mut ph = Phase {
        ctx: SymCtx::new(&m.symbols, &m.constraints),
        f,
        wl_rounds,
        graphs: Vec::new(),
        of_func: BTreeMap::new(),
        func_ops,
    };
    let (model, phase) = (m.model.as_str(), m.phase.as_str());
    let mut instances = Vec::new();
    // Composite instances, depth first from the entry.
    let mut stack: Vec<(FuncId, Vec<u32>, Option<String>)> = vec![(m.entry, Vec::new(), None)];
    while let Some((func, path, parent)) = stack.pop() {
        let mut kids = Vec::new();
        for &o in &ph.func_ops[func.index()].clone() {
            if !ph.f.is_call(o) {
                continue;
            }
            let op = m.op(o);
            let callee = op
                .callee
                .ok_or_else(|| format!("composite {} without decomposition", o.0))?;
            let g = ph.func_graph(callee)?;
            let mut p = path.clone();
            p.push(o.0);
            let id = instance_id(model, phase, &p);
            let own: Vec<u32> = ph.func_ops[callee.index()]
                .iter()
                .filter(|&&x| !ph.f.is_call(x))
                .map(|x| x.0)
                .collect();
            let size_ops = own.iter().filter(|&&x| ph.f.shape_only[x as usize]).count() as u64;
            let mut attrs = BTreeMap::new();
            for k in ["composite_attributes", "version"] {
                if let Some(a) = op.attrs.get(k) {
                    attrs.insert(k.to_string(), a.clone());
                }
            }
            let empty = ph.graphs[g as usize].graph.node_count() == 0;
            kids.push(id.clone());
            instances.push(Draft {
                id: id.clone(),
                anon: false,
                name: op.attrs.get("name").and_then(Attr::as_str).map(str::to_string),
                attrs,
                path: p.clone(),
                parent: parent.clone(),
                children: Vec::new(),
                func: Some(callee.0),
                ops: own,
                size_ops,
                graph: (!empty).then_some(g),
            });
            stack.push((callee, p, Some(id)));
        }
        if let Some(pid) = &parent
            && let Some(d) = instances.iter_mut().rev().find(|d| &d.id == pid)
        {
            d.children = kids;
        }
    }
    // Anon regions of the entry.
    let entry_ops = ph.func_ops[m.entry.index()].clone();
    let body = m.func(m.entry).body;
    let root = m.region(body).ops.last().copied();
    let cand: Vec<OpId> = entry_ops
        .iter()
        .copied()
        .filter(|&o| {
            let t = ph.f.top(o);
            !ph.f.is_call(t) && Some(t) != root
        })
        .collect();
    let mut uf = UnionFind::new(m.ops.len());
    let in_cand: std::collections::HashSet<OpId> = cand.iter().copied().collect();
    let top = |o: OpId| ph.f.top(o);
    // Nested ops join their top-level op.
    for &o in &cand {
        uf.union(o.0, top(o).0);
    }
    // Compute ops: connected by def-use at non-shape ports.
    for &o in &cand {
        let op = m.op(o);
        for (p, &v) in op.operands.iter().enumerate() {
            if let Def::Result { op: u, .. } = m.value(v).def
                && in_cand.contains(&u)
            {
                let both_graph = ph.f.in_graph(o) && ph.f.in_graph(u);
                let both_shape = !ph.f.in_graph(o) && !ph.f.in_graph(u);
                let shape_port = crate::import::algo::size_tag::is_shape_port(m, op, p);
                if (both_graph && !shape_port && !op.dim_as_data.contains(&(p as u32))) || both_shape {
                    uf.union(o.0, u.0);
                }
            }
        }
    }
    // Shape-only groups join the region of their first compute consumer.
    let mut groups: BTreeMap<u32, Vec<OpId>> = BTreeMap::new();
    for &o in &cand {
        if !ph.f.in_graph(top(o)) {
            groups.entry(uf.find(o.0)).or_default().push(o);
        }
    }
    for members in groups.values() {
        let consumer = members
            .iter()
            .flat_map(|&o| m.op(o).results.iter())
            .flat_map(|r| ph.f.users[r.index()].iter())
            .map(|&(u, _)| u)
            .filter(|u| in_cand.contains(u) && ph.f.in_graph(top(*u)))
            .min();
        if let Some(c) = consumer {
            uf.union(members[0].0, c.0);
        }
    }
    let mut regions: BTreeMap<u32, Vec<OpId>> = BTreeMap::new();
    for &o in &cand {
        regions.entry(uf.find(o.0)).or_default().push(o);
    }
    let mut regions: Vec<Vec<OpId>> = regions.into_values().collect();
    for r in &mut regions {
        r.sort();
    }
    regions.sort();
    for ops in regions {
        let own: Vec<u32> = ops.iter().map(|o| o.0).collect();
        let size_ops = own.iter().filter(|&&x| ph.f.shape_only[x as usize]).count() as u64;
        let has_compute = ops.iter().any(|&o| ph.f.in_graph(o));
        let g = if has_compute {
            let b =
                graph::build(&ph.f, &ops, None).map_err(|e| format!("anon region at op {}: {e}", own[0]))?;
            Some(
                ph.finish(b)
                    .map_err(|e| format!("anon region at op {}: {e}", own[0]))?,
            )
        } else {
            None
        };
        instances.push(Draft {
            id: anon_id(model, phase, &own),
            anon: true,
            name: None,
            attrs: BTreeMap::new(),
            path: Vec::new(),
            parent: None,
            children: Vec::new(),
            func: None,
            ops: own,
            size_ops,
            graph: g,
        });
    }
    instances.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(PhaseOut {
        model: model.into(),
        phase: phase.into(),
        graphs: ph.graphs,
        instances,
        root: root.map(|o| o.0),
        ops: m.ops.len() as u64,
    })
}
