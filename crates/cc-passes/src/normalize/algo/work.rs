//! The working graph of a rewrite run: nodes keep the input component's ids (removed nodes leave a hole),
//! edges are a multiset of ends (`x * x` has two equal ends), and every change goes through an [`Edit`].

use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};

use crate::normalize::json::{Edit, NNode};

#[derive(Debug, Clone)]
pub struct Work {
    pub nodes: Vec<Option<NNode>>,
    /// Per node, its incoming ends `(src, end)`, sorted (duplicates kept).
    pub ins: Vec<Vec<(u32, EdgeEnd)>>,
    /// Per node, its outgoing ends `(dst, end)`, sorted (duplicates kept).
    pub outs: Vec<Vec<(u32, EdgeEnd)>>,
}

/// A data port (operand), as opposed to region and yield structure.
pub fn is_data(p: &PortLabel) -> bool {
    matches!(p, PortLabel::Port(_) | PortLabel::Commutative)
}

impl Work {
    pub fn from_graph(g: &Graph<NNode>) -> Self {
        let n = g.node_count();
        let mut w = Work {
            nodes: g.nodes.iter().cloned().map(Some).collect(),
            ins: vec![Vec::new(); n],
            outs: vec![Vec::new(); n],
        };
        for (u, o) in g.out.iter().enumerate() {
            for (v, ends) in o {
                for e in ends {
                    w.link(u as u32, *v, *e);
                }
            }
        }
        w
    }

    pub fn node(&self, n: u32) -> Option<&NNode> {
        self.nodes.get(n as usize).and_then(Option::as_ref)
    }

    pub fn live(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.as_ref().map(|_| i as u32))
    }

    fn link(&mut self, u: u32, v: u32, e: EdgeEnd) {
        let o = &mut self.outs[u as usize];
        let at = o.partition_point(|x| x <= &(v, e));
        o.insert(at, (v, e));
        let i = &mut self.ins[v as usize];
        let at = i.partition_point(|x| x <= &(u, e));
        i.insert(at, (u, e));
    }

    fn unlink(&mut self, u: u32, v: u32, e: EdgeEnd) -> Result<(), String> {
        let o = &mut self.outs[u as usize];
        let Some(a) = o.iter().position(|x| *x == (v, e)) else {
            return Err(format!("no edge {u} -> {v} {e:?}"));
        };
        o.remove(a);
        let i = &mut self.ins[v as usize];
        if let Some(b) = i.iter().position(|x| *x == (u, e)) {
            i.remove(b);
        }
        Ok(())
    }

    /// Apply one edit; errors leave the graph unchanged.
    pub fn apply(&mut self, e: &Edit) -> Result<(), String> {
        match e {
            Edit::Unlink { src, dst, end } => self.unlink(*src, *dst, *end),
            Edit::Link { src, dst, end } => {
                if self.node(*src).is_none() || self.node(*dst).is_none() {
                    return Err(format!("link {src} -> {dst}: node removed"));
                }
                self.link(*src, *dst, *end);
                Ok(())
            }
            Edit::Relabel { node, value } => match self.nodes.get_mut(*node as usize) {
                Some(Some(n)) if n.origin == value.origin => {
                    *n = (**value).clone();
                    Ok(())
                }
                Some(Some(_)) => Err(format!("relabel {node}: origin changed")),
                _ => Err(format!("relabel {node}: no node")),
            },
            Edit::Remove { node } => {
                let i = *node as usize;
                if self.node(*node).is_none() {
                    return Err(format!("remove {node}: no node"));
                }
                if !self.ins[i].is_empty() || !self.outs[i].is_empty() {
                    return Err(format!("remove {node}: node still has edges"));
                }
                self.nodes[i] = None;
                Ok(())
            }
        }
    }

    /// Data uses of result `res` of `n`: `(dst, end)`.
    pub fn uses(&self, n: u32, res: u32) -> Vec<(u32, EdgeEnd)> {
        self.outs[n as usize]
            .iter()
            .filter(|(_, e)| e.res == res && is_data(&e.port))
            .copied()
            .collect()
    }

    /// Every data use of `n`.
    pub fn all_uses(&self, n: u32) -> impl Iterator<Item = &(u32, EdgeEnd)> {
        self.outs[n as usize].iter().filter(|(_, e)| is_data(&e.port))
    }

    /// Incoming data ends `(src, end)`.
    pub fn data_ins(&self, n: u32) -> impl Iterator<Item = &(u32, EdgeEnd)> {
        self.ins[n as usize].iter().filter(|(_, e)| is_data(&e.port))
    }

    /// The single data operand at `Port(0)` (or the only data operand), if exactly one exists.
    pub fn operand0(&self, n: u32) -> Option<(u32, EdgeEnd)> {
        let v: Vec<_> = self.data_ins(n).copied().collect();
        match v.as_slice() {
            [x] => Some(*x),
            _ => None,
        }
    }

    /// Whether `n` owns a region or yields to an owner.
    pub fn has_regions(&self, n: u32) -> bool {
        let i = n as usize;
        self.outs[i].iter().any(|(_, e)| !is_data(&e.port))
            || self.ins[i].iter().any(|(_, e)| !is_data(&e.port))
    }

    /// The compacted graph: live nodes in ascending id.
    pub fn compact(&self) -> Graph<NNode> {
        let mut map = vec![u32::MAX; self.nodes.len()];
        let mut g: Graph<NNode> = Graph::default();
        for n in self.live() {
            if let Some(x) = self.node(n) {
                map[n as usize] = g.add_node(x.clone());
            }
        }
        for u in self.live() {
            for (v, e) in &self.outs[u as usize] {
                g.add_edge(map[u as usize], map[*v as usize], *e);
            }
        }
        g
    }
}
