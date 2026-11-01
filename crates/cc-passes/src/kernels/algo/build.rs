//! A builder of small normalized graphs (tests and the examples of docs/passes/kernels.md). Every
//! tensor has float elements of class 0; dims are given as class numbers (`0` is the literal 1 nowhere:
//! every number is a class variable).

use std::collections::BTreeMap;

use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};

use crate::components::algo::graph::{ADim, ANode, AType, Family, Kind};
use crate::normalize::json::NNode;

#[derive(Debug, Default, Clone)]
pub struct GraphBuilder {
    pub g: Graph<NNode>,
    inputs: u32,
}

/// A tensor type with the given dim classes.
pub fn ty(dims: &[u32], family: Family) -> AType {
    AType {
        family,
        elem: 0,
        dims: dims.iter().map(|&d| ADim::Var(d)).collect(),
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn node(&mut self, kind: Kind, results: Vec<AType>) -> u32 {
        let id = self.g.node_count() as u32;
        self.g.add_node(NNode {
            label: ANode {
                kind,
                attrs: BTreeMap::new(),
                results,
                child: None,
            },
            omega: None,
            splats: Vec::new(),
            origin: id,
        })
    }

    fn link(&mut self, src: u32, dst: u32, port: PortLabel) {
        self.g.add_edge(src, dst, EdgeEnd { res: 0, port });
    }

    pub fn input(&mut self, t: AType) -> u32 {
        let k = self.inputs;
        self.inputs += 1;
        self.node(Kind::Input { index: Some(k) }, vec![t])
    }

    /// An op with one result reading `operands` on ports 0, 1, ….
    pub fn op(&mut self, name: &str, t: AType, operands: &[u32]) -> u32 {
        let n = self.node(
            Kind::Op {
                name: name.into(),
                target: None,
            },
            vec![t],
        );
        for (p, &o) in operands.iter().enumerate() {
            self.link(o, n, PortLabel::Port(p as u32));
        }
        n
    }

    /// An op that owns one region with `args` arguments: `body(builder, args) -> yielded values`.
    pub fn with_region(
        &mut self,
        name: &str,
        t: AType,
        operands: &[u32],
        args: u32,
        body: impl FnOnce(&mut Self, &[u32]) -> Vec<u32>,
    ) -> u32 {
        let n = self.op(name, t.clone(), operands);
        let scalar = AType {
            family: t.family.clone(),
            elem: t.elem,
            dims: Vec::new(),
        };
        let a: Vec<u32> = (0..args)
            .map(|i| {
                let x = self.node(Kind::RegionArg { region: 0, index: i }, vec![scalar.clone()]);
                self.link(n, x, PortLabel::Region { region: 0, index: i });
                x
            })
            .collect();
        let ys = body(self, &a);
        let ret = self.node(
            Kind::Op {
                name: "stablehlo.return".into(),
                target: None,
            },
            Vec::new(),
        );
        for (p, y) in ys.into_iter().enumerate() {
            self.link(y, ret, PortLabel::Port(p as u32));
        }
        self.link(ret, n, PortLabel::Yield { region: 0 });
        n
    }

    /// `reduce` of `x` with init `init` and a binary combiner `comb`.
    pub fn reduce(&mut self, comb: &str, t: AType, x: u32, init: u32) -> u32 {
        let comb = comb.to_string();
        self.with_region("stablehlo.reduce", t, &[x, init], 2, move |b, a| {
            let s = AType {
                family: Family::Float,
                elem: 0,
                dims: Vec::new(),
            };
            vec![b.op(&comb, s, &[a[0], a[1]])]
        })
    }

    pub fn constant(&mut self, t: AType) -> u32 {
        self.op("stablehlo.constant", t, &[])
    }

    /// `func.return` of the values.
    pub fn ret(&mut self, values: &[u32]) -> u32 {
        let n = self.node(
            Kind::Op {
                name: "func.return".into(),
                target: None,
            },
            Vec::new(),
        );
        for (p, &v) in values.iter().enumerate() {
            self.link(v, n, PortLabel::Port(p as u32));
        }
        n
    }

    pub fn finish(self) -> Graph<NNode> {
        self.g
    }
}
