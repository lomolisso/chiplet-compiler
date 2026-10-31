//! The pass DAG (design §3.3), derived from `requires`/`provides` only. A cycle, a duplicate provider or a
//! requirement nobody provides is a configuration error.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::pass::Pass;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DagError {
    #[error("duplicate pass name `{0}`")]
    DuplicatePass(String),

    #[error("pass name `{0}` is not snake_case ([a-z][a-z0-9_]*)")]
    BadPassName(String),

    #[error("pass `{pass}`: invalid artifact kind `{kind}` (allowed: [a-z0-9_.])")]
    BadKind { pass: String, kind: String },

    #[error("artifact `{kind}` is provided by both `{first}` and `{second}`")]
    DuplicateProvider {
        kind: String,
        first: String,
        second: String,
    },

    #[error("pass `{pass}` requires `{kind}`, which no registered pass provides")]
    MissingProvider { pass: String, kind: String },

    #[error("cycle between passes: {}", .0.join(", "))]
    Cycle(Vec<String>),

    #[error("unknown pass `{name}` in {context}")]
    UnknownPass { name: String, context: String },
}

#[derive(Debug, Clone)]
pub struct PassGraph {
    names: Vec<&'static str>,
    index: BTreeMap<&'static str, usize>,
    graph: DiGraph<usize, &'static str>,
    nodes: Vec<NodeIndex>,
    providers: BTreeMap<&'static str, usize>,
}

fn is_snake_case(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

impl PassGraph {
    pub fn build(passes: &[Box<dyn Pass>]) -> Result<Self, DagError> {
        let mut names = Vec::new();
        let mut index = BTreeMap::new();
        let mut graph = DiGraph::new();
        let mut nodes = Vec::new();
        for (i, p) in passes.iter().enumerate() {
            let name = p.name();
            if !is_snake_case(name) {
                return Err(DagError::BadPassName(name.into()));
            }
            if index.insert(name, i).is_some() {
                return Err(DagError::DuplicatePass(name.into()));
            }
            names.push(name);
            nodes.push(graph.add_node(i));
        }
        let mut providers: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (i, p) in passes.iter().enumerate() {
            for k in p.provides().iter().chain(p.requires()) {
                if !k.is_valid() {
                    return Err(DagError::BadKind {
                        pass: names[i].into(),
                        kind: k.name().into(),
                    });
                }
            }
            for k in p.provides() {
                if let Some(&first) = providers.get(k.name()) {
                    return Err(DagError::DuplicateProvider {
                        kind: k.name().into(),
                        first: names[first].into(),
                        second: names[i].into(),
                    });
                }
                providers.insert(k.name(), i);
            }
        }
        for (i, p) in passes.iter().enumerate() {
            for k in p.requires() {
                let &prod = providers.get(k.name()).ok_or_else(|| DagError::MissingProvider {
                    pass: names[i].into(),
                    kind: k.name().into(),
                })?;
                graph.add_edge(nodes[prod], nodes[i], k.name());
            }
        }
        let g = Self {
            names,
            index,
            graph,
            nodes,
            providers,
        };
        g.check_acyclic()?;
        Ok(g)
    }

    fn check_acyclic(&self) -> Result<(), DagError> {
        let mut cycles: Vec<Vec<String>> = petgraph::algo::tarjan_scc(&self.graph)
            .into_iter()
            .filter(|scc| scc.len() > 1 || self.graph.contains_edge(scc[0], scc[0]))
            .map(|scc| {
                let mut v: Vec<String> = scc
                    .iter()
                    .map(|n| self.names[self.graph[*n]].to_string())
                    .collect();
                v.sort();
                v
            })
            .collect();
        cycles.sort();
        match cycles.into_iter().next() {
            Some(c) => Err(DagError::Cycle(c)),
            None => Ok(()),
        }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn name(&self, i: usize) -> &'static str {
        self.names[i]
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    pub fn resolve(&self, name: &str, context: &str) -> Result<usize, DagError> {
        self.index_of(name).ok_or_else(|| DagError::UnknownPass {
            name: name.into(),
            context: context.into(),
        })
    }

    pub fn producer(&self, kind: &str) -> Option<usize> {
        self.providers.get(kind).copied()
    }

    pub fn predecessors(&self, i: usize) -> BTreeSet<usize> {
        self.graph
            .neighbors_directed(self.nodes[i], Direction::Incoming)
            .map(|n| self.graph[n])
            .collect()
    }

    pub fn successors(&self, i: usize) -> BTreeSet<usize> {
        self.graph
            .neighbors_directed(self.nodes[i], Direction::Outgoing)
            .map(|n| self.graph[n])
            .collect()
    }

    fn closure(&self, roots: impl IntoIterator<Item = usize>, dir: Direction) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut queue: VecDeque<usize> = roots.into_iter().collect();
        while let Some(i) = queue.pop_front() {
            if seen.insert(i) {
                queue.extend(
                    self.graph
                        .neighbors_directed(self.nodes[i], dir)
                        .map(|n| self.graph[n]),
                );
            }
        }
        seen
    }

    /// The roots and everything they (transitively) depend on.
    pub fn with_ancestors(&self, roots: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
        self.closure(roots, Direction::Incoming)
    }

    /// The roots and everything that (transitively) depends on them.
    pub fn with_descendants(&self, roots: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
        self.closure(roots, Direction::Outgoing)
    }

    /// Kahn's order, ties broken by pass name (deterministic).
    pub fn topo_order(&self) -> Vec<usize> {
        let mut indeg: Vec<usize> = (0..self.len()).map(|i| self.predecessors(i).len()).collect();
        let mut ready: BTreeSet<(&str, usize)> = (0..self.len())
            .filter(|&i| indeg[i] == 0)
            .map(|i| (self.names[i], i))
            .collect();
        let mut out = Vec::new();
        while let Some((_, i)) = ready.pop_first() {
            out.push(i);
            for s in self.successors(i) {
                indeg[s] -= 1;
                if indeg[s] == 0 {
                    ready.insert((self.names[s], s));
                }
            }
        }
        out
    }

    /// Graphviz DOT, nodes and edges sorted.
    pub fn to_dot(&self) -> String {
        let mut s = String::from("digraph passes {\n  rankdir=LR;\n  node [shape=box];\n");
        let mut names: Vec<&str> = self.names.clone();
        names.sort();
        for n in names {
            s.push_str(&format!("  \"{n}\";\n"));
        }
        let mut edges: Vec<(&str, &str, &str)> = self
            .graph
            .edge_indices()
            .filter_map(|e| {
                let (a, b) = self.graph.edge_endpoints(e)?;
                Some((
                    self.names[self.graph[a]],
                    self.names[self.graph[b]],
                    self.graph[e],
                ))
            })
            .collect();
        edges.sort();
        for (a, b, k) in edges {
            s.push_str(&format!("  \"{a}\" -> \"{b}\" [label=\"{k}\"];\n"));
        }
        s.push_str("}\n");
        s
    }
}
