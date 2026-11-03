//! Families, units and the GraMi encoding (docs/passes/mine.md §4.1).
//!
//! A unit is one datapath class (route `datapath`) or one stage of a template class. Its graph has one
//! node per non-constant vertex; a dataflow edge between two such vertices becomes a merged, labelled
//! edge (`absgraph::Graph`: the ends are (result, port), `Commutative` for a `*` port). Constants,
//! inputs, accumulators and core results are not nodes: their uses are pattern inputs.

use std::collections::{BTreeSet, HashMap};

use cc_ir::absgraph::{EdgeEnd, EdgeLabel, Graph, PortLabel};

use super::lg::LgGraph;
use crate::datapaths::json::{Datapath, Src, Stage, Vertex};
use crate::iteration_space::json::{Direction, Route, SkelClass, Skeleton};
use crate::kernel_classes::json::InstancesTable;
use crate::mine::json::{Family, LABELS_SCHEMA, LGMAP_SCHEMA, Labels, LgMap, Unit};

/// Attributes that change what an op computes (value columns are parameters, not identity).
pub const SEMANTIC_ATTRS: &[&str] = &["comparison_direction", "compare_type"];

/// The node label key of a vertex: `op@family`, then `[v1,v2]` with the values of its semantic
/// attributes in `SEMANTIC_ATTRS` order.
pub fn node_key(v: &Vertex) -> String {
    let vals: Vec<&str> = SEMANTIC_ATTRS
        .iter()
        .filter_map(|k| v.attrs.get(*k).map(String::as_str))
        .collect();
    if vals.is_empty() {
        v.label.clone()
    } else {
        format!("{}[{}]", v.label, vals.join(","))
    }
}

/// The edge end of an operand port: `*` is commutative, `p<i>` is port `i`.
pub fn port_label(port: &str) -> Result<PortLabel, String> {
    if port == "*" {
        return Ok(PortLabel::Commutative);
    }
    port.strip_prefix('p')
        .and_then(|i| i.parse().ok())
        .map(PortLabel::Port)
        .ok_or_else(|| format!("unknown port {port:?}"))
}

/// A unit with its graph (node labels are keys until `intern`).
#[derive(Debug, Clone)]
pub struct UnitGraph {
    pub family: Family,
    pub unit: Unit,
    pub graph: Graph<String>,
}

/// The id of a stage unit.
pub fn stage_unit_id(class: &str, stage: Stage) -> Option<String> {
    match stage {
        Stage::Prologue { port } => Some(format!("{class}/prologue{port}")),
        Stage::Epilogue => Some(format!("{class}/epilogue")),
        Stage::Core | Stage::Map => None,
    }
}

/// Trip-count-weighted elements of a template's core operand: input `port` (prologue) or the first
/// result (epilogue).
pub fn stage_workload(skel: &Skeleton, table: &InstancesTable, stage: Stage) -> Result<u128, String> {
    let ins: Vec<_> = skel
        .operands
        .iter()
        .filter(|o| o.core && o.direction == Direction::In)
        .collect();
    let op = match stage {
        Stage::Prologue { port } => ins.get(port as usize).copied(),
        Stage::Epilogue => skel
            .operands
            .iter()
            .find(|o| o.core && o.direction == Direction::Out),
        _ => None,
    }
    .ok_or_else(|| format!("{}: no core operand for {stage:?}", skel.id))?;
    let mut total = 0u128;
    for r in &table.rows {
        let mut n = r.trip_count as u128;
        for &a in &op.axes {
            let e = r
                .extents
                .get(a as usize)
                .ok_or_else(|| format!("{}: axis class {a} not in a row", skel.id))?;
            n = n.saturating_mul(*e as u128);
        }
        total = total.saturating_add(n);
    }
    Ok(total)
}

fn family_of(class: SkelClass) -> Family {
    match class {
        SkelClass::Map => Family::Map,
        SkelClass::Reduce => Family::Reduce,
        SkelClass::Contraction => Family::Contraction,
    }
}

/// The graph of the vertices `members` of `dp` (in that order).
fn unit_graph(dp: &Datapath, members: &[&Vertex]) -> Result<Graph<String>, String> {
    let at: HashMap<u32, u32> = members
        .iter()
        .enumerate()
        .map(|(i, v)| (v.id, i as u32))
        .collect();
    let mut g = Graph::default();
    for v in members {
        g.add_node(node_key(v));
    }
    for (i, v) in members.iter().enumerate() {
        for a in &v.args {
            if let Src::Vertex { v: u, res } = &a.src
                && let Some(&j) = at.get(u)
            {
                let port = port_label(&a.port).map_err(|e| format!("{} vertex {}: {e}", dp.id, v.id))?;
                g.add_edge(j, i as u32, EdgeEnd { res: *res, port });
            }
        }
    }
    Ok(g)
}

/// Every unit, sorted by (family, id). `skeletons` and `tables` are keyed by class id.
pub fn units(
    dps: &[Datapath],
    skeletons: &HashMap<String, Skeleton>,
    tables: &HashMap<String, InstancesTable>,
) -> Result<Vec<UnitGraph>, String> {
    let mut out = Vec::new();
    for dp in dps {
        let table = tables
            .get(&dp.id)
            .ok_or_else(|| format!("{}: no instance table", dp.id))?;
        let live: Vec<&Vertex> = dp.vertices.iter().filter(|v| v.op != "const").collect();
        match dp.route {
            Route::Datapath => {
                let workload = table.rows.iter().map(|r| r.weighted as u128).sum();
                out.push(UnitGraph {
                    family: family_of(dp.class),
                    unit: Unit {
                        id: dp.id.clone(),
                        class: dp.id.clone(),
                        stage: None,
                        workload,
                        first: 0,
                        vertices: live.iter().map(|v| v.id).collect(),
                    },
                    graph: unit_graph(dp, &live)?,
                });
            }
            Route::Template => {
                let skel = skeletons
                    .get(&dp.id)
                    .ok_or_else(|| format!("{}: no skeleton", dp.id))?;
                let stages: BTreeSet<Stage> = live.iter().map(|v| v.stage).collect();
                for st in stages {
                    let id = stage_unit_id(&dp.id, st)
                        .ok_or_else(|| format!("{}: template vertex in stage {st:?}", dp.id))?;
                    let members: Vec<&Vertex> = live.iter().copied().filter(|v| v.stage == st).collect();
                    out.push(UnitGraph {
                        family: Family::Stage,
                        unit: Unit {
                            id,
                            class: dp.id.clone(),
                            stage: Some(st),
                            workload: stage_workload(skel, table, st)?,
                            first: 0,
                            vertices: members.iter().map(|v| v.id).collect(),
                        },
                        graph: unit_graph(dp, &members)?,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| (a.family, &a.unit.id).cmp(&(b.family, &b.unit.id)));
    Ok(out)
}

/// The label tables: every node key and edge label of every unit, sorted.
pub fn labels(units: &[UnitGraph]) -> Labels {
    let mut nodes = BTreeSet::new();
    let mut edges: BTreeSet<EdgeLabel> = BTreeSet::new();
    for u in units {
        nodes.extend(u.graph.nodes.iter().cloned());
        for o in &u.graph.out {
            edges.extend(o.iter().map(|e| e.1.clone()));
        }
    }
    Labels {
        schema: LABELS_SCHEMA.into(),
        nodes: nodes.into_iter().collect(),
        edges: edges.into_iter().collect(),
    }
}

/// Integer labels.
#[derive(Debug, Clone)]
pub struct Interner {
    pub nodes: HashMap<String, u32>,
    pub edges: HashMap<EdgeLabel, u32>,
}

impl Interner {
    pub fn new(l: &Labels) -> Self {
        Self {
            nodes: l
                .nodes
                .iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), i as u32))
                .collect(),
            edges: l
                .edges
                .iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), i as u32))
                .collect(),
        }
    }
}

/// One family's GraMi input: the `.lg` text, its map, and the units' graphs with integer node labels.
#[derive(Debug, Clone)]
pub struct FamilyGraph {
    pub family: Family,
    pub lg: String,
    pub map: LgMap,
    pub graphs: Vec<Graph<u32>>,
}

/// The `.lg` text of a family (GraMi's `Graph.loadFromFile_Ehab`: a header line, `v <id> <label>` with
/// ids `0, 1, …` in order, then `e <src> <dst> <label>`; no blank line), the units' first vertices
/// assigned in unit order.
pub fn family_graph(family: Family, units: &[UnitGraph], it: &Interner) -> Result<FamilyGraph, String> {
    let mut lg = LgGraph::default();
    let mut map_units = Vec::new();
    let mut graphs = Vec::new();
    let mut next = 0u32;
    for u in units.iter().filter(|u| u.family == family) {
        let g = u.graph.map(|k| it.nodes[k]);
        lg.nodes.extend(g.nodes.iter().copied());
        for (s, o) in g.out.iter().enumerate() {
            for (d, l) in o {
                let id = it
                    .edges
                    .get(l)
                    .ok_or_else(|| format!("{}: edge label not interned", u.unit.id))?;
                lg.edges.push((next + s as u32, next + d, *id));
            }
        }
        let mut unit = u.unit.clone();
        unit.first = next;
        next += g.node_count() as u32;
        map_units.push(unit);
        graphs.push(g);
    }
    let edges = lg.edges.len() as u64;
    Ok(FamilyGraph {
        family,
        lg: lg.to_text(),
        map: LgMap {
            schema: LGMAP_SCHEMA.into(),
            family,
            units: map_units,
            vertices: next as u64,
            edges,
        },
        graphs,
    })
}
