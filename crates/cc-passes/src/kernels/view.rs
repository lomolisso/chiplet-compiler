//! `views/<id>.dot` (kernels as clusters, boundary ops as grey boxes) and `views/kernels.md` (the
//! statistics).

use std::fmt::Write as _;

use cc_ir::absgraph::{Graph, PortLabel};

use super::algo::Partition;
use super::json::{KernelsSummary, PartitionSummary, Role};
use crate::components::algo::graph::{ADim, Kind};
use crate::normalize::json::NNode;

fn label(n: &NNode) -> String {
    let base = match &n.label.kind {
        Kind::Input { index } => format!("input {}", index.map_or("*".into(), |i| i.to_string())),
        Kind::Output => "output".into(),
        Kind::RegionArg { region, index } => format!("arg {region}.{index}"),
        Kind::SizeIn => "size_in".into(),
        Kind::Op { name, target } => {
            let s = name.strip_prefix("stablehlo.").unwrap_or(name).to_string();
            match target {
                Some(t) => format!("{s} @{t}"),
                None => s,
            }
        }
    };
    let dims: Vec<String> = n
        .label
        .results
        .first()
        .map(|t| {
            t.dims
                .iter()
                .map(|d| match d {
                    ADim::One => "1".into(),
                    ADim::Var(v) => format!("d{v}"),
                })
                .collect()
        })
        .unwrap_or_default();
    let mut s = base;
    if let Some(c) = &n.label.child {
        s.push_str(&format!(" {c}"));
    }
    if !n.splats.is_empty() {
        s.push_str(&format!(" +{} splat", n.splats.len()));
    }
    if !dims.is_empty() {
        s.push_str(&format!("\\n[{}]", dims.join(",")));
    }
    s.replace('"', "'")
}

pub fn dot(id: &str, g: &Graph<NNode>, p: &Partition) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "digraph \"{id}\" {{\n  rankdir=TB;\n  node [shape=box, fontsize=10];"
    );
    for k in &p.kernels {
        let title = match k.core {
            Some(c) => format!("{} core {}", k.id, super::core_key(g, c)),
            None => format!("{} map", k.id),
        };
        let _ = writeln!(s, "  subgraph cluster_{} {{\n    label=\"{title}\";", k.id);
        for m in &k.members {
            let style = match m.role {
                Role::Core => ", style=filled, fillcolor=\"#f4c27a\"",
                Role::Body => ", style=dashed",
                Role::Prologue => ", style=filled, fillcolor=\"#cfe3f7\"",
                Role::Epilogue => ", style=filled, fillcolor=\"#d7efd2\"",
                Role::Map => ", style=filled, fillcolor=\"#eadcf3\"",
                Role::Broadcast => ", style=dotted",
            };
            let _ = writeln!(
                s,
                "    n{} [label=\"{}\\n{}\"{style}];",
                m.node,
                label(&g.nodes[m.node as usize]),
                m.role.name()
            );
        }
        s.push_str("  }\n");
    }
    for b in &p.boundaries {
        let _ = writeln!(
            s,
            "  n{} [label=\"{}\\n{}\", style=filled, fillcolor=\"#e6e6e6\", color=\"#888888\"];",
            b.node,
            label(&g.nodes[b.node as usize]),
            b.kind.name()
        );
    }
    for (u, out) in g.out.iter().enumerate() {
        for (v, ends) in out {
            let lab: Vec<String> = ends
                .iter()
                .map(|e| match e.port {
                    PortLabel::Port(k) => format!("{}:{k}", e.res),
                    PortLabel::Commutative => format!("{}:*", e.res),
                    PortLabel::Region { region, index } => format!("r{region}.{index}"),
                    PortLabel::Yield { region } => format!("y{region}"),
                })
                .collect();
            let _ = writeln!(s, "  n{u} -> n{v} [label=\"{}\", fontsize=8];", lab.join(" "));
        }
    }
    s.push_str("}\n");
    s
}

pub fn table(t: &KernelsSummary, ss: &[PartitionSummary]) -> String {
    let mut s = String::from("# Kernels\n\n");
    let _ = writeln!(
        s,
        "{} components: {} kernels ({} with a core, {} map kernels); {} map kernels read a composite \
         call's result; {} map kernels have rank at most 1; {} evictions.\n",
        t.components, t.kernels, t.core_kernels, t.map_kernels, t.call_fed_maps, t.low_rank_maps, t.evictions
    );
    s.push_str("## Kernels per component\n\n| kernels | components |\n|---:|---:|\n");
    for (k, n) in &t.kernels_per_component {
        let _ = writeln!(s, "| {k} | {n} |");
    }
    s.push_str("\n## Cores\n\n| core | kernels | kernels × instances |\n|---|---:|---:|\n");
    for (k, (n, w)) in &t.cores {
        let _ = writeln!(s, "| `{k}` | {n} | {w} |");
    }
    s.push_str("\n## Members per role\n\n| role | members |\n|---|---:|\n");
    for (k, n) in &t.roles {
        let _ = writeln!(s, "| {k} | {n} |");
    }
    s.push_str("\n## Boundary nodes\n\n| kind | nodes |\n|---|---:|\n");
    for (k, n) in &t.boundaries {
        let _ = writeln!(s, "| {k} | {n} |");
    }
    s.push_str(
        "\n## Per model (weighted by instances)\n\n| model | kernels | with a core |\n|---|---:|---:|\n",
    );
    for (m, (a, b)) in &t.models {
        let _ = writeln!(s, "| {m} | {a} | {b} |");
    }
    s.push_str("\n## Components\n\n| component | nodes | kernels | core | map | instances |\n|---|---:|---:|---:|---:|---:|\n");
    for c in ss {
        let _ = writeln!(
            s,
            "| `{}` | {} | {} | {} | {} | {} |",
            c.id, c.nodes, c.kernels, c.core_kernels, c.map_kernels, c.instances
        );
    }
    s
}
