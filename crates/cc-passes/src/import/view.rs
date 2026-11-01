//! `views/<model>/<phase>.dot`: the entry function with every composite collapsed into one node; size
//! computation and constants are left out.

use std::collections::HashMap;
use std::fmt::Write;

use cc_ir::{Def, Module, OpClass};

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn dot(m: &Module) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "digraph \"{}/{}\" {{", esc(&m.model), esc(&m.phase));
    let _ = writeln!(s, "  rankdir=TB; node [shape=box, fontsize=10];");
    let body = m.region(m.func(m.entry).body);
    let mut shown: HashMap<u32, usize> = HashMap::new();
    for (i, &o) in body.ops.iter().enumerate() {
        let op = m.op(o);
        let class = cc_ir::lookup(&op.name, op.target.as_deref()).map(|e| e.class);
        if op.size || matches!(class, Some(OpClass::Constant)) {
            continue;
        }
        let label = match (op.name.as_str(), op.attrs.get("name")) {
            ("stablehlo.composite", Some(cc_ir::Attr::Str(n))) => {
                let path = match op.attrs.get("composite_attributes") {
                    Some(cc_ir::Attr::Dict(d)) => match d.get("cc_path") {
                        Some(cc_ir::Attr::Str(p)) => p.clone(),
                        _ => String::new(),
                    },
                    _ => String::new(),
                };
                format!("{n}\\n{}", esc(&path))
            }
            (n, _) => n.trim_start_matches("stablehlo.").to_string(),
        };
        let shape = if op.name == "stablehlo.composite" {
            ", style=filled, fillcolor=lightgrey"
        } else {
            ""
        };
        let _ = writeln!(
            s,
            "  n{i} [label=\"{}\"{shape}];",
            esc(&label).replace("\\\\n", "\\n")
        );
        shown.insert(o.0, i);
    }
    for (i, &o) in body.ops.iter().enumerate() {
        if !shown.contains_key(&o.0) {
            continue;
        }
        for v in &m.op(o).operands {
            if let Def::Result { op, .. } = m.value(*v).def
                && let Some(j) = shown.get(&op.0)
            {
                let _ = writeln!(s, "  n{j} -> n{i};");
            }
        }
    }
    s.push_str("}\n");
    s
}
