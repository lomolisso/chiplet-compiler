//! `views/datapaths.md`: the census of vertex labels and one row per class.

use std::fmt::Write as _;

use super::json::{Datapath, Summary};
use super::proofs::Obligations;

pub fn view(built: &[(Datapath, Obligations)], s: &Summary) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "# Datapaths\n");
    let _ = writeln!(
        o,
        "{} classes ({}), {} vertices, {} width tuples, {} proofs, {} order queries ({} free, {} bound by ω), {} encoding lemmas.\n",
        s.classes,
        s.by_route
            .iter()
            .map(|(k, v)| format!("{v} {k}"))
            .collect::<Vec<_>>()
            .join(", "),
        s.vertices,
        s.tuples,
        s.proofs,
        s.order_queries,
        s.order_free,
        s.order_bound,
        s.lemmas
    );
    if !s.uninterpreted.is_empty() {
        let _ = writeln!(o, "Uninterpreted functions: {}.\n", s.uninterpreted.join(", "));
    }
    let _ = writeln!(o, "| label | vertices |\n|---|---:|");
    let mut labels: Vec<(&String, &u64)> = s.labels.iter().collect();
    labels.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (l, n) in labels {
        let _ = writeln!(o, "| `{l}` | {n} |");
    }
    let _ = writeln!(
        o,
        "\n| class | route | core | vertices | inputs | outputs | tuples | order |\n|---|---|---|---:|---:|---:|---:|---|"
    );
    for (dp, _) in built {
        let order = dp
            .accumulator
            .as_ref()
            .map(|a| {
                a.order
                    .iter()
                    .map(|x| if x.free { "free" } else { "ω" })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let _ = writeln!(
            o,
            "| `{}` | {} | `{}` | {} | {} | {} | {} | {} |",
            dp.id,
            serde_json::to_value(dp.route)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
            dp.core,
            dp.vertices.len(),
            dp.inputs.len(),
            dp.outputs.len(),
            dp.tuples.len(),
            order
        );
    }
    o
}
