//! The StableHLO specification's index formulas, transcribed a second time and independently of the
//! op table's iteration hooks (https://openxla.org/stablehlo/spec). For every member of a class's
//! representative kernel, the stored access of each operand is claimed equal to the index the spec
//! gives for it: in terms of the iteration vector for the core (whose iteration convention is checked
//! here as well), and in terms of the member's stored result access for every other member. The
//! claims become the SMT obligation of the class (`proofs.rs`).

use cc_ir::Attr;
use cc_ir::absgraph::Graph;
use cc_ir::ops::iter::Tag;

use crate::components::algo::graph::{AAttr, ADim, ANode};
use crate::iteration_space::json::{Access, Extent, Factor, Row, Skeleton, Term};
use crate::kernel_classes::json::{ClassFile, Group as GroupKind};
use crate::kernels::json::KNode;

/// One claim: the stored row equals the specified row.
#[derive(Debug, Clone)]
pub struct Claim {
    pub what: String,
    pub stored: Row,
    pub spec: Row,
}

fn row(terms: Vec<Term>, divisor: Option<Factor>, padded: bool, data: bool) -> Row {
    Row {
        terms,
        divisor,
        padded,
        data,
    }
}

fn var(d: u32) -> Term {
    Term {
        k: 1,
        factors: Vec::new(),
        dim: Some(d),
    }
}

fn idx(d: u32) -> Row {
    row(vec![var(d)], None, false, false)
}

fn zero() -> Row {
    row(Vec::new(), None, false, false)
}

/// A size attribute's entry: `Ok(Some(v))` literal, `Ok(None)` absent, else a column of the class.
enum Entry {
    Lit(i64),
    Sym(Factor),
}

fn entry(node: &ANode, n: u32, name: &str, i: u32, default: i64) -> Result<Entry, String> {
    match node.attrs.get(name) {
        None => Ok(Entry::Lit(default)),
        Some(AAttr::Sized { entries, .. }) => match entries.get(i as usize) {
            Some(Some(v)) => Ok(Entry::Lit(*v)),
            Some(None) => Ok(Entry::Sym(Factor::Column {
                column: format!("n{n}.{name}"),
                index: i,
            })),
            None => Err(format!("`{name}` has no entry {i}")),
        },
        Some(AAttr::Lit(a)) => a
            .as_ints()
            .and_then(|v| v.get(i as usize).copied())
            .map(Entry::Lit)
            .ok_or_else(|| format!("`{name}` is not an integer list")),
        Some(AAttr::Value { .. }) => Err(format!("`{name}` is a value")),
    }
}

/// `c · e · i_dim` for an entry `e`.
fn scaled(e: &Entry, k: i64, dim: Option<u32>) -> Option<Term> {
    match e {
        Entry::Lit(0) => None,
        Entry::Lit(v) => Some(Term {
            k: k * v,
            factors: Vec::new(),
            dim,
        }),
        Entry::Sym(f) => Some(Term {
            k,
            factors: vec![f.clone()],
            dim,
        }),
    }
}

fn divisor(e: Entry) -> Result<Option<Factor>, String> {
    match e {
        Entry::Lit(1) => Ok(None),
        Entry::Lit(v) => Err(format!("literal dilation {v}")),
        Entry::Sym(f) => Ok(Some(f)),
    }
}

fn lit<'a>(node: &'a ANode, k: &str) -> Option<&'a Attr> {
    match node.attrs.get(k)? {
        AAttr::Lit(a) => Some(a),
        _ => None,
    }
}

/// Producers by port (`(node, result)`), from the edges.
fn producers(g: &Graph<KNode>, n: u32) -> Vec<Option<(u32, u32)>> {
    crate::kernel_classes::algo::local::operands(g)
        .get(n as usize)
        .cloned()
        .unwrap_or_default()
}

fn dims_of(g: &Graph<KNode>, n: u32, r: u32) -> Vec<ADim> {
    match &g.nodes[n as usize] {
        KNode::Member { node, .. } => node.label.results.get(r as usize).map(|t| t.dims.clone()),
        KNode::Input { ty, .. } => (r == 0).then(|| ty.dims.clone()),
        KNode::Output { .. } => None,
    }
    .unwrap_or_default()
}

/// The expected iteration dims of a core: `(tag, extent)` per dim.
type Expect = Vec<(Tag, Extent)>;

struct Core<'a> {
    class: &'a ClassFile,
    g: &'a Graph<KNode>,
    n: u32,
    node: &'a ANode,
    prod: Vec<Option<(u32, u32)>>,
}

impl Core<'_> {
    fn class_of(&self, n: u32, r: u32, a: u32) -> Result<Extent, String> {
        self.class
            .local
            .dims
            .get(n as usize)
            .and_then(|x| x.get(r as usize))
            .and_then(|x| x.get(a as usize))
            .map(|&c| Extent::Class { class: c })
            .ok_or_else(|| format!("no class for node {n} result {r} axis {a}"))
    }
    fn res(&self, a: u32) -> Result<Extent, String> {
        self.class_of(self.n, 0, a)
    }
    fn opnd(&self, port: usize, a: u32) -> Result<Extent, String> {
        let (pn, pr) = self.prod.get(port).copied().flatten().ok_or("missing operand")?;
        self.class_of(pn, pr, a)
    }
    fn rank(&self, port: usize) -> usize {
        self.prod
            .get(port)
            .copied()
            .flatten()
            .map_or(0, |(pn, pr)| dims_of(self.g, pn, pr).len())
    }
    fn grouped(&self, k: &str) -> bool {
        self.class
            .representative
            .groups
            .get(&self.n)
            .and_then(|m| m.get(k))
            .is_some_and(|x| *x != GroupKind::One)
    }
}

/// `#dot_general`: `result_index = batch ++ lhs_free ++ rhs_free`; `lhs_index` takes the batch index
/// at `lhs_batching_dimensions`, the contracting index at `lhs_contracting_dimensions`, the lhs free
/// index elsewhere (rhs likewise).
fn dot(c: &Core<'_>) -> Result<(Expect, Vec<Access>, Access), String> {
    let Some(Attr::DotDims(d)) = lit(c.node, "dot_dimension_numbers") else {
        return Err("no dot_dimension_numbers".into());
    };
    let (nl, nr) = (c.rank(0), c.rank(1));
    let nb = d.lhs_batching.len();
    let in_list = |v: &[i64], a: usize| v.iter().position(|&x| x as usize == a);
    let lfree: Vec<usize> = (0..nl)
        .filter(|&a| in_list(&d.lhs_batching, a).is_none() && in_list(&d.lhs_contracting, a).is_none())
        .collect();
    let rfree: Vec<usize> = (0..nr)
        .filter(|&a| in_list(&d.rhs_batching, a).is_none() && in_list(&d.rhs_contracting, a).is_none())
        .collect();
    let nres = nb + lfree.len() + rfree.len();
    let mut exp: Expect = (0..nres as u32)
        .map(|a| Ok((Tag::Par, c.res(a)?)))
        .collect::<Result<_, String>>()?;
    for &a in &d.lhs_contracting {
        exp.push((Tag::Red, c.opnd(0, a as u32)?));
    }
    let side = |n: usize, batch: &[i64], contr: &[i64], free: &[usize], off: usize| -> Access {
        (0..n)
            .map(|a| {
                if let Some(j) = in_list(batch, a) {
                    idx(j as u32)
                } else if let Some(j) = in_list(contr, a) {
                    idx((nres + j) as u32)
                } else {
                    idx((off + free.iter().position(|&x| x == a).unwrap_or(0)) as u32)
                }
            })
            .collect()
    };
    let lhs = side(nl, &d.lhs_batching, &d.lhs_contracting, &lfree, nb);
    let rhs = side(nr, &d.rhs_batching, &d.rhs_contracting, &rfree, nb + lfree.len());
    Ok((exp, vec![lhs, rhs], (0..nres as u32).map(idx).collect()))
}

/// `#convolution` / `#dynamic_conv`. Per spatial dim `i`, output index `o_i` and window index `r_i`
/// read the padded, lhs-dilated input at `o_i · window_strides[i] + r_i · rhs_dilation[i]`, which is
/// input element `(… − padding[i][0]) / lhs_dilation[i]`; `window_reversal[i]` reads the kernel at
/// `R_i − 1 − r_i`. With `feature_group_count` G > 1 the spec splits lhs along the input feature and rhs
/// along the kernel output feature into G parts and concatenates the results along the output feature:
/// group `g`, in-group output feature `m`, in-group input feature `c` give result feature
/// `g·(F/G) + m`, input feature `g·(C/G) + c` and kernel output feature `g·(F/G) + m`.
fn conv(c: &Core<'_>, dynamic: bool) -> Result<(Expect, Vec<Access>, Access), String> {
    let Some(Attr::ConvDims(d)) = lit(c.node, "dimension_numbers") else {
        return Err("no dimension_numbers".into());
    };
    if c.grouped("batch_group_count") {
        return Err("batch_group_count > 1".into());
    }
    let grouped = c.grouped("feature_group_count");
    let nres = dims_of(c.g, c.n, 0).len();
    let (ni, nk) = (c.rank(0), c.rank(1));
    let ns = d.input_spatial.len();
    let of = d.output_feature as u32;
    let mut exp: Expect = Vec::new();
    for a in 0..nres as u32 {
        if a == of && grouped {
            let e = match entry(c.node, c.n, "feature_group_count", 0, 1)? {
                Entry::Lit(v) => Extent::Const { value: v },
                Entry::Sym(Factor::Column { column, index }) => Extent::Column { column, index },
                Entry::Sym(_) => return Err("feature_group_count".into()),
            };
            exp.push((Tag::Par, e));
        } else {
            exp.push((Tag::Par, c.res(a)?));
        }
    }
    let m = if grouped {
        let by = match entry(c.node, c.n, "feature_group_count", 0, 1)? {
            Entry::Sym(f) => f,
            Entry::Lit(_) => return Err("a literal group count".into()),
        };
        let Extent::Class { class } = c.res(of)? else {
            return Err("output feature".into());
        };
        exp.push((Tag::Par, Extent::Quot { class, by }));
        Some(exp.len() as u32 - 1)
    } else {
        None
    };
    let k0 = exp.len() as u32;
    for &ka in &d.kernel_spatial {
        exp.push((Tag::Red, c.opnd(1, ka as u32)?));
    }
    let ci = exp.len() as u32;
    exp.push((Tag::Red, c.opnd(1, d.kernel_input_feature as u32)?));
    let o = |a: i64| a as u32; // result axis a is dim a
    let group_row = |inner: u32| -> Row {
        row(
            vec![
                Term {
                    k: 1,
                    factors: vec![Factor::Extent { dim: inner }],
                    dim: Some(o(d.output_feature)),
                },
                var(inner),
            ],
            None,
            false,
            false,
        )
    };
    let mut input = vec![zero(); ni];
    let mut kernel = vec![zero(); nk];
    let mut result: Access = (0..nres as u32).map(idx).collect();
    input[d.input_batch as usize] = idx(o(d.output_batch));
    match m {
        None => {
            input[d.input_feature as usize] = idx(ci);
            kernel[d.kernel_output_feature as usize] = idx(o(d.output_feature));
        }
        Some(m) => {
            input[d.input_feature as usize] = group_row(ci);
            kernel[d.kernel_output_feature as usize] = group_row(m);
            result[of as usize] = group_row(m);
        }
    }
    kernel[d.kernel_input_feature as usize] = idx(ci);
    let reversal: Vec<bool> = match lit(c.node, "window_reversal") {
        Some(Attr::Bools(b)) => b.clone(),
        _ => vec![false; ns],
    };
    for i in 0..ns {
        let (od, rd) = (o(d.output_spatial[i]), k0 + i as u32);
        let mut terms = Vec::new();
        terms.extend(scaled(
            &entry(c.node, c.n, "window_strides", i as u32, 1)?,
            1,
            Some(od),
        ));
        terms.extend(scaled(
            &entry(c.node, c.n, "rhs_dilation", i as u32, 1)?,
            1,
            Some(rd),
        ));
        let pad = if dynamic {
            Entry::Sym(Factor::Runtime {
                node: c.n,
                port: 2,
                index: 2 * i as u32,
            })
        } else {
            entry(c.node, c.n, "padding", 2 * i as u32, 0)?
        };
        terms.extend(scaled(&pad, -1, None));
        input[d.input_spatial[i] as usize] = row(
            terms,
            divisor(entry(c.node, c.n, "lhs_dilation", i as u32, 1)?)?,
            true,
            false,
        );
        kernel[d.kernel_spatial[i] as usize] = if reversal.get(i).copied().unwrap_or(false) {
            row(
                vec![
                    Term {
                        k: 1,
                        factors: vec![Factor::Extent { dim: rd }],
                        dim: None,
                    },
                    Term {
                        k: -1,
                        factors: Vec::new(),
                        dim: None,
                    },
                    Term {
                        k: -1,
                        factors: Vec::new(),
                        dim: Some(rd),
                    },
                ],
                None,
                false,
                false,
            )
        } else {
            idx(rd)
        };
    }
    Ok((exp, vec![input, kernel], result))
}

/// `#reduce`: the input is visited at every index; `result_index` drops the `dimensions` axes; init
/// values are scalars.
fn reduce(c: &Core<'_>) -> Result<(Expect, Vec<Access>, Access), String> {
    let red: Vec<u32> = lit(c.node, "dimensions")
        .and_then(Attr::as_ints)
        .ok_or("no dimensions")?
        .into_iter()
        .map(|a| a as u32)
        .collect();
    let n = c.rank(0);
    let k = c.node.results.len();
    let exp: Expect = (0..n as u32)
        .map(|a| Ok((if red.contains(&a) { Tag::Red } else { Tag::Par }, c.opnd(0, a)?)))
        .collect::<Result<_, String>>()?;
    let input: Access = (0..n as u32).map(idx).collect();
    let ops = (0..c.prod.len())
        .map(|p| if p < k { input.clone() } else { Vec::new() })
        .collect();
    Ok((
        exp,
        ops,
        (0..n as u32).filter(|a| !red.contains(a)).map(idx).collect(),
    ))
}

/// `#reduce_window` (and `dynamic_reduce_window`, whose window attributes are the five shape operands
/// after the inputs and init values): the padded, base-dilated input is read at
/// `o_i · window_strides[i] + w_i · window_dilations[i]`, input element `(… − padding[i][0]) /
/// base_dilations[i]`.
fn windowed(c: &Core<'_>, dynamic: bool) -> Result<(Expect, Vec<Access>, Access), String> {
    let n = c.rank(0);
    let k = c.node.results.len();
    let e = |name: &str, slot: u32, i: u32, default: i64| -> Result<Entry, String> {
        if dynamic {
            Ok(Entry::Sym(Factor::Runtime {
                node: c.n,
                port: 2 * k as u32 + slot,
                index: i,
            }))
        } else {
            entry(c.node, c.n, name, i, default)
        }
    };
    let mut exp: Expect = (0..n as u32)
        .map(|a| Ok((Tag::Par, c.res(a)?)))
        .collect::<Result<_, String>>()?;
    for i in 0..n as u32 {
        let w = match e("window_dimensions", 0, i, 1)? {
            Entry::Lit(v) => Extent::Const { value: v },
            Entry::Sym(Factor::Column { column, index }) => Extent::Column { column, index },
            Entry::Sym(Factor::Runtime { node, port, index }) => Extent::Runtime { node, port, index },
            Entry::Sym(_) => return Err("window".into()),
        };
        exp.push((Tag::Red, w));
    }
    let input: Access = (0..n as u32)
        .map(|i| {
            let mut terms = Vec::new();
            terms.extend(scaled(&e("window_strides", 1, i, 1)?, 1, Some(i)));
            terms.extend(scaled(&e("window_dilations", 3, i, 1)?, 1, Some(n as u32 + i)));
            terms.extend(scaled(&e("padding", 4, 2 * i, 0)?, -1, None));
            Ok(row(terms, divisor(e("base_dilations", 2, i, 1)?)?, true, false))
        })
        .collect::<Result<_, String>>()?;
    let ops = (0..c.prod.len())
        .map(|p| if p < k { input.clone() } else { Vec::new() })
        .collect();
    Ok((exp, ops, (0..n as u32).map(idx).collect()))
}

/// `#scatter`: `update_scatter_dims` are the update axes outside `update_window_dims`; the target is
/// `full_start_index + full_batching_index + full_window_index`, where the window index places the
/// update window dims at the operand axes outside `inserted_window_dims` and `input_batching_dims`,
/// the batching index places the update scatter dim of each `scatter_indices_batching_dims` entry at
/// its `input_batching_dims` axis, and the start index (read from `scatter_indices`) is data.
fn scatter(c: &Core<'_>) -> Result<(Expect, Vec<Access>, Access), String> {
    let Some(Attr::ScatterDims(s)) = lit(c.node, "scatter_dimension_numbers") else {
        return Err("no scatter_dimension_numbers".into());
    };
    let k = c.node.results.len();
    let (ni, nu) = (c.rank(k), c.rank(k + 1));
    let no = dims_of(c.g, c.n, 0).len();
    let has = |v: &[i64], a: usize| v.iter().any(|&x| x as usize == a);
    let exp: Expect = (0..nu as u32)
        .map(|a| {
            Ok((
                if has(&s.update_window_dims, a as usize) {
                    Tag::Par
                } else {
                    Tag::Red
                },
                c.opnd(k + 1, a)?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let usd: Vec<u32> = (0..nu as u32)
        .filter(|&a| !has(&s.update_window_dims, a as usize))
        .collect();
    let ivd = s.index_vector_dim as usize;
    let idx_axes: Vec<usize> = (0..ni).filter(|&a| a != ivd).collect();
    let mut target = vec![zero(); no];
    let window_axes: Vec<usize> = (0..no)
        .filter(|&a| !has(&s.inserted_window_dims, a) && !has(&s.input_batching_dims, a))
        .collect();
    for (pos, &a) in window_axes.iter().enumerate() {
        if let Some(&w) = s.update_window_dims.get(pos) {
            target[a] = idx(w as u32);
        }
    }
    for (j, &a) in s.input_batching_dims.iter().enumerate() {
        let sb = *s.scatter_indices_batching_dims.get(j).ok_or("batching dims")? as usize;
        let pos = idx_axes.iter().position(|&x| x == sb).ok_or("batching dim")?;
        target[a as usize] = idx(*usd.get(pos).ok_or("scatter dims")?);
    }
    for &a in &s.scattered_dims_to_operand_dims {
        target[a as usize].data = true;
    }
    let mut indices = vec![zero(); ni];
    for (pos, &a) in idx_axes.iter().enumerate() {
        indices[a] = idx(*usd.get(pos).ok_or("scatter dims")?);
    }
    if ivd < ni {
        indices[ivd].data = true;
    }
    let updates: Access = (0..nu as u32).map(idx).collect();
    let ops = (0..c.prod.len())
        .map(|p| {
            if p < k {
                target.clone()
            } else if p == k {
                indices.clone()
            } else {
                updates.clone()
            }
        })
        .collect();
    Ok((exp, ops, target))
}

fn compose(r: &Row, sub: &[Row]) -> Row {
    // `index = i_j` is the row of `j` itself, divisor and guards included.
    if let [t] = r.terms.as_slice()
        && t.k == 1
        && t.factors.is_empty()
        && r.divisor.is_none()
        && let Some(j) = t.dim
        && let Some(s) = sub.get(j as usize)
    {
        return s.clone();
    }
    let mut terms = Vec::new();
    let mut data = r.data;
    for t in &r.terms {
        match t.dim {
            None => terms.push(t.clone()),
            Some(j) => {
                if let Some(s) = sub.get(j as usize) {
                    data |= s.data;
                    for u in &s.terms {
                        let mut f = t.factors.clone();
                        f.extend(u.factors.iter().cloned());
                        terms.push(Term {
                            k: t.k * u.k,
                            factors: f,
                            dim: u.dim,
                        });
                    }
                }
            }
        }
    }
    row(terms, r.divisor.clone(), r.padded, data)
}

/// Every claim of a class.
pub fn claims(class: &ClassFile, s: &Skeleton, g: &Graph<KNode>) -> Result<Vec<Claim>, String> {
    let mut out = Vec::new();
    let mut push = |what: String, stored: &Access, spec: &Access| -> Result<(), String> {
        if stored.len() != spec.len() {
            return Err(format!(
                "{what}: {} rows, the spec gives {}",
                stored.len(),
                spec.len()
            ));
        }
        for (i, (a, b)) in stored.iter().zip(spec).enumerate() {
            if a.data != b.data {
                return Err(format!("{what} row {i}: data dependence differs from the spec"));
            }
            out.push(Claim {
                what: format!("{what}[{i}]"),
                stored: a.clone(),
                spec: b.clone(),
            });
        }
        Ok(())
    };
    for acc in &s.accesses {
        let n = acc.node;
        let KNode::Member { node, .. } = &g.nodes[n as usize] else {
            return Err(format!("node {n} is not a member"));
        };
        let node = &node.label;
        let crate::components::algo::graph::Kind::Op { name, target } = &node.kind else {
            return Err(format!("node {n} is not an op"));
        };
        let prod = producers(g, n);
        if Some(n) == s.core_node {
            let c = Core {
                class,
                g,
                n,
                node,
                prod: prod.clone(),
            };
            let (exp, ops, res) = match (name.as_str(), target.as_deref()) {
                ("stablehlo.dot_general", _) => dot(&c)?,
                ("stablehlo.convolution", _) => conv(&c, false)?,
                ("stablehlo.dynamic_conv", _) => conv(&c, true)?,
                ("stablehlo.reduce", _) => reduce(&c)?,
                ("stablehlo.reduce_window", _) => windowed(&c, false)?,
                ("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")) => windowed(&c, true)?,
                ("stablehlo.scatter", _) => scatter(&c)?,
                (op, _) => return Err(format!("core `{op}` has no transcribed semantics")),
            };
            if exp.len() != s.dims.len() {
                return Err(format!(
                    "{} iteration dims, the spec convention gives {}",
                    s.dims.len(),
                    exp.len()
                ));
            }
            for (i, ((tag, ext), d)) in exp.iter().zip(&s.dims).enumerate() {
                if *tag != d.tag || *ext != d.extent {
                    return Err(format!(
                        "dim {i}: stored {:?} {:?}, the spec gives {tag:?} {ext:?}",
                        d.tag, d.extent
                    ));
                }
            }
            for (p, a) in acc.operands.iter().enumerate() {
                let (Some(a), Some(_)) = (a, prod.get(p).copied().flatten()) else {
                    continue;
                };
                push(format!("n{n}.operand{p}"), a, ops.get(p).ok_or("operand")?)?;
            }
            for (r, a) in acc.results.iter().enumerate() {
                push(format!("n{n}.result{r}"), a, &res)?;
            }
            continue;
        }
        let r = acc
            .results
            .first()
            .ok_or_else(|| format!("node {n}: no result access"))?;
        let rdims = node.results.first().map(|t| t.dims.clone()).unwrap_or_default();
        let spec_elementwise = cc_ir::lookup(name, target.as_deref()).is_some_and(|x| x.elementwise);
        for (p, a) in acc.operands.iter().enumerate() {
            let (Some(a), Some((pn, pr))) = (a, prod.get(p).copied().flatten()) else {
                continue;
            };
            let odims = dims_of(g, pn, pr);
            let spec: Access = match name.as_str() {
                // Element-wise: `operand_index = result_index`; a rank-0 operand is read whole.
                _ if spec_elementwise => {
                    if odims.len() == rdims.len() {
                        r.clone()
                    } else if odims.is_empty() {
                        Vec::new()
                    } else {
                        return Err(format!("node {n}: operand {p} of rank {}", odims.len()));
                    }
                }
                // `#broadcast_in_dim`: `operand_index[d] = 0` if `dim(operand, d) = 1` (and its result
                // axis is not; where both are 1 the result index is 0 too), else
                // `result_index[broadcast_dimensions[d]]`.
                "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim" => {
                    let bd: Vec<usize> = lit(node, "broadcast_dimensions")
                        .and_then(Attr::as_ints)
                        .ok_or("no broadcast_dimensions")?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect();
                    (0..odims.len())
                        .map(|d| {
                            let to = bd.get(d).copied().unwrap_or(usize::MAX);
                            let expands =
                                odims[d] == ADim::One && rdims.get(to).is_some_and(|x| *x != ADim::One);
                            if expands {
                                zero()
                            } else {
                                compose(&idx(to as u32), r)
                            }
                        })
                        .collect()
                }
                op => return Err(format!("node {n}: `{op}` has no transcribed semantics")),
            };
            push(format!("n{n}.operand{p}"), a, &spec)?;
        }
    }
    Ok(out)
}
