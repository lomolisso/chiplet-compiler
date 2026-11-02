//! Iteration-space semantics of the ops (design §4.7): for an op, its iteration dims (parallel or
//! reduction) and, per operand and result, the affine map from the iteration vector to the element it
//! reads or writes. Every formula is the StableHLO specification's index formula
//! (https://openxla.org/stablehlo/spec, the anchor named in each function).
//!
//! The maps are parametric: strides, dilations, padding and window sizes are named by their attribute
//! and entry (`Factor::Attr`) or by their shape operand (`Factor::Runtime`) rather than by value, so one
//! map serves every instance of a kernel class. The caller substitutes the literal entries it knows.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::Attr;

/// Parallel (indexes the result) or reduction (combined away).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Tag {
    Par,
    Red,
}

/// A symbolic factor of a coefficient.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Factor {
    /// Entry `index` of the size attribute `name` (row-major for `N×2` padding: `2i` low, `2i + 1` high).
    Attr { name: String, index: u32 },
    /// Entry `index` of the integer vector at operand `port` (a shape operand).
    Runtime { port: u32, index: u32 },
    /// The extent of iteration dim `dim`.
    Extent { dim: u32 },
}

/// `k · Π factors · i_dim` (no dim: a constant term).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Term {
    pub k: i64,
    pub factors: Vec<Factor>,
    pub dim: Option<u32>,
}

/// One index of an operand or result: `(Σ terms) / divisor`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Row {
    pub terms: Vec<Term>,
    /// Exact division (base or lhs dilation): only iterations where it divides read an element.
    pub divisor: Option<Factor>,
    /// Out-of-range indices read the padding value (a domain guard, not part of the map).
    pub padded: bool,
    /// The index also depends on data (scatter): the terms are its affine part.
    pub data: bool,
}

impl Row {
    pub fn dim(d: u32) -> Row {
        Row {
            terms: vec![Term {
                k: 1,
                factors: Vec::new(),
                dim: Some(d),
            }],
            divisor: None,
            padded: false,
            data: false,
        }
    }

    pub fn zero() -> Row {
        Row {
            terms: Vec::new(),
            divisor: None,
            padded: false,
            data: false,
        }
    }
}

/// The access of one value: one row per axis.
pub type Access = Vec<Row>;

/// Where an iteration dim's extent comes from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Extent {
    /// Axis `axis` of result `res`.
    Result { res: u32, axis: u32 },
    /// Axis `axis` of operand `port`.
    Operand { port: u32, axis: u32 },
    /// A size attribute or shape-operand entry.
    Factor(Factor),
    /// Axis `axis` of result `res`, divided by a factor (a convolution's features per group).
    Quot { res: u32, axis: u32, by: Factor },
}

/// One iteration dim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Dim {
    pub tag: Tag,
    pub extent: Extent,
    /// Its axis in the space of the op's reduction order ω (`normalize`): for `dot_general` a lhs axis,
    /// for a convolution a kernel axis, for `reduce` an operand axis, for a windowed reduction a window
    /// axis.
    pub omega: Option<u32>,
    /// What the dim is, for people (`batch`, `lhs_free`, `contract`, `out_spatial`, `window`, …).
    pub role: String,
}

/// The iteration semantics of one op.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OpIteration {
    pub dims: Vec<Dim>,
    /// Per operand port; `None` for a shape operand (not data).
    pub operands: Vec<Option<Access>>,
    pub results: Vec<Access>,
}

/// What a hook sees of one op.
pub struct IterIn<'a> {
    /// Rank of every data operand port (in port order; shape operands are not data and have none).
    pub operands: &'a [Option<usize>],
    pub results: &'a [usize],
    /// A literal attribute (dimension numbers, `dimensions`, `window_reversal`, …).
    pub lit: &'a dyn Fn(&str) -> Option<&'a Attr>,
    /// Whether an attribute is present at all (size attributes have spec defaults when absent).
    pub has: &'a dyn Fn(&str) -> bool,
    /// Operand axes of port 0 that a broadcast expands (an operand extent of 1 in every instance).
    pub expanded: &'a [u32],
    /// Whether the op's `feature_group_count` (resp. `batch_group_count`) is not 1 in its instances.
    pub feature_grouped: bool,
    pub batch_grouped: bool,
}

impl std::fmt::Debug for IterIn<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IterIn")
            .field("operands", &self.operands)
            .field("results", &self.results)
            .field("expanded", &self.expanded)
            .field("feature_grouped", &self.feature_grouped)
            .field("batch_grouped", &self.batch_grouped)
            .finish_non_exhaustive()
    }
}

/// Iteration-space semantics of an op: the hook in the op table.
pub type IterationFn = fn(&IterIn<'_>) -> Result<OpIteration, String>;

fn ints(x: &IterIn<'_>, k: &str) -> Result<Vec<u32>, String> {
    (x.lit)(k)
        .and_then(Attr::as_ints)
        .map(|v| v.into_iter().map(|a| a as u32).collect())
        .ok_or_else(|| format!("no `{k}`"))
}

fn rank(x: &IterIn<'_>, port: usize) -> Result<usize, String> {
    x.operands
        .get(port)
        .copied()
        .flatten()
        .ok_or_else(|| format!("operand {port} has no rank"))
}

fn par(res: u32, axis: u32, role: &str) -> Dim {
    Dim {
        tag: Tag::Par,
        extent: Extent::Result { res, axis },
        omega: None,
        role: role.into(),
    }
}

fn term(k: i64, factors: Vec<Factor>, dim: Option<u32>) -> Term {
    Term { k, factors, dim }
}

/// Element-wise ops (spec: each op's constraint `shape(operands) = shape(result)`; `clamp` and `select`
/// also take rank-0 operands, which every element reads).
pub fn elementwise(x: &IterIn<'_>) -> Result<OpIteration, String> {
    let r = *x.results.first().ok_or("no result")?;
    let dims = (0..r as u32).map(|a| par(0, a, "elem")).collect();
    let id: Access = (0..r as u32).map(Row::dim).collect();
    let operands = x
        .operands
        .iter()
        .map(|o| o.map(|n| if n == r { id.clone() } else { Vec::new() }))
        .collect();
    Ok(OpIteration {
        dims,
        operands,
        results: x.results.iter().map(|_| id.clone()).collect(),
    })
}

/// `#broadcast_in_dim`: `operand_index[d] = 0` if `dim(operand, d) = 1`, else
/// `result_index[broadcast_dimensions[d]]`. Only the expanded axes are known to be 1 in every instance;
/// any other operand axis of extent 1 has a result axis of extent 1, where both forms give 0.
pub fn broadcast_in_dim(x: &IterIn<'_>) -> Result<OpIteration, String> {
    let r = *x.results.first().ok_or("no result")?;
    let bd = ints(x, "broadcast_dimensions")?;
    let n = rank(x, 0)?;
    if bd.len() != n {
        return Err(format!("{} broadcast dimensions for rank {n}", bd.len()));
    }
    let op: Access = (0..n as u32)
        .map(|d| {
            if x.expanded.contains(&d) {
                Row::zero()
            } else {
                Row::dim(bd[d as usize])
            }
        })
        .collect();
    let mut operands = vec![Some(op)];
    operands.extend(x.operands.iter().skip(1).map(|_| None));
    Ok(OpIteration {
        dims: (0..r as u32).map(|a| par(0, a, "elem")).collect(),
        operands,
        results: vec![(0..r as u32).map(Row::dim).collect()],
    })
}

/// `#dot_general`: the result is `batch ++ lhs free ++ rhs free`; every contracting index pair is summed.
/// Dims: the result axes, then one reduction dim per contracting pair (ω axis: the lhs axis).
pub fn dot_general(x: &IterIn<'_>) -> Result<OpIteration, String> {
    let Some(Attr::DotDims(d)) = (x.lit)("dot_dimension_numbers") else {
        return Err("no dot_dimension_numbers".into());
    };
    let (nl, nr) = (rank(x, 0)?, rank(x, 1)?);
    let u = |v: &[i64]| v.iter().map(|&a| a as usize).collect::<Vec<_>>();
    let (lb, rb, lc, rc) = (
        u(&d.lhs_batching),
        u(&d.rhs_batching),
        u(&d.lhs_contracting),
        u(&d.rhs_contracting),
    );
    let lfree: Vec<usize> = (0..nl).filter(|a| !lb.contains(a) && !lc.contains(a)).collect();
    let rfree: Vec<usize> = (0..nr).filter(|a| !rb.contains(a) && !rc.contains(a)).collect();
    let nres = lb.len() + lfree.len() + rfree.len();
    let mut dims: Vec<Dim> = Vec::new();
    for a in 0..nres as u32 {
        let role = if (a as usize) < lb.len() {
            "batch"
        } else if (a as usize) < lb.len() + lfree.len() {
            "lhs_free"
        } else {
            "rhs_free"
        };
        dims.push(par(0, a, role));
    }
    for &a in &lc {
        dims.push(Dim {
            tag: Tag::Red,
            extent: Extent::Operand {
                port: 0,
                axis: a as u32,
            },
            omega: Some(a as u32),
            role: "contract".into(),
        });
    }
    let mut lhs = vec![Row::zero(); nl];
    let mut rhs = vec![Row::zero(); nr];
    for (i, (&a, &b)) in lb.iter().zip(&rb).enumerate() {
        lhs[a] = Row::dim(i as u32);
        rhs[b] = Row::dim(i as u32);
    }
    for (i, &a) in lfree.iter().enumerate() {
        lhs[a] = Row::dim((lb.len() + i) as u32);
    }
    for (i, &b) in rfree.iter().enumerate() {
        rhs[b] = Row::dim((lb.len() + lfree.len() + i) as u32);
    }
    for (i, (&a, &b)) in lc.iter().zip(&rc).enumerate() {
        lhs[a] = Row::dim((nres + i) as u32);
        rhs[b] = Row::dim((nres + i) as u32);
    }
    Ok(OpIteration {
        dims,
        operands: vec![Some(lhs), Some(rhs)],
        results: vec![(0..nres as u32).map(Row::dim).collect()],
    })
}

/// A size-attribute entry: `Attr` when the attribute is present, `None` when it takes its default.
fn attr_factor(x: &IterIn<'_>, name: &str, index: u32) -> Option<Factor> {
    (x.has)(name).then(|| Factor::Attr {
        name: name.into(),
        index,
    })
}

/// Where a convolution reads its window attributes: attributes, or (`dynamic_conv`) padding from
/// operand 2.
#[derive(Clone, Copy)]
enum ConvForm {
    Static,
    Dynamic,
}

/// `#convolution` (and `#dynamic_conv`, whose padding is operand 2). For output spatial index `o`,
/// kernel spatial index `r`: the padded, lhs-dilated input is read at `o·window_strides + r·rhs_dilation`,
/// which is the input element `(o·s + r·d − padding_low) / lhs_dilation` where it divides and lies in
/// range (padding otherwise); `window_reversal` reads the kernel at `R − 1 − r`. With
/// `feature_group_count` G > 1 the output feature `f = g·(F/G) + m` and the input feature
/// `g·C_g + c` (C_g = the kernel's input features). Dims: the result axes (the output feature split
/// into `g`, `m` when grouped), the kernel spatial axes, the kernel input feature (ω: kernel axes).
fn conv(x: &IterIn<'_>, form: ConvForm) -> Result<OpIteration, String> {
    let Some(Attr::ConvDims(d)) = (x.lit)("dimension_numbers") else {
        return Err("no dimension_numbers".into());
    };
    if x.batch_grouped {
        return Err("batch_group_count > 1 is not supported".into());
    }
    let (ni, nk) = (rank(x, 0)?, rank(x, 1)?);
    let nres = *x.results.first().ok_or("no result")?;
    let ns = d.input_spatial.len();
    let reversal: Vec<bool> = match (x.lit)("window_reversal") {
        Some(Attr::Bools(b)) => b.clone(),
        _ => vec![false; ns],
    };
    let of = d.output_feature as u32;
    let mut dims: Vec<Dim> = Vec::new();
    // Result axes, in result order; the output feature becomes `g` (here) and `m` (appended).
    let mut res_dim = vec![0u32; nres];
    for a in 0..nres as u32 {
        res_dim[a as usize] = dims.len() as u32;
        let role = if a == d.output_batch as u32 {
            "out_batch"
        } else if a == of {
            if x.feature_grouped { "group" } else { "out_feature" }
        } else {
            "out_spatial"
        };
        if a == of && x.feature_grouped {
            dims.push(Dim {
                tag: Tag::Par,
                extent: Extent::Factor(
                    attr_factor(x, "feature_group_count", 0).ok_or("no feature_group_count")?,
                ),
                omega: None,
                role: role.into(),
            });
        } else {
            dims.push(par(0, a, role));
        }
    }
    let m_dim = if x.feature_grouped {
        dims.push(Dim {
            tag: Tag::Par,
            extent: Extent::Quot {
                res: 0,
                axis: of,
                by: attr_factor(x, "feature_group_count", 0).ok_or("no feature_group_count")?,
            },
            omega: None,
            role: "group_feature".into(),
        });
        Some(dims.len() as u32 - 1)
    } else {
        None
    };
    let ks0 = dims.len() as u32;
    for &ka in &d.kernel_spatial {
        dims.push(Dim {
            tag: Tag::Red,
            extent: Extent::Operand {
                port: 1,
                axis: ka as u32,
            },
            omega: Some(ka as u32),
            role: "window".into(),
        });
    }
    let c_dim = dims.len() as u32;
    dims.push(Dim {
        tag: Tag::Red,
        extent: Extent::Operand {
            port: 1,
            axis: d.kernel_input_feature as u32,
        },
        omega: Some(d.kernel_input_feature as u32),
        role: "in_feature".into(),
    });
    let g_dim = res_dim[of as usize];
    // Feature rows.
    let feature = |inner: u32| -> Row {
        match m_dim {
            None => Row::dim(if inner == c_dim { c_dim } else { g_dim }),
            Some(m) => {
                let (unit, inner_dim) = if inner == c_dim { (c_dim, c_dim) } else { (m, m) };
                Row {
                    terms: vec![
                        term(1, vec![Factor::Extent { dim: unit }], Some(g_dim)),
                        term(1, Vec::new(), Some(inner_dim)),
                    ],
                    divisor: None,
                    padded: false,
                    data: false,
                }
            }
        }
    };
    let mut input = vec![Row::zero(); ni];
    let mut kernel = vec![Row::zero(); nk];
    let mut result: Access = (0..nres as u32).map(|a| Row::dim(res_dim[a as usize])).collect();
    input[d.input_batch as usize] = Row::dim(res_dim[d.output_batch as usize]);
    input[d.input_feature as usize] = feature(c_dim);
    kernel[d.kernel_input_feature as usize] = Row::dim(c_dim);
    kernel[d.kernel_output_feature as usize] = feature(g_dim);
    result[of as usize] = feature(g_dim);
    for i in 0..ns {
        let o = res_dim[d.output_spatial[i] as usize];
        let r = ks0 + i as u32;
        let mut terms = vec![term(
            1,
            attr_factor(x, "window_strides", i as u32).into_iter().collect(),
            Some(o),
        )];
        terms.push(term(
            1,
            attr_factor(x, "rhs_dilation", i as u32).into_iter().collect(),
            Some(r),
        ));
        let pad = match form {
            ConvForm::Static => attr_factor(x, "padding", 2 * i as u32),
            ConvForm::Dynamic => Some(Factor::Runtime {
                port: 2,
                index: 2 * i as u32,
            }),
        };
        if let Some(p) = pad {
            terms.push(term(-1, vec![p], None));
        }
        input[d.input_spatial[i] as usize] = Row {
            terms,
            divisor: attr_factor(x, "lhs_dilation", i as u32),
            padded: true,
            data: false,
        };
        kernel[d.kernel_spatial[i] as usize] = if reversal.get(i).copied().unwrap_or(false) {
            Row {
                terms: vec![
                    term(1, vec![Factor::Extent { dim: r }], None),
                    term(-1, Vec::new(), None),
                    term(-1, Vec::new(), Some(r)),
                ],
                divisor: None,
                padded: false,
                data: false,
            }
        } else {
            Row::dim(r)
        };
    }
    let mut operands = vec![Some(input), Some(kernel)];
    operands.extend(x.operands.iter().skip(2).map(|_| None));
    Ok(OpIteration {
        dims,
        operands,
        results: vec![result],
    })
}

pub fn convolution(x: &IterIn<'_>) -> Result<OpIteration, String> {
    conv(x, ConvForm::Static)
}

pub fn dynamic_conv(x: &IterIn<'_>) -> Result<OpIteration, String> {
    conv(x, ConvForm::Dynamic)
}

/// `#reduce(inputs…, init_values…)`: dims are the input axes; `dimensions` are reduced (ω: input
/// axes); the result index is the input index without the reduced axes; init values are rank 0.
pub fn reduce(x: &IterIn<'_>) -> Result<OpIteration, String> {
    let red = ints(x, "dimensions")?;
    let n = rank(x, 0)?;
    let k = x.results.len();
    let dims = (0..n as u32)
        .map(|a| {
            let r = red.contains(&a);
            Dim {
                tag: if r { Tag::Red } else { Tag::Par },
                extent: Extent::Operand { port: 0, axis: a },
                omega: r.then_some(a),
                role: if r { "reduced" } else { "kept" }.into(),
            }
        })
        .collect();
    let input: Access = (0..n as u32).map(Row::dim).collect();
    let out: Access = (0..n as u32).filter(|a| !red.contains(a)).map(Row::dim).collect();
    let mut operands = Vec::new();
    for p in 0..x.operands.len() {
        operands.push(x.operands[p].map(|_| if p < k { input.clone() } else { Vec::new() }));
    }
    Ok(OpIteration {
        dims,
        operands,
        results: vec![out; k],
    })
}

/// Where a windowed reduction reads its window: attributes, or the five shape operands after the inputs
/// and init values (`dynamic_reduce_window`).
#[derive(Clone, Copy)]
enum WinForm {
    Static,
    Dynamic,
}

/// `#reduce_window(inputs…, init_values…)`: for result index `o` and window index `w`, the padded,
/// base-dilated input is read at `o·window_strides + w·window_dilations`, i.e. the input element
/// `(o·s + w·wd − padding_low) / base_dilations` where it divides and lies in range. Dims: the result
/// axes, then one window dim per axis (ω: the window axes).
fn windowed(x: &IterIn<'_>, form: WinForm) -> Result<OpIteration, String> {
    let n = rank(x, 0)?;
    let k = x.results.len();
    let f = |name: &str, slot: u32, i: u32| -> Option<Factor> {
        match form {
            WinForm::Static => attr_factor(x, name, i),
            WinForm::Dynamic => Some(Factor::Runtime {
                port: 2 * k as u32 + slot,
                index: i,
            }),
        }
    };
    let mut dims: Vec<Dim> = (0..n as u32).map(|a| par(0, a, "out")).collect();
    for i in 0..n as u32 {
        dims.push(Dim {
            tag: Tag::Red,
            extent: Extent::Factor(f("window_dimensions", 0, i).ok_or("no window_dimensions")?),
            omega: Some(i),
            role: "window".into(),
        });
    }
    let input: Access = (0..n as u32)
        .map(|i| {
            let mut terms = vec![
                term(1, f("window_strides", 1, i).into_iter().collect(), Some(i)),
                term(
                    1,
                    f("window_dilations", 3, i).into_iter().collect(),
                    Some(n as u32 + i),
                ),
            ];
            if let Some(p) = f("padding", 4, 2 * i) {
                terms.push(term(-1, vec![p], None));
            }
            Row {
                terms,
                divisor: f("base_dilations", 2, i),
                padded: true,
                data: false,
            }
        })
        .collect();
    let mut operands = Vec::new();
    for p in 0..x.operands.len() {
        operands.push(x.operands[p].map(|_| if p < k { input.clone() } else { Vec::new() }));
    }
    Ok(OpIteration {
        dims,
        operands,
        results: vec![(0..n as u32).map(Row::dim).collect(); k],
    })
}

pub fn reduce_window(x: &IterIn<'_>) -> Result<OpIteration, String> {
    windowed(x, WinForm::Static)
}

pub fn dynamic_reduce_window(x: &IterIn<'_>) -> Result<OpIteration, String> {
    windowed(x, WinForm::Dynamic)
}

/// `#scatter(inputs…, scatter_indices, updates…)`: dims are the update axes; `update_scatter_dims` (the
/// axes not in `update_window_dims`) are combined into data-dependent targets (reduction dims, ω: update
/// axes), window axes are parallel. The result index is `full_start_index + full_window_index` (plus the
/// batching index): affine in the window and batching dims, plus a part read from `scatter_indices`.
pub fn scatter(x: &IterIn<'_>) -> Result<OpIteration, String> {
    let Some(Attr::ScatterDims(s)) = (x.lit)("scatter_dimension_numbers") else {
        return Err("no scatter_dimension_numbers".into());
    };
    let k = x.results.len();
    let nu = rank(x, k + 1)?;
    let ni = rank(x, k)?;
    let no = *x.results.first().ok_or("no result")?;
    let u = |v: &[i64]| v.iter().map(|&a| a as u32).collect::<Vec<_>>();
    let uw = u(&s.update_window_dims);
    let iw = u(&s.inserted_window_dims);
    let ib = u(&s.input_batching_dims);
    let sib = u(&s.scatter_indices_batching_dims);
    let sd2od = u(&s.scattered_dims_to_operand_dims);
    let ivd = s.index_vector_dim as u32;
    let dims: Vec<Dim> = (0..nu as u32)
        .map(|a| {
            let w = uw.contains(&a);
            Dim {
                tag: if w { Tag::Par } else { Tag::Red },
                extent: Extent::Operand {
                    port: (k + 1) as u32,
                    axis: a,
                },
                omega: Some(a),
                role: if w { "window" } else { "scatter" }.into(),
            }
        })
        .collect();
    let scatter_dims: Vec<u32> = (0..nu as u32).filter(|a| !uw.contains(a)).collect();
    // Result: window axes of the operand (not inserted, not batching) take the update window dims in
    // order; batching axes take the scatter dim matching their indices batching dim; scattered axes
    // add a start index read from the indices.
    let window_axes: Vec<u32> = (0..no as u32)
        .filter(|a| !iw.contains(a) && !ib.contains(a))
        .collect();
    // Indices axes other than `index_vector_dim`, in order, are the update scatter dims in order.
    let idx_axes: Vec<u32> = (0..ni as u32).filter(|&a| a != ivd).collect();
    let mut out = vec![Row::zero(); no];
    for (i, &a) in window_axes.iter().enumerate() {
        if let Some(&w) = uw.get(i) {
            out[a as usize] = Row::dim(w);
        }
    }
    for (j, &a) in ib.iter().enumerate() {
        let sb = *sib.get(j).ok_or("batching dims of unequal length")?;
        let pos = idx_axes
            .iter()
            .position(|&x| x == sb)
            .ok_or("bad indices batching dim")?;
        out[a as usize] = Row::dim(*scatter_dims.get(pos).ok_or("bad scatter dims")?);
    }
    for &a in &sd2od {
        out[a as usize].data = true;
    }
    let mut idx = vec![Row::zero(); ni];
    for (pos, &a) in idx_axes.iter().enumerate() {
        idx[a as usize] = Row::dim(*scatter_dims.get(pos).ok_or("bad scatter dims")?);
    }
    if (ivd as usize) < ni {
        idx[ivd as usize].data = true;
    }
    let upd: Access = (0..nu as u32).map(Row::dim).collect();
    let mut operands: Vec<Option<Access>> = Vec::new();
    for p in 0..x.operands.len() {
        operands.push(x.operands[p].map(|_| {
            if p < k {
                out.clone()
            } else if p == k {
                idx.clone()
            } else {
                upd.clone()
            }
        }));
    }
    Ok(OpIteration {
        dims,
        operands,
        results: vec![out; k],
    })
}
