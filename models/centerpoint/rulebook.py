"""Host-side sparse-conv rulebook for the static CenterPoint graph (numpy only).

This is the data-dependent part of spconv that cannot live in a static graph:
voxel coordinates per resolution level and, per sparse conv layer and kernel
offset, the (input, output) voxel pairs. It mirrors spconv 2.x's CPU generators
(`generate_subm_conv_inds_cpu`, `generate_conv_inds_cpu`):

  SubMConv3d (stride 1, pad k//2): output set = input set; for output o and
      offset k the input is o - pad + k (if active). The centre offset pairs
      every voxel with itself; spconv handles it as one dense GEMM and lists no
      pairs for it, so neither do we.
  SparseConv3d (stride s, pad p): input i and offset k produce output
      o = (i + p - k) / s if divisible and inside the output grid; the output
      set is the union of all such o.

Offsets are enumerated row-major over (kz, ky, kx) = spconv's weight layout
[Cout, kz, ky, kx, Cin]. Coordinates are (z, y, x), batch 1.

Static layout (sizes are the worst-case bounds of `net.level_bounds`):
  levels    l = 1..5, n_l valid voxels first, then padding rows.
  subm_l    int32 [kv - 1, N_l]: input row of each output row per non-centre
            offset, N_l (= a zero dump row) if absent or padding.
  down_l    strided layer l -> l+1. Pairs per offset are at most
            min(N_l, N_{l+1}) (one output per (input, offset), one input per
            (output, offset)); the buffer is laid out along the smaller side:
              "in"  int32 [kv, N_l]:     output row of each input row (dump N_{l+1})
              "out" int32 [kv, N_{l+1}]: input row of each output row (dump N_l)
  bev       int32 [N_5, 3]: level-5 coordinates (padding rows = (D_5, 0, 0),
            i.e. one past the dense map, a dump row).
"""

from __future__ import annotations

import itertools

import numpy as np

from models.centerpoint.net import Config, capacities, layouts, level_shapes


def _keys(c: np.ndarray, shape) -> np.ndarray:
    c = c.astype(np.int64)
    return (c[:, 0] * shape[1] + c[:, 1]) * shape[2] + c[:, 2]


def _decode(k: np.ndarray, shape) -> np.ndarray:
    z, r = np.divmod(k, shape[1] * shape[2])
    y, x = np.divmod(r, shape[2])
    return np.stack([z, y, x], 1).astype(np.int64)


def offsets(ksize) -> np.ndarray:
    return np.array(list(itertools.product(*[range(k) for k in ksize])), np.int64)


def _lookup(skeys, order, q, shape, dump):
    inb = np.all((q >= 0) & (q < np.asarray(shape)), axis=1)
    qk = _keys(np.where(inb[:, None], q, 0), shape)
    pos = np.minimum(np.searchsorted(skeys, qk), len(skeys) - 1)
    found = inb & (skeys[pos] == qk)
    return np.where(found, order[pos], dump)


def subm_pairs(c: np.ndarray, shape, ksize=(3, 3, 3)):
    """[(offset index, in rows, out rows)] for the non-centre offsets."""
    k = _keys(c, shape)
    order = np.argsort(k, kind="stable")
    sk = k[order]
    pad = np.asarray(ksize) // 2
    offs = offsets(ksize)
    centre = len(offs) // 2
    out = []
    for j, off in enumerate(offs):
        if j == centre:
            continue
        src = _lookup(sk, order, c - pad + off, shape, -1)
        o = np.nonzero(src >= 0)[0]
        out.append((j, src[o], o))
    return out


def down_pairs(c: np.ndarray, in_shape, out_shape, ksize, stride, padding):
    """(output coords sorted by linear index, [(offset index, in rows, out rows)])."""
    offs = offsets(ksize)
    stride, padding = np.asarray(stride), np.asarray(padding)
    cand = []
    for off in offs:
        num = c + padding - off
        ok = np.all((num >= 0) & (num % stride == 0), axis=1)
        o = num // stride
        ok &= np.all(o < np.asarray(out_shape), axis=1)
        cand.append((ok, o))
    allk = np.concatenate([_keys(o[ok], out_shape) for ok, o in cand])
    uk = np.unique(allk)
    pairs = []
    for j, (ok, o) in enumerate(cand):
        i = np.nonzero(ok)[0]
        pairs.append((j, i, np.searchsorted(uk, _keys(o[i], out_shape))))
    return _decode(uk, out_shape), pairs


def build(cfg: Config, coords1: np.ndarray) -> dict:
    """coords1: int [n1, 3] (z, y, x) valid level-1 voxels in input order.

    Returns {"levels": [coords_l], "pairs": {key: [(k, in, out)]},
    "inputs": {name: int32 array}} (the padded graph inputs)."""
    shapes, bounds, lays = level_shapes(cfg), capacities(cfg), layouts(cfg)
    levels = [np.asarray(coords1, np.int64)]
    pairs, inputs = {}, {}
    for l, (key, ks, st, pd) in enumerate(cfg.down):
        oc, pr = down_pairs(levels[l], shapes[l], shapes[l + 1], ks, st, pd)
        pairs[key] = pr
        levels.append(oc)
    for l, c in enumerate(levels):
        if len(c) > bounds[l]:
            raise ValueError(f"level {l + 1}: {len(c)} voxels exceed the static bound {bounds[l]}")
    kv3 = 27
    for l, key in enumerate(cfg.subm_keys):
        pr = subm_pairs(levels[l], shapes[l])
        pairs[key] = pr
        a = np.full((kv3 - 1, bounds[l]), bounds[l], np.int32)
        for r, (_, i, o) in enumerate(pr):
            a[r, o] = i
        inputs[key] = a
    for l, (key, ks, st, pd) in enumerate(cfg.down):
        kv = int(np.prod(ks))
        n_in, n_out = bounds[l], bounds[l + 1]
        if lays[l] == "in":
            a = np.full((kv, n_in), n_out, np.int32)
            for j, i, o in pairs[key]:
                a[j, i] = o
        else:
            a = np.full((kv, n_out), n_in, np.int32)
            for j, i, o in pairs[key]:
                a[j, o] = i
        inputs[key] = a
    d5 = shapes[-1][0]
    bev = np.zeros((bounds[-1], 3), np.int32)
    bev[:, 0] = d5
    bev[:len(levels[-1])] = levels[-1]
    inputs["bev_coords"] = bev
    return {"levels": levels, "pairs": pairs, "inputs": inputs}


def real_pair_counts(rb: dict) -> dict:
    """Unpadded pairs per offset (what spconv's per-offset GEMMs actually see)."""
    return {k: [len(i) for _, i, _ in v] for k, v in rb["pairs"].items()}
