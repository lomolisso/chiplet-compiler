"""Host-side (excluded) index generation for the PTv3 graph, in numpy.

Everything PTv3 computes from integer grid coordinates with data-dependent
shapes lives here; the graph consumes the results as padded static-size index
inputs (see model.py). Each function mirrors one piece of the Pointcept
reference (Pointcept @ 1342eda) and was checked against the
reference's own output in `extra:index_generators`:

  serialization  pointcept/models/utils/structure.py Point.serialization
                 + utils/serialization/{default,z_order,hilbert}.py
  pooling        point_transformer_v3m1_base.py SerializedPooling.forward
                 (torch.unique / sort / argsort part)
  patches        SerializedAttention.get_padding_and_inverse (batch 1, the
                 enable_flash=False patch size min(n, patch_size))
  rulebook       spconv SubMConv3d indice pairs (gather/scatter pairs per
                 kernel offset, coord_in = coord_out + offset)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


# ----------------------------------------------------------------------------
# GridSample (pointcept/datasets/transform.py) grid coordinates
# ----------------------------------------------------------------------------

def grid_coord_of(coord: np.ndarray, grid_size: float) -> np.ndarray:
    """GridSample: floor(coord / grid_size) - min. Every fragment of the test-mode
    GridSample holds one point of every occupied voxel, so the min over the
    fragment equals the min over the whole cloud."""
    g = np.floor(np.asarray(coord) / np.array(grid_size)).astype(int)
    return g - g.min(0)


# ----------------------------------------------------------------------------
# Serialization codes
# ----------------------------------------------------------------------------

def z_order_encode(grid: np.ndarray, depth: int) -> np.ndarray:
    """z_order.xyz2key: bit i of (x, y, z) -> bits (3i+2, 3i+1, 3i)."""
    x, y, z = (grid[:, i].astype(np.int64) for i in range(3))
    key = np.zeros_like(x)
    for i in range(depth):
        key |= (((x >> i) & 1) << (3 * i + 2)) | (((y >> i) & 1) << (3 * i + 1)) \
            | (((z >> i) & 1) << (3 * i))
    return key


def hilbert_encode(locs: np.ndarray, num_dims: int, num_bits: int) -> np.ndarray:
    """hilbert.encode (Skilling's transpose algorithm, numpy-hilbert-curve)."""
    n = locs.shape[0]
    locs = np.ascontiguousarray(locs.astype("<i8"))
    # big-endian bits of every coordinate, truncated to num_bits
    by = locs.view(np.uint8).reshape(n, num_dims, 8)[..., ::-1]
    gray = np.unpackbits(by, axis=-1).reshape(n, num_dims, 64)[..., -num_bits:].copy()
    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit].copy()
            gray[:, 0, bit + 1:] ^= mask[:, None]
            to_flip = (1 - mask[:, None]) & (gray[:, 0, bit + 1:] ^ gray[:, dim, bit + 1:])
            gray[:, dim, bit + 1:] ^= to_flip
            gray[:, 0, bit + 1:] ^= to_flip
    gray = gray.swapaxes(1, 2).reshape(n, num_bits * num_dims)
    binary = np.bitwise_xor.accumulate(gray, axis=1).astype(np.int64)   # gray2binary
    nb = binary.shape[1]
    weights = np.left_shift(np.int64(1), np.arange(nb - 1, -1, -1, dtype=np.int64))
    return (binary * weights).sum(1)


def encode(grid: np.ndarray, depth: int, order: str) -> np.ndarray:
    """serialization/default.encode with batch = 0 (batch << 3*depth is 0)."""
    if order == "z":
        return z_order_encode(grid, depth)
    if order == "z-trans":
        return z_order_encode(grid[:, [1, 0, 2]], depth)
    if order == "hilbert":
        return hilbert_encode(grid, 3, depth)
    if order == "hilbert-trans":
        return hilbert_encode(grid[:, [1, 0, 2]], 3, depth)
    raise ValueError(order)


def _order_inverse(code: np.ndarray):
    order = np.argsort(code, axis=1, kind="stable")     # codes are unique per order
    inverse = np.empty_like(order)
    np.put_along_axis(inverse, order, np.arange(code.shape[1])[None].repeat(len(code), 0), 1)
    return order, inverse


@dataclass
class Stage:
    """One point set of the U-Net (the reference's `Point` at that level)."""
    grid: np.ndarray            # [n, 3] int64 grid coordinates
    depth: int                  # serialized_depth
    code: np.ndarray            # [k, n] int64 (shuffled order slots)
    order: np.ndarray           # [k, n]
    inverse: np.ndarray         # [k, n]
    cluster: np.ndarray | None = None     # pooling_inverse: parent row -> this row
    indices: np.ndarray | None = None     # parent rows sorted by cluster

    @property
    def n(self) -> int:
        return self.grid.shape[0]


def serialize(grid: np.ndarray, perm, orders=ORDERS) -> Stage:
    """Point.serialization(order, shuffle_orders=True) with the drawn randperm."""
    grid = np.asarray(grid, np.int64)
    depth = int(grid.max() + 1).bit_length()
    assert depth <= 16
    code = np.stack([encode(grid, depth, o) for o in orders])
    order, inverse = _order_inverse(code)
    perm = np.asarray(perm)
    return Stage(grid, depth, code[perm], order[perm], inverse[perm])


def pool(st: Stage, perm, stride: int = 2) -> Stage:
    """SerializedPooling.forward index part (grid pooling by code >> 3)."""
    pooling_depth = (math.ceil(stride) - 1).bit_length()
    if pooling_depth > st.depth:
        pooling_depth = 0
    code = st.code >> (pooling_depth * 3)
    _, cluster, counts = np.unique(code[0], return_inverse=True, return_counts=True)
    cluster = cluster.reshape(-1)
    indices = np.argsort(cluster, kind="stable")
    idx_ptr = np.concatenate([[0], np.cumsum(counts)])
    head = indices[idx_ptr[:-1]]
    code = code[:, head]
    order, inverse = _order_inverse(code)
    perm = np.asarray(perm)
    return Stage(st.grid[head] >> pooling_depth, st.depth - pooling_depth, code[perm],
                 order[perm], inverse[perm], cluster=cluster, indices=indices)


def stages(grid: np.ndarray, perms, n_pool: int, orders=ORDERS) -> list[Stage]:
    out = [serialize(grid, perms[0], orders)]
    for s in range(n_pool):
        out.append(pool(out[-1], perms[s + 1]))
    return out


# ----------------------------------------------------------------------------
# Serialized attention patches (get_padding_and_inverse, batch 1)
# ----------------------------------------------------------------------------

def patches(n: int, patch_size: int):
    """(pad, unpad, K) for one cloud of n points. enable_flash=False sets the
    patch size to min(n, patch_size); the last patch is filled with the points
    at the same positions of the previous patch."""
    K = min(n, patch_size)
    bp = (n + K - 1) // K * K if n > K else n
    pad = np.arange(bp)
    unpad = np.arange(n)
    if n != bp:
        r = n % K
        pad[bp - K + r: bp] = pad[bp - 2 * K + r: bp - K]
    return pad, unpad, K


# ----------------------------------------------------------------------------
# SubMConv3d rulebook (spconv indice pairs)
# ----------------------------------------------------------------------------

def offsets(k: int) -> np.ndarray:
    """Kernel offsets in spconv's flat kernel order (x major, z minor)."""
    r = np.arange(k) - k // 2
    return np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)


def rulebook(grid: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per kernel offset (center excluded): (in rows, out rows) with
    grid[in] = grid[out] + offset; out rows ascending."""
    grid = np.asarray(grid, np.int64)
    m = k // 2
    ext = grid.max(0) + 2 * m + 1
    g = grid + m
    key = (g[:, 0] * ext[1] + g[:, 1]) * ext[2] + g[:, 2]
    srt = np.argsort(key)
    skey = key[srt]
    out = []
    ofs = offsets(k)
    for i, o in enumerate(ofs):
        if i == len(ofs) // 2:
            continue
        t = g + o
        tk = (t[:, 0] * ext[1] + t[:, 1]) * ext[2] + t[:, 2]
        pos = np.clip(np.searchsorted(skey, tk), 0, len(skey) - 1)
        hit = skey[pos] == tk
        rows = np.nonzero(hit)[0]
        out.append((srt[pos[rows]], rows))
    return out


# ----------------------------------------------------------------------------
# Example clouds (no nuScenes data is used)
# ----------------------------------------------------------------------------

HDL32E_ELEVATION = np.deg2rad(np.linspace(-30.67, 10.67, 32))   # nuScenes LIDAR_TOP beams
LIDAR_HEIGHT = 1.84                                             # m above ground


def simulate_sweep(rng: np.random.Generator, n_azimuth: int = 1085, max_range: float = 70.0):
    """Ray-cast a synthetic HDL-32E sweep (32 beams x n_azimuth firings) against
    a ground plane, parked cars, building blocks and poles. Returns
    (coord [n, 3] f32 in the lidar frame, strength [n, 1] f32 in [0, 1])."""
    az = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    el, az = np.meshgrid(HDL32E_ELEVATION, az, indexing="ij")
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[:, 2] < 0, -LIDAR_HEIGHT / d[:, 2], np.inf)
    t = np.minimum(t, tg)
    boxes = []
    for _ in range(40):                                     # cars: 4.5 x 1.9 x 1.6 m
        r, a = rng.uniform(4, 40), rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(a), r * np.sin(a)])
        h = np.array([2.25, 0.95]) if rng.random() < 0.5 else np.array([0.95, 2.25])
        boxes.append((*(c - h), -LIDAR_HEIGHT, *(c + h), -LIDAR_HEIGHT + 1.6))
    for _ in range(12):                                     # buildings
        r, a = rng.uniform(20, 55), rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(a), r * np.sin(a)])
        h = rng.uniform(3, 12, 2)
        boxes.append((*(c - h), -LIDAR_HEIGHT, *(c + h), rng.uniform(4, 15)))
    for _ in range(30):                                     # poles / trunks: 0.3 m squares
        r, a = rng.uniform(3, 30), rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(a), r * np.sin(a)])
        boxes.append((*(c - 0.15), -LIDAR_HEIGHT, *(c + 0.15), 3.0))
    b = np.asarray(boxes)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0 = (b[None, :, 0:3] - 0) * inv[:, None, :]
        t1 = (b[None, :, 3:6] - 0) * inv[:, None, :]
    tn = np.nanmax(np.minimum(t0, t1), -1)
    tf = np.nanmin(np.maximum(t0, t1), -1)
    hit = (tf >= np.maximum(tn, 0)) & (tn > 0)
    tb = np.where(hit, tn, np.inf).min(1)
    t = np.minimum(t, tb)
    ok = t < max_range
    t = t[ok] + rng.normal(0, 0.02, ok.sum())
    coord = (d[ok] * t[:, None]).astype(np.float32)
    strength = (rng.integers(0, 256, (len(coord), 1)) / 255).astype(np.float32)
    return coord, strength


def grid_sample(coord, strength, grid_size: float, rng: np.random.Generator | None = None):
    """One point per occupied voxel (GridSample; train mode picks a random point
    per voxel, test-mode fragment i picks the i-th): returns the representatives."""
    g = np.floor(coord / np.array(grid_size)).astype(int)
    g -= g.min(0)
    _, first, inv, cnt = np.unique(g, axis=0, return_index=True, return_inverse=True,
                                   return_counts=True)
    if rng is not None:
        order = np.argsort(inv.reshape(-1), kind="stable")
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        first = order[start + rng.integers(0, cnt.max(), cnt.size) % cnt]
    return coord[first], strength[first], int(cnt.max())


def saturated_cloud(n: int, rng: np.random.Generator, grid_size: float = 0.05):
    """Worst case of every static bound: n points that never merge in the 4
    poolings (at most one point per 16^3 cell), so every stage has exactly n
    points. Points come in 2x2x2 groups straddling 16-cell corners, so every
    point has SubM neighbours at every stage; (0, 0, 0) anchors the grid min."""
    pts = [np.zeros((1, 3), np.int64)]
    cube = np.stack(np.meshgrid([-1, 0], [-1, 0], [-1, 0], indexing="ij"), -1).reshape(-1, 3)
    need = n - 1
    corners = []
    side = int(np.ceil((need / 8) ** (1 / 3))) + 1
    for i in range(side):
        for j in range(side):
            for k in range(side):
                corners.append((2 + 2 * i, 2 + 2 * j, 2 + 2 * k))
    corners = np.asarray(corners)[rng.permutation(len(corners))]
    for c in corners:
        if need <= 0:
            break
        take = cube[:min(8, need)]
        pts.append(16 * c[None] + take)
        need -= len(take)
    grid = np.concatenate(pts)[:n]
    grid = grid[rng.permutation(n)]
    coord = ((grid + rng.uniform(0.1, 0.9, grid.shape)) * grid_size).astype(np.float32)
    strength = rng.random((n, 1)).astype(np.float32)
    assert np.array_equal(grid_coord_of(coord, grid_size), grid)
    return coord, strength
