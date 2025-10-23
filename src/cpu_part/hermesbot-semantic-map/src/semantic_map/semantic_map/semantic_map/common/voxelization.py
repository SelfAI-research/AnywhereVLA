"""
# : Method                          Voxels   Mean [ms]   Std [ms]   Speedup   Mpts/s
# : ----------------------------------------------------------------------------
# : voxel_centroids_unique3d          75280    328.189     1.00x     1.25
# : voxel_centroids_sort_reduce       75280     60.657     5.41x     6.78
# : voxel_centroids_ravel_bincount    75280     41.706     7.87x     9.85
# : voxel_centroids_bitpack_reduce    75280     47.263     6.94x     8.70
# : Points: 413,920  |  voxel_size: 0.1 m 
"""
import numpy as np

def voxelize_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return voxel_centroids_ravel_bincount(points, voxel_size)

# ---------- Existing baselines ----------
def voxel_centroids_unique3d(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    uniq, inv, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inv, points.astype(np.float64))
    return (sums / counts[:, None]).astype(np.float32)

def voxel_centroids_sort_reduce(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    g = grid[order]
    p = points[order].astype(np.float64)
    change = np.any(np.diff(g, axis=0) != 0, axis=1)
    idx = np.concatenate(([True], change)).nonzero()[0]
    counts = np.diff(np.append(idx, g.shape[0]))
    sums = np.add.reduceat(p, idx, axis=0)
    return (sums / counts[:, None]).astype(np.float32)

def voxel_centroids_ravel_bincount(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    gmin = grid.min(axis=0)
    grid -= gmin
    span = grid.max(axis=0) + 1
    cap = np.iinfo(np.int64).max
    if int(span[0]) * int(span[1]) * int(span[2]) >= cap:
        return voxel_centroids_sort_reduce(points, voxel_size)
    lin = grid[:, 0] + span[0] * (grid[:, 1] + span[1] * grid[:, 2])
    lin = lin.astype(np.int64)
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    x = np.bincount(inv, weights=points[:, 0].astype(np.float64), minlength=uniq.shape[0])
    y = np.bincount(inv, weights=points[:, 1].astype(np.float64), minlength=uniq.shape[0])
    z = np.bincount(inv, weights=points[:, 2].astype(np.float64), minlength=uniq.shape[0])
    centers = np.stack((x, y, z), axis=1) / counts[:, None]
    return centers.astype(np.float32)

# ---------- New: bit-pack + sort + single reduce ----------

def _bitpack_keys(grid: np.ndarray) -> np.ndarray:
    # grid must be non-negative int32; spans must fit < 2**21 per axis
    gx = grid[:, 0].astype(np.uint64)
    gy = grid[:, 1].astype(np.uint64)
    gz = grid[:, 2].astype(np.uint64)
    return (gx << 42) | (gy << 21) | gz  # 21 bits each

def voxel_centroids_bitpack_reduce(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    inv_vs = 1.0 / vs

    # int32 grid via floor; keep memory traffic low
    g = np.floor(points * inv_vs).astype(np.int32)
    g -= g.min(axis=0)

    span = g.max(axis=0) + 1
    if np.any(span >= (1 << 21)):
        # fall back if span too large for 21-bit packing
        return voxel_centroids_sort_reduce(points, voxel_size)

    key = _bitpack_keys(g)
    order = np.argsort(key)
    key_s = key[order]
    p = points[order].astype(np.float64)

    # run-length encoding over sorted keys
    starts = np.empty(key_s.size, dtype=bool)
    starts[0] = True
    np.not_equal(key_s[1:], key_s[:-1], out=starts[1:])
    idx = np.flatnonzero(starts)
    counts = np.diff(np.append(idx, key_s.size))

    sums = np.add.reduceat(p, idx, axis=0)
    return (sums / counts[:, None]).astype(np.float32)