"""Normal estimators with concise logging."""
from typing import List, Optional, Tuple
import math
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
import logging

log = logging.getLogger(__name__)

def estimate_normal_frontier(free_mask: NDArray[np.bool_], boundary: List[Tuple[int,int]],
                             obj_idx: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    if not boundary: return None
    dt = distance_transform_edt(free_mask)
    h, w = free_mask.shape; oi, oj = obj_idx
    for i, j in sorted(boundary, key=lambda p: (p[0]-oi)**2 + (p[1]-oj)**2):
        best = None
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0<=ni<w and 0<=nj<h and free_mask[nj, ni]:
                s = dt[nj, ni]
                if best is None or s > best[0]: best = (s, ni, nj)
        if best is None: continue
        _, fx, fy = best
        gx = (dt[fy, min(fx+1,w-1)] - dt[fy, max(fx-1,0)]) * 0.5
        gy = (dt[min(fy+1,h-1), fx] - dt[max(fy-1,0), fx]) * 0.5
        m = math.hypot(gx, gy)
        if m == 0: continue
        n = (-gx/m, -gy/m)
        log.debug(f"Frontier normal at boundary({i},{j}) -> {n}")
        return n
    return None

def estimate_normal_distance(region_mask: NDArray[np.bool_], free_mask: NDArray[np.bool_],
                             obj_idx: Tuple[int,int], ox: float, oy: float, oyaw: float, res: float
                             ) -> Optional[Tuple[float,float]]:
    dt = distance_transform_edt(free_mask)
    h, w = region_mask.shape; oi, oj = obj_idx
    free_neighbors = []
    for j in range(h):
        for i in range(w):
            if not region_mask[j, i]: continue
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0<=ni<w and 0<=nj<h and free_mask[nj, ni]:
                    free_neighbors.append((ni, nj)); break
    if not free_neighbors: return None
    free_neighbors.sort(key=lambda p: (p[0]-oi)**2 + (p[1]-oj)**2)
    fx, fy = free_neighbors[0]
    gx = (dt[fy, min(fx+1,w-1)] - dt[fy, max(fx-1,0)]) * 0.5
    gy = (dt[min(fy+1,h-1), fx] - dt[max(fy-1,0), fx]) * 0.5
    m = math.hypot(gx, gy)
    if m == 0: return None
    n = (-gx/m, -gy/m)
    log.debug(f"Distance-gradient normal -> {n}")
    return n

def estimate_normal_pca(boundary: List[Tuple[int,int]], ox: float, oy: float, oyaw: float, res: float,
                        region_mask: NDArray[np.bool_], free_mask: NDArray[np.bool_],
                        obj_idx: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    if len(boundary) < 2: return None
    pts = np.array(boundary, dtype=float)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals)
    n = eigvecs[:, idx[0]]
    m = math.hypot(n[0], n[1])
    if m == 0: return None
    n = (-n[0]/m, -n[1]/m)
    log.debug(f"PCA normal -> {n}")
    return n

def estimate_normal_raycast(free_mask: NDArray[np.bool_], obj_idx: Tuple[int,int],
                            res: float, max_radius: float, step: float) -> Tuple[float,float]:
    oi, oj = obj_idx; best_len, best_ang = -1.0, 0.0
    for k in range(36):
        ang = k * (2.0*math.pi/36.0); length = 0.0
        while length < max_radius:
            x = oi + math.cos(ang) * (length/res)
            y = oj + math.sin(ang) * (length/res)
            ix, iy = int(round(x)), int(round(y))
            if iy<0 or ix<0 or iy>=free_mask.shape[0] or ix>=free_mask.shape[1]: break
            if not free_mask[iy, ix]: break
            length += step
        if length > best_len: best_len, best_ang = length, ang
    n = (-math.cos(best_ang), -math.sin(best_ang))
    log.debug(f"Raycast normal -> {n}")
    return n
