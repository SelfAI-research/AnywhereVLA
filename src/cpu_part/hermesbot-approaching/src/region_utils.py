"""Local flood-fill and free-facing boundary extraction with logs."""
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import logging

log = logging.getLogger(__name__)

def crop_roi_around(idx: Tuple[int,int], r_cells: int, shape: Tuple[int,int]) -> Tuple[int,int,int,int]:
    oi, oj = idx; h, w = shape
    r = max(1, int(r_cells))
    return max(0, oi-r), min(w, oi+r+1), max(0, oj-r), min(h, oj+r+1)

def flood_fill_local(mask_occ: NDArray[np.bool_], start: Tuple[int,int], r_cells: int) -> NDArray[np.bool_]:
    h, w = mask_occ.shape
    i0, i1, j0, j1 = crop_roi_around(start, r_cells, mask_occ.shape)
    sub = mask_occ[j0:j1, i0:i1]
    si, sj = start[0]-i0, start[1]-j0
    if si<0 or sj<0 or si>=sub.shape[1] or sj>=sub.shape[0] or not sub[sj, si]:
        log.debug("Flood fill start not in occupied ROI; returning empty region")
        return np.zeros_like(mask_occ, dtype=bool)
    vis = np.zeros_like(sub, dtype=bool)
    st = [(si, sj)]
    while st:
        i, j = st.pop()
        if vis[j, i] or not sub[j, i]:
            continue
        vis[j, i] = True
        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di==0 and dj==0: continue
                ni, nj = i+di, j+dj
                if 0<=ni<sub.shape[1] and 0<=nj<sub.shape[0] and not vis[nj, ni] and sub[nj, ni]:
                    st.append((ni, nj))
    out = np.zeros_like(mask_occ, dtype=bool)
    out[j0:j1, i0:i1] = vis
    log.debug(f"Flood region size: {vis.sum()} cells; ROI=({i0}:{i1},{j0}:{j1})")
    return out

def boundary_points_toward_free(region_occ: NDArray[np.bool_], free_mask: NDArray[np.bool_]) -> List[Tuple[int,int]]:
    h, w = region_occ.shape; out: List[Tuple[int,int]] = []
    for j in range(h):
        for i in range(w):
            if not region_occ[j, i]: continue
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0<=ni<w and 0<=nj<h and free_mask[nj, ni]:
                    out.append((i, j)); break
    log.debug(f"Boundary points toward free: {len(out)}")
    return out
