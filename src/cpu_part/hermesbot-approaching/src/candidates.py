"""Candidate generation and clearance checks with light logging."""
from typing import Generator, Tuple
import math
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
import logging

try:
    from .coord_utils import world_to_map
except ImportError:  # pragma: no cover
    from coord_utils import world_to_map  # type: ignore

log = logging.getLogger(__name__)

def find_edge_distance(ox: float, oy: float, nx: float, ny: float, free_mask: NDArray[np.bool_],
                       mx: float, my: float, myaw: float, res: float, max_search: float) -> float:
    step = res * 0.5
    w, h = free_mask.shape[1], free_mask.shape[0]
    for k in range(int(max_search / step)):
        tx, ty = ox - nx*k*step, oy - ny*k*step
        ti, tj = world_to_map(tx, ty, mx, my, myaw, res)
        if ti<0 or tj<0 or ti>=w or tj>=h: break
        if free_mask[tj, ti]:
            d = k * step
            log.debug(f"Edge distance found: {d:.3f} m")
            return d
    log.debug("Edge distance not found within max_search")
    return -1.0

def generate_candidates(ex: float, ey: float, nx: float, ny: float, base_standoff: float,
                        max_search: float, sample_step: float, num_angles: int, num_offsets: int,
                        offset_step: float, max_slide: float, slide_step: float
                        ) -> Generator[Tuple[float,float,float], None, None]:
    fx, fy = ny, -nx
    total = 0
    for ai in range(0, num_angles+1):
        deg = ai * 5.0
        angles = [0.0] if ai==0 else [math.radians(deg), math.radians(-deg)]
        for a in angles:
            ca, sa = math.cos(a), math.sin(a)
            nxr, nyr = ca*nx - sa*ny, sa*nx + ca*ny
            oxr, oyr = -nxr, -nyr
            fxr, fyr = ca*fx - sa*fy, sa*fx + ca*fy
            for oi in range(0, num_offsets+1):
                offs = [0.0] if oi==0 else [oi*offset_step, -oi*offset_step]
                for off in offs:
                    for si in range(0, int(max_slide/slide_step)+1):
                        slides = [0.0] if si==0 else [si*slide_step, -si*slide_step]
                        for sl in slides:
                            standoff = base_standoff + off
                            cx = ex + oxr*standoff + fxr*sl
                            cy = ey + oyr*standoff + fyr*sl
                            yaw = math.atan2(nyr, nxr)
                            total += 1
                            yield (cx, cy, yaw)
    log.debug(f"Generated {total} candidates (pre-cap)")

def clearance_ok(x: float, y: float, res: float, free_mask: NDArray[np.bool_], dist_map: NDArray[np.float64],
                 mx: float, my: float, myaw: float, min_clear: float) -> bool:
    w, h = free_mask.shape[1], free_mask.shape[0]
    i, j = world_to_map(x, y, mx, my, myaw, res)
    if i<0 or j<0 or i>=w or j>=h: return False
    ok = (dist_map[j, i] * res) >= min_clear
    return ok
