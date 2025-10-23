"""Morphological gap-bridging to repair small boundary breaks (with logs)."""
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_closing
import logging

log = logging.getLogger(__name__)

def close_small_gaps(occ: NDArray[np.bool_], res_m: float, gap_close_m: float) -> NDArray[np.bool_]:
    if gap_close_m <= 0.0:
        return occ
    r = max(1, int(round(gap_close_m / res_m)))
    kernel = np.ones((2*r+1, 2*r+1), dtype=bool)
    out = binary_closing(occ, structure=kernel)
    log.debug(f"Gap closing with r={r} cells; changed={int((out^occ).sum())} cells")
    return out
