import math
from dataclasses import dataclass
from array import array

from skimage.draw import line as _sk_line
import numpy as np

from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header


def _logit(p: float) -> float:
    """Numerically safe logit."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass
class Octo2DConfig:
    """Configuration for 2D occupancy mapping with log-odds updates."""
    resolution: float = 0.05
    prob_hit: float = 0.7
    prob_miss: float = 0.4
    clamp_min: float = 0.12
    clamp_max: float = 0.97
    occ_threshold: float = 0.5
    # Initial canvas sizes in X/Y (meters)
    init_x_size: float = 0.0
    init_y_size: float = 0.0


class Octomap2D:
    """
    Lightweight 2D occupancy grid with log-odds + fast, batched updates. 
    - Grid dynamically expands to fit new observations.

    Exports both:
      - Binary grid: (-1 unknown, 0 free, 100 occupied)
      - Probability grid: (-1 unknown, else 0..100)
    """

    UNKNOWN_EPS = 1e-5  # treat |log-odds| <= eps as unknown

    def __init__(self, cfg: Octo2DConfig):
        self.cfg = cfg
        self.res = cfg.resolution
        self._l_occ = _logit(cfg.prob_hit)     # > 0
        self._l_free = _logit(cfg.prob_miss)   # < 0
        self._lmin = _logit(cfg.clamp_min)
        self._lmax = _logit(cfg.clamp_max)
        self._locc_thresh = _logit(cfg.occ_threshold)

        # initialize canvas centered at origin with minimum requested size
        self.origin_x = -0.5 * max(self.res, cfg.init_x_size)
        self.origin_y = -0.5 * max(self.res, cfg.init_y_size)
        self.width = max(1, int(round(max(self.res, cfg.init_x_size) / self.res)))
        self.height = max(1, int(round(max(self.res, cfg.init_y_size) / self.res)))

        self._log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    # ---------- geometry helpers ----------

    def world_to_map(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (x,y) to integer grid indices (i,j)."""
        i = int(math.floor((x - self.origin_x) / self.res))
        j = int(math.floor((y - self.origin_y) / self.res))
        return i, j

    def map_to_world(self, i: int, j: int) -> tuple[float, float]:
        """Convert grid indices (i,j) to world (x,y) at cell centers."""
        return (self.origin_x + (i + 0.5) * self.res,
                self.origin_y + (j + 0.5) * self.res)

    def _ensure_bounds_for_points(self, points_xy: np.ndarray, origin_xy: tuple[float, float]) -> None:
        """Grow the grid so that all points and the origin fit inside."""
        if points_xy.size == 0:
            return

        xs = np.concatenate([points_xy[:, 0], np.array([origin_xy[0]], dtype=np.float32)])
        ys = np.concatenate([points_xy[:, 1], np.array([origin_xy[1]], dtype=np.float32)])

        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        # enforce minimum canvas size around origin
        half_x = max(self.cfg.init_x_size * 0.5, 0.0)
        half_y = max(self.cfg.init_y_size * 0.5, 0.0)
        min_x = min(min_x, -half_x)
        max_x = max(max_x, half_x)
        min_y = min(min_y, -half_y)
        max_y = max(max_y, half_y)

        imin, jmin = self.world_to_map(min_x, min_y)
        imax, jmax = self.world_to_map(max_x, max_y)

        need_left = max(0, -imin)
        need_bottom = max(0, -jmin)
        need_right = max(0, imax - (self.width - 1))
        need_top = max(0, jmax - (self.height - 1))
        if (need_left | need_bottom | need_right | need_top) == 0:
            return

        new_w = self.width + need_left + need_right
        new_h = self.height + need_bottom + need_top

        new_lo = np.zeros((new_h, new_w), dtype=np.float32)
        y0 = need_bottom
        x0 = need_left
        new_lo[y0:y0 + self.height, x0:x0 + self.width] = self._log_odds

        self._log_odds = new_lo
        self.origin_x -= need_left * self.res
        self.origin_y -= need_bottom * self.res
        self.width = new_w
        self.height = new_h

    def _line_cells(self, i0: int, j0: int, i1: int, j1: int) -> tuple[np.ndarray, np.ndarray]:
        """Bresenham. Returns arrays of (ii, jj) excluding endpoint."""
        # skimage returns rr (rows=j), cc (cols=i), including endpoint
        rr, cc = _sk_line(j0, i0, j1, i1)
        if rr.size > 0:
            rr = rr[:-1]; cc = cc[:-1]  # exclude endpoint
        return cc.astype(np.int32, copy=False), rr.astype(np.int32, copy=False)

    def integrate_scan(
        self,
        points_xy: np.ndarray,
        origin_xy: tuple[float, float],
        raycast_range: tuple[float, float] | None = None,
        do_raycast: bool = True,
        angle_bin_deg: float | None = 0.25,
        use_map_occlusion: bool = True,
        clear_occluder: bool = True,
    ) -> None:
        """
        Vectorized endpoint updates + batched free-space carving.
        Uses np.add.at to accumulate repeated hits efficiently.

        Raycast parameters:
          - angle_bin_deg: quantize bearings and keep nearest point per bearing (stop at first hit in scan).
          - use_map_occlusion: if True, stop a ray early if it meets an already-occupied cell in the map
            (do not clear beyond it; skip endpoint occ update in that case).
          - clear_occluder: if True, when the first occupied cell on the path is found, also apply a free update
            to that cell (so dynamic obstacles can be cleared), then stop.
          - raycast_range: (min_m, max_m). If max < 0 -> infinite; if min < 0 -> 0.
            Applies ONLY to free-space carving (self._l_free), not to occupied endpoints.
        """
        if points_xy.ndim != 2 or (points_xy.shape[1] not in (2, 3)):
            raise ValueError("points_xy must be (N,2) or (N,3)")

        pts = points_xy[:, :2].astype(np.float32, copy=False)

        if pts.size == 0:
            return

        # Keep only the closest point per bearing -> stop rays at first hit in the scan
        if angle_bin_deg is not None and angle_bin_deg > 0.0:
            v = pts - np.array(origin_xy, dtype=np.float32)
            d = np.linalg.norm(v, axis=1)
            valid = d > 0.0
            if not np.any(valid):
                return
            v = v[valid]; d = d[valid]
            theta = np.arctan2(v[:, 1], v[:, 0])
            bin_size = math.radians(float(angle_bin_deg))
            bins = np.rint(theta / bin_size).astype(np.int32, copy=False)
            order = np.lexsort((d, bins))
            bins_sorted = bins[order]
            _, first_idx = np.unique(bins_sorted, return_index=True)
            sel = order[first_idx]
            pts = (v[sel] + np.array(origin_xy, dtype=np.float32))

        self._ensure_bounds_for_points(pts, origin_xy)

        i0, j0 = self.world_to_map(origin_xy[0], origin_xy[1])
        i0 = int(np.clip(i0, 0, self.width - 1))
        j0 = int(np.clip(j0, 0, self.height - 1))

        # Endpoints → grid indices (vectorized)
        i1 = ((pts[:, 0] - self.origin_x) / self.res).astype(np.int32, copy=False)
        j1 = ((pts[:, 1] - self.origin_y) / self.res).astype(np.int32, copy=False)
        inb = (i1 >= 0) & (j1 >= 0) & (i1 < self.width) & (j1 < self.height)
        if not np.any(inb):
            return
        i1 = i1[inb]; j1 = j1[inb]

        # Take only unique
        keys = np.ravel_multi_index((j1, i1), (self.height, self.width))
        _, uniq_idx = np.unique(keys, return_index=True)
        i1 = i1[uniq_idx]; j1 = j1[uniq_idx]

        # Occlusion-aware free-space carving + endpoint accumulation
        free_ii, free_jj = [], []
        occ_i, occ_j = [], []
        _line = self._line_cells
        th = self._locc_thresh

        # ---------- range mask setup (applies only to FREE cells) ----------
        use_range = raycast_range is not None
        if use_range:
            rmin, rmax = float(raycast_range[0]), float(raycast_range[1])
            if rmin < 0.0: rmin = 0.0
            if rmax < 0.0: rmax = float("inf")
            rmin2 = rmin * rmin
            rmax2 = (rmax * rmax) if np.isfinite(rmax) else float("inf")
            res2 = self.res * self.res

            def _mask_by_range(ii_arr: np.ndarray, jj_arr: np.ndarray) -> np.ndarray:
                # distance between cell centers: (ii - i0, jj - j0) * res
                dx = (ii_arr - i0).astype(np.int32, copy=False)
                dy = (jj_arr - j0).astype(np.int32, copy=False)
                d2 = (dx * dx + dy * dy).astype(np.float32, copy=False) * res2
                return (d2 >= rmin2) & (d2 <= rmax2)

        # ---------- occlusion-aware carving ----------
        if do_raycast and i1.size:
            for k in range(i1.size):
                ii, jj = _line(i0, j0, int(i1[k]), int(j1[k]))  # endpoint excluded
                if ii.size and use_map_occlusion:
                    lo_path = self._log_odds[jj, ii]
                    mask_occ = lo_path > th
                    if mask_occ.any():
                        first = int(np.flatnonzero(mask_occ)[0])
                        upto = first + 1 if clear_occluder else first
                        if upto > 0:
                            sii = ii[:upto]; sjj = jj[:upto]
                            if use_range:
                                rm = _mask_by_range(sii, sjj)
                                if rm.any():
                                    free_ii.append(sii[rm]); free_jj.append(sjj[rm])
                            else:
                                free_ii.append(sii); free_jj.append(sjj)
                        # skip endpoint occ update (occluded)
                        continue

                # No occluder → all cells free, and endpoint is the first hit
                if ii.size:
                    if use_range:
                        rm = _mask_by_range(ii, jj)
                        if rm.any():
                            free_ii.append(ii[rm]); free_jj.append(jj[rm])
                    else:
                        free_ii.append(ii); free_jj.append(jj)

                occ_i.append(int(i1[k])); occ_j.append(int(j1[k]))
        else:
            # no carving requested → just mark endpoints occupied
            occ_i = i1.tolist(); occ_j = j1.tolist()

        if free_ii:
            free_ii = np.concatenate(free_ii)
            free_jj = np.concatenate(free_jj)
            np.add.at(self._log_odds, (free_jj, free_ii), self._l_free)

        if occ_i:
            np.add.at(self._log_odds, (np.asarray(occ_j, dtype=np.int32),
                                       np.asarray(occ_i, dtype=np.int32)), self._l_occ)

        # Single global clamp
        np.clip(self._log_odds, self._lmin, self._lmax, out=self._log_odds)

    # ---- One-shot exporter (binary + probability) ----
    def build_grids(self, header: Header, z_level: float = 0.0) -> tuple[OccupancyGrid, OccupancyGrid]:
        """
        Build (binary_grid, probability_grid) in one pass.
        Binary:  -1 unknown, 0 free, 100 occupied
        Prob:    -1 unknown, else round(p*100)
        """
        lo = self._log_odds
        th = self._locc_thresh
        unknown = np.abs(lo) <= self.UNKNOWN_EPS

        # probability once
        prob = 1.0 / (1.0 + np.exp(-lo, dtype=np.float32))
        prob100 = np.rint(prob * 100.0, dtype=np.float32).astype(np.int8, copy=False)
        prob100[unknown] = -1

        # binary classification with strict thresholds; leave band as unknown
        bin_grid = np.full(lo.shape, -1, dtype=np.int8)
        np.putmask(bin_grid, (~unknown) & (lo >  th), 100)
        np.putmask(bin_grid, (~unknown) & (lo < -th),   0)

        # common info
        info = MapMetaData()
        info.resolution = float(self.res)
        info.width = int(self.width)
        info.height = int(self.height)
        info.origin.position.x = float(self.origin_x)
        info.origin.position.y = float(self.origin_y)
        info.origin.position.z = float(z_level)

        # ---- fast message payloads (no tolist) ----
        flat_bin = bin_grid.ravel(order="C")
        flat_prob = prob100.ravel(order="C")

        msg_bin = OccupancyGrid()
        msg_bin.header = header
        msg_bin.info = info
        msg_bin.data = array('b', flat_bin.tobytes())

        msg_prob = OccupancyGrid()
        msg_prob.header = header
        msg_prob.info = info
        msg_prob.data = array('b', flat_prob.tobytes())

        return msg_bin, msg_prob
