from dataclasses import dataclass
import numpy as np

from utils import Transform

@dataclass
class PreprocConfig:
    z_min: float = -1.0
    z_max: float = 5.0
    range_filter: tuple[float, float] = (0.0, 25.0)  # Min and Max point distance (meters); set max range -1 for unlimited
    fov_filter: tuple[float, float] = (-180, 180)  # Min and Max field of view (degrees);
    outlier_mode: str = "none"  # "none", "density"
    grid_r: float = 0.10        # cell size for density filter (m)
    min_pts_per_cell: int = 3   # keep cells with >= N pts


class Preprocessor:
    """
    Turns 3D points (N,3) into a 2D scan (N2,2) in the same WORLD frame.
    Steps: z filter in SENSOR orientation -> optional outlier filter (GLOBAL XY).
    """
    def __init__(self, cfg: PreprocConfig):
        self.cfg = cfg

    def _filter_z_axis(self, local_pts: np.ndarray) -> np.ndarray:
        z_local = local_pts[:, 2]
        z_mask = (z_local >= self.cfg.z_min) & (z_local <= self.cfg.z_max)
        return local_pts[z_mask]

    def _density_filter(self, xy: np.ndarray) -> np.ndarray:
        if self.cfg.outlier_mode != "density" or xy.shape[0] == 0:
            return xy
        r = float(self.cfg.grid_r)
        inv_r = 1.0 / r
        # hash to grid
        gi = np.floor(xy[:, 0] * inv_r).astype(np.int64)
        gj = np.floor(xy[:, 1] * inv_r).astype(np.int64)
        key = gi * np.int64(73856093) ^ gj * np.int64(19349663)
        # count per cell
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        keep = counts[inv] >= self.cfg.min_pts_per_cell
        return xy[keep]

    def _range_filter(self, xy: np.ndarray) -> np.ndarray:
        """
        :param xy: (N,2) float32 in Local frame
        """
        if not self.cfg.range_filter or xy.shape[0] == 0:
            return xy
        min_range = 0 if self.cfg.range_filter[0] < 0 else self.cfg.range_filter[0]
        max_range = np.infty if self.cfg.range_filter[1] < 0 else self.cfg.range_filter[1]
        sq_dists = np.square(xy[:, 0]) + np.square(xy[:, 1])
        keep_mask = (sq_dists >= min_range**2) & (sq_dists <= max_range**2)
        return xy[keep_mask]

    def _fov_filter(self, xy: np.ndarray) -> np.ndarray:
        """
        :param xy: (N,2) float32 in Local frame
        """
        if not self.cfg.fov_filter or xy.shape[0] == 0:
            return xy
        min_fov, max_fov = np.radians(self.cfg.fov_filter)
        angles = np.arctan2(xy[:, 1], xy[:, 0])
        keep_mask = (angles >= min_fov) & (angles <= max_fov)
        return xy[keep_mask]

    def process(self, points_xyz: np.ndarray, transform: Transform) -> np.ndarray:
        """
        :param points_xyz: (N,3) float32 in GLOBAL frame
        :param transform: sensor->global (p_g = R p_s + t)
        :return: 2D scan (M,2) in GLOBAL frame
        """
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError("points_xyz must be (N,3)")

        local_pts = transform.inverse().transform(points_xyz)

        # Apply Z-axis filtering
        local_pts = self._filter_z_axis(local_pts)
        xy_local = local_pts[:, :2]
        
        # Apply filters
        xy_local = self._range_filter(xy_local)
        xy_local = self._fov_filter(xy_local)
        xy_local = self._density_filter(xy_local)

        # Transform to global
        xyz_local = np.hstack([xy_local, np.ones((xy_local.shape[0], 1), dtype=xy_local.dtype)])
        xyz_global = transform.transform(xyz_local)
        xy_global = xyz_global[:, :2]
        return xy_global
