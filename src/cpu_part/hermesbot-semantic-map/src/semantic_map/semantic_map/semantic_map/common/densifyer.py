# semantic_map/common/densifyer.py
import numpy as np
import cv2
from dataclasses import dataclass, fields
from typing import Optional
from semantic_map.common import common_utils

@dataclass
class LidarDensifierConfig:
    num_rings: Optional[int] = 16
    ring_angles_deg: Optional[np.ndarray] = None
    az_bins: int = 2048
    interp_per_gap: int = 1
    max_range_jump: Optional[float] = 1.5
    max_gap_m: Optional[float] = None

    # To filter left params
    def __init__(self, **kwargs):
        allowed = {f.name for f in fields(LidarDensifierConfig)}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(self, k, v)

class LidarDensifier:
    """
    LiDAR → camera FOV filtering + Velodyne ring densification.
    Methods:
      - filter_points_on_image(xyz_lidar, cv_image) -> (xyz_filtered, uvz_filtered)
      - densify_velodyne_rings(xyz_lidar, rings=None) -> xyz_aug
      - draw_on_image(cv_image, uvz) -> debug_img
    """

    def __init__(
        self,
        T_lidar_to_cam: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        **cfg_kwargs
    ):
        self.T_lc = np.asarray(T_lidar_to_cam, dtype=np.float32)
        self.K = np.asarray(camera_matrix, dtype=np.float32)
        self.D = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1) if dist_coeffs is not None else None

        self.cfg = LidarDensifierConfig(**cfg_kwargs)
                                        
        # Default VLP-16 angles if user didn’t provide ring metadata
        if self.cfg.ring_angles_deg is None and (self.cfg.num_rings == 16):
            self.cfg.ring_angles_deg = np.array(
                [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15],
                dtype=np.float32
            )

    # ---- public API ----

    def densify_velodyne_rings(
        self,
        xyz_lidar: np.ndarray,
        rings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Interpolate between adjacent rings inside the same azimuth bin; append new samples."""
        if xyz_lidar.size == 0 or self.cfg.interp_per_gap <= 0:
            return xyz_lidar

        xyz = xyz_lidar.astype(np.float32, copy=False)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r_xy = np.hypot(x, y).astype(np.float32)
        rng = np.sqrt(r_xy * r_xy + z * z).astype(np.float32)
        phi = np.arctan2(y, x).astype(np.float32)
        phi[phi < 0.0] += 2.0 * np.pi

        az_step = (2.0 * np.pi) / float(self.cfg.az_bins)
        az_bin = np.minimum((phi / az_step).astype(np.int32), self.cfg.az_bins - 1)

        elev = np.arctan2(z, r_xy).astype(np.float32)

        # ---- ring assignment ----
        if rings is not None:
            rings = np.asarray(rings)
            if rings.shape[0] != xyz.shape[0]:
                raise ValueError("rings length must match number of points")
            rings = rings.astype(np.int32, copy=False)
            n_rings = int(rings.max()) + 1 if rings.size else 0
        else:
            if self.cfg.ring_angles_deg is not None:
                ang = self.cfg.ring_angles_deg * (np.pi / 180.0)
                rings = np.argmin(np.abs(elev[:, None] - ang[None, :]), axis=1).astype(np.int32)
                n_rings = int(ang.size)
            else:
                # Uniform quantization if angles unknown
                n_rings = int(self.cfg.num_rings) if self.cfg.num_rings is not None else 16
                e_min, e_max = float(np.min(elev)), float(np.max(elev))
                if not np.isfinite(e_min) or not np.isfinite(e_max) or e_max <= e_min:
                    return xyz
                scale = (elev - e_min) / max(e_max - e_min, 1e-6)
                rings = np.clip(np.round(scale * (n_rings - 1)).astype(np.int32), 0, n_rings - 1)

        if n_rings < 1:
            return xyz

        # One representative per (az_bin, ring): smallest range
        cells = (az_bin.astype(np.int64) * int(n_rings) + rings.astype(np.int64))
        order = np.lexsort((rng, cells))
        cells_sorted = cells[order]
        first_idx = np.concatenate(([0], np.flatnonzero(np.diff(cells_sorted)) + 1))
        rep_idx = order[first_idx]
        rep_cells = cells_sorted[first_idx]
        rep_xyz = xyz[rep_idx]
        rep_rng = rng[rep_idx]

        grid_xyz = np.full((self.cfg.az_bins, n_rings, 3), np.nan, dtype=np.float32)
        grid_rng = np.full((self.cfg.az_bins, n_rings), np.nan, dtype=np.float32)
        az_u = (rep_cells // n_rings).astype(np.int32)
        ring_u = (rep_cells %  n_rings).astype(np.int32)
        grid_xyz[az_u, ring_u] = rep_xyz
        grid_rng[az_u, ring_u] = rep_rng

        S = grid_xyz[:, :-1, :]
        E = grid_xyz[:,  1:, :]
        valid = np.isfinite(S[..., 0]) & np.isfinite(E[..., 0])

        if self.cfg.max_range_jump is not None:
            Sr = grid_rng[:, :-1]; Er = grid_rng[:, 1:]
            mnr = np.minimum(Sr, Er)
            ratio = (np.maximum(Sr, Er) / np.maximum(mnr, 1e-6))
            valid &= (ratio <= self.cfg.max_range_jump)

        if self.cfg.max_gap_m is not None:
            d2 = np.sum((E - S) * (E - S), axis=2)
            valid &= (d2 <= (self.cfg.max_gap_m ** 2))

        if not np.any(valid):
            return xyz

        # Interpolate between S and E
        pieces = []
        M = max(1, self.cfg.interp_per_gap)
        for t in range(1, M + 1):
            a = t / float(M + 1)
            P = (1.0 - a) * S + a * E
            P = P[valid]
            if P.size:
                pieces.append(P.astype(np.float32, copy=False))

        if not pieces:
            return xyz
        return np.vstack((xyz, np.concatenate(pieces, axis=0)))

    def filter_points_on_image(self, xyz_lidar: np.ndarray, cv_image: np.ndarray):
        """Keep points projecting into the image. Return points (LiDAR frame) and uvz in image coords."""
        if xyz_lidar.size == 0:
            return xyz_lidar, np.empty((0, 3), dtype=np.float32)

        xyz = xyz_lidar.astype(np.float32, copy=False)
        xyz_cam = common_utils.transform_points_matrix(xyz, self.T_lc)

        front = xyz_cam[:, 2] > 0.0
        if not np.any(front):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        xyz_cam_f = xyz_cam[front]

        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        uv, _ = cv2.projectPoints(xyz_cam_f, rvec, tvec, self.K, self.D)
        uv = uv.reshape(-1, 2)

        h, w = cv_image.shape[:2]
        u = np.round(uv[:, 0])
        v = np.round(uv[:, 1])

        # Remove NaN/Inf before int cast
        valid = np.isfinite(u) & np.isfinite(v)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)
        xyz_cam_f = xyz_cam_f[valid]

        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        # Map back to original indices
        keep_src = np.flatnonzero(front)[in_img]
        
        xyz_filtered = xyz[keep_src]
        u = u[in_img].astype(np.float32)
        v = v[in_img].astype(np.float32)
        z = xyz_cam_f[in_img, 2].astype(np.float32)
        uvz_filtered = np.column_stack((u, v, z))
        return xyz_filtered, uvz_filtered
