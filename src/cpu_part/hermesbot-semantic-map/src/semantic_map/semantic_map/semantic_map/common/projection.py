# semantic_map/common/projection.py
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose
from vision_msgs.msg import BoundingBox3D

from semantic_map.common import common_utils

@dataclass
class ObjectProjectionConfig:
    depth_margin_m: float = 0.20    # keep points within z_min + margin
    min_points: int = 5             # min points threshold
    increase_x_size: int = 5        # increase the bb x size by pixs
    increase_y_size: int = 5        # increase the bb y size by pixs 

class ObjectProjection:
    def __init__(self, **cfg_kwargs):
        self.cfg = ObjectProjectionConfig(**cfg_kwargs)

    @staticmethod
    def points_to_bbox(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mn = xyz.min(axis=0); mx = xyz.max(axis=0)
        center = 0.5 * (mn + mx)
        size = (mx - mn)
        return center, size

    def __call__(
        self,
        detections2d: Detection2DArray,
        xyz_in_fov: np.ndarray,
        uvz_in_fov: np.ndarray,
        frame_id: str,
        stamp: Optional[Time] = None,
    ) -> Tuple[Detection3DArray, List[np.ndarray]]:
        out = Detection3DArray()
        out.header = Header()
        out.header.frame_id = frame_id
        if stamp is not None:
            out.header.stamp = stamp
            
        point_clouds = []  # List to store point clouds for each detection

        if xyz_in_fov.size == 0 or len(detections2d.detections) == 0:
            return out, point_clouds

        uv, z = uvz_in_fov[:, :2], uvz_in_fov[:, 2]

        # pre-clip to sane ranges (avoid NaNs)
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(z)
        if not np.any(valid):
            return out, point_clouds
        uv = uv[valid]; z = z[valid]; xyz = xyz_in_fov[valid]

        for det2d in detections2d.detections:
            # bbox in pixel coords
            cx = det2d.bbox.center.position.x
            cy = det2d.bbox.center.position.y
            w  = det2d.bbox.size_x + self.cfg.increase_x_size
            h  = det2d.bbox.size_y + self.cfg.increase_y_size
            x0 = cx - 0.5 * w; y0 = cy - 0.5 * h
            x1 = cx + 0.5 * w; y1 = cy + 0.5 * h

            # select points inside bbox
            iu = (uv[:, 0] >= x0) & (uv[:, 0] <= x1)
            iv = (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
            mask2d = iu & iv
            if not np.any(mask2d):
                continue

            pts = xyz[mask2d]
            depth = z[mask2d]
            # depth gate: keep near-surface
            zmin = float(depth.min())
            pts = pts[depth <= (zmin + self.cfg.depth_margin_m)]

            if pts.shape[0] < self.cfg.min_points:
                continue

            center, size = self.points_to_bbox(pts)

            # pack Detection3D
            d3 = Detection3D()
            d3.header = out.header

            # class + score from the top hypothesis in 2D
            if det2d.results:
                top = det2d.results[0]
                oh = ObjectHypothesisWithPose()
                oh.hypothesis.class_id = top.hypothesis.class_id
                oh.hypothesis.score = top.hypothesis.score
                oh.pose = common_utils.pose_with_covariance(center)
                d3.id = top.hypothesis.class_id
                d3.results.append(oh)

            bb = BoundingBox3D()
            bb.center = common_utils.pose(center)
            bb.size = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
            d3.bbox = bb

            # Store the point cloud for this detection
            point_clouds.append(pts.copy())  # Copy to avoid reference issues
            out.detections.append(d3)

        return out, point_clouds