import os
import yaml

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import numpy as np

from yacs.config import CfgNode
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def load_cfg(path: str) -> CfgNode:
    """Load a YAML file or return an empty config if the file is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as file:
        data = yaml.safe_load(file) or {}
    cfg = CfgNode(data)
    cfg.freeze()
    return cfg

def _reliability(s: str) -> ReliabilityPolicy:
    s = s.lower()
    if s == "reliable":
        return ReliabilityPolicy.RELIABLE
    if s == "best_effort":
        return ReliabilityPolicy.BEST_EFFORT
    raise ValueError(f"Unknown reliability '{s}'")

def _durability(s: str) -> DurabilityPolicy:
    s = s.lower()
    if s == "transient_local":
        return DurabilityPolicy.TRANSIENT_LOCAL
    if s == "volatile":
        return DurabilityPolicy.VOLATILE
    raise ValueError(f"Unknown durability '{s}'")

@dataclass
class Transform:
    translation: np.ndarray  # shape (3,)
    rotation: np.ndarray     # shape (3,3)

    def to_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    @classmethod
    def from_msg(cls, msg) -> "Transform":
        # extract translation
        t = msg.transform.translation
        translation = np.array([t.x, t.y, t.z], dtype=float)
    
        # extract quaternion and convert to rotation matrix
        q = msg.transform.rotation
        quat = [q.x, q.y, q.z, q.w]
        rotation = R.from_quat(quat).as_matrix()

        return cls(translation=translation, rotation=rotation)

    def inverse(self) -> "Transform":
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return Transform(translation=inv_translation, rotation=inv_rotation)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points using current translation and rotation
        :param points: (N,3) float32 points
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (N,3)")

        # Apply Transformation
        matrix = self.to_matrix()
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
        transformed_points = (matrix @ points_homogeneous.T).T
        return transformed_points[:, :3]
