import os
import yaml

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import numpy as np

from yacs.config import CfgNode


def load_cfg(path: str) -> CfgNode:
    """Load a YAML file or return an empty config if the file is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as file:
        data = yaml.safe_load(file) or {}
    cfg = CfgNode(data)
    cfg.freeze()
    return cfg

def quat_from_yaw(yaw: float) -> np.ndarray:
    """Return quaternion [x, y, z, w] from yaw angle (in radians)."""
    return R.from_euler("z", yaw).as_quat()

@dataclass
class Transform:
    translation: np.ndarray  # shape (3,)
    rotation: np.ndarray     # shape (3,3)

    def to_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    def yaw(self) -> float:
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

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
