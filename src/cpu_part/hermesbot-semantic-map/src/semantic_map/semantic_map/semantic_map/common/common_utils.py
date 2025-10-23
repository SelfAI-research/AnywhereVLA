# semantic_map/common/common_utils.py
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import List, Optional, Tuple

import hashlib
import random

from cv_bridge import CvBridge
import os, yaml, cv2
from math import pi

import sensor_msgs_py.point_cloud2 as pc2

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovariance

from vision_msgs.msg import Detection2DArray

# ---------- Geometry / transforms ----------
def quat_to_mat_xyzw(x, y, z, w) -> np.ndarray:
    return R.from_quat([x, y, z, w]).as_matrix()

def transform_points(pts: np.ndarray, tf_msg) -> np.ndarray:
    Rm = quat_to_mat_xyzw(tf_msg.transform.rotation.x,
                           tf_msg.transform.rotation.y,
                           tf_msg.transform.rotation.z,
                           tf_msg.transform.rotation.w)
    t = np.array([tf_msg.transform.translation.x,
                  tf_msg.transform.translation.y,
                  tf_msg.transform.translation.z], dtype=np.float64)
    return (pts @ Rm.T) + t

def tf_msg_to_matrix(tf_msg) -> np.ndarray:
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = [t.x, t.y, t.z]
    return T

def transform_points_matrix(pts: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    one = np.ones((pts.shape[0], 1), dtype=np.float32)
    P = np.concatenate([pts.astype(np.float32), one], axis=1)  # N x 4
    Pw = (T_4x4.astype(np.float32) @ P.T).T
    return Pw[:, :3]

def invert_h(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def project_points(pts_cam, K, D):
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    uv,_ = cv2.projectPoints(pts_cam.astype(np.float64), rvec, tvec, K, D)
    return uv.reshape(-1,2)

def transform_to_yaw(T: np.ndarray) -> float:
    assert T.shape == (4, 4)
    R = T[:3, :3]  # rotation matrix part
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw

def quat_to_R(quat: Tuple) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)

def quat_to_R_tf(q: Quaternion) -> np.ndarray:
    quat = (q.x, q.y, q.z, q.w)
    return quat_to_R(quat)

def R_to_quat(R: np.ndarray) -> Quaternion:
    q = Quaternion()
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        q.w = 0.25 * s
        q.x = (R[2,1] - R[1,2]) / s
        q.y = (R[0,2] - R[2,0]) / s
        q.z = (R[1,0] - R[0,1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q.w = (R[2,1] - R[1,2]) / s
            q.x = 0.25 * s
            q.y = (R[0,1] + R[1,0]) / s
            q.z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q.w = (R[0,2] - R[2,0]) / s
            q.x = (R[0,1] + R[1,0]) / s
            q.y = 0.25 * s
            q.z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q.w = (R[1,0] - R[0,1]) / s
            q.x = (R[0,2] + R[2,0]) / s
            q.y = (R[1,2] + R[2,1]) / s
            q.z = 0.25 * s
    return q


# ----------------------- config helpers -----------------------

def extract_configuration(cfg_path:str, cfg_file:str='general_configuration.yaml'):
    cfg_path = os.path.join(cfg_path, cfg_file)
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def load_intrinsics(yaml_path):
    with open(yaml_path, 'r') as f: y = yaml.safe_load(f)
    K = np.array(y['camera_matrix']['data'], float).reshape(3, 3)
    D = np.array(y['distortion_coefficients']['data'], float).reshape(1, -1)
    W = int(y['image_size']['width']); H = int(y['image_size']['height'])
    return K, D, (W, H)

def load_extrinsic(yaml_path):
    with open(yaml_path, 'r') as f: y = yaml.safe_load(f)
    T = np.array(y['transformation_matrix'], float)
    assert T.shape == (4, 4)
    return T


# --- Zero-copy XYZ view over PointCloud2 ---
def cloud_to_xyz(msg: PointCloud2, skip_rate=1):
    fields = [f.name for f in msg.fields]
    if not all(k in fields for k in ('x','y','z')):
        return np.zeros((0, 3), dtype=np.float32)
    dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('_', f'V{msg.point_step - 12}')
    ])
    raw = np.frombuffer(msg.data, dtype=dtype)
    pts = np.vstack((raw['x'], raw['y'], raw['z'])).T
    return pts[::skip_rate] if skip_rate > 1 else pts


def pose(center: np.ndarray, q: Optional[Quaternion] = None) -> Pose:
    p = Pose()
    p.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
    p.orientation = q if q is not None else Quaternion(x=0.,y=0.,z=0.,w=1.)
    return p

def pose_with_covariance(
    center: np.ndarray,
    q: Optional[Quaternion] = None,
    covariance: np.ndarray | None = None
) -> PoseWithCovariance:
    pwc = PoseWithCovariance()
    pwc.pose = pose(center, q)
    if covariance is None:
        covariance = np.zeros((6, 6), dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (6, 6):
        raise ValueError(f"Covariance must be 6x6, got {covariance.shape}")
    pwc.covariance[:] = covariance.flatten().tolist()
    return pwc

# ----------------------- pub helpers -----------------------

def publish_image(pub, cv_img: np.ndarray, header: Header, *, encoding='bgr8', flip_image=-2):
    img = cv_img.copy()
    if flip_image in [0, 1, -1]:
        img = cv2.flip(img, flip_image)
    br = CvBridge()
    msg = br.cv2_to_imgmsg(img, encoding=encoding)
    msg.header = header
    pub.publish(msg)

def publish_xyz_cloud(pub, xyz: np.ndarray, header: Header):
    if xyz is None or xyz.size == 0: return
    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg = pc2.create_cloud(header, fields, xyz.astype(np.float32))
    pub.publish(msg)

def publish_xyzr_cloud(pub, xyzr_points: np.ndarray, header: Header):
    if xyzr_points.shape[0] == 0:
        return
    # pts = np.column_stack((xyzr_points.astype(np.float32), rgb_float.astype(np.float32)))
    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_msg = pc2.create_cloud(header, fields, xyzr_points.tolist())
    pub.publish(cloud_msg)

def color_from_id(id_str):
    """Generate deterministic RGB color from a string (class_id, etc)."""
    h = int(hashlib.sha1(id_str.encode("utf-8")).hexdigest(), 16)
    random.seed(h)
    return [int(255 * random.random()) for _ in range(3)]  # [r, g, b]

def draw_on_image(cv_image: np.ndarray, uvz_filtered: np.ndarray, detections2d: Detection2DArray=None, add=(5,5)) -> np.ndarray:
    img = cv_image.copy()

    # points
    if uvz_filtered is not None and uvz_filtered.size:
        for uu, vv, _ in uvz_filtered:
            cv2.circle(img, (int(uu), int(vv)), 1, (0, 255, 0), -1)

    # boxes
    if detections2d and getattr(detections2d, "detections", None):
        for d in detections2d.detections:
            b = d.bbox
            c = getattr(b.center, "position", b.center)     # supports Pose2D (x,y,theta) or Pose(position.x/y)
            rect1 = float(c.x), float(c.y), float(b.size_x), float(b.size_y)
            rect2 = float(c.x), float(c.y), float(b.size_x+add[0]), float(b.size_y+add[1])
            ang = -np.degrees(getattr(b.center, "theta", 0.0))

            for cx, cy, sx, sy in [rect1,rect2]:
                pts = cv2.boxPoints(((cx, cy), (sx, sy), ang)).astype(int)
                hyp = d.results[0].hypothesis if getattr(d, "results", None) else None
                cid, score = (getattr(hyp, "class_id", ""), getattr(hyp, "score", None)) if hyp else ("", None)
                col = tuple(reversed(color_from_id(str(cid or "det"))))  # RGB->BGR
                cv2.polylines(img, [pts], True, col, 2)
                if cid: cv2.putText(img, f"{cid}{'' if score is None else f' {score:.2f}'}", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    return img

def publish_colored_cloud(
    pub,
    xyz: np.ndarray,
    uvz: np.ndarray,
    header: Header,
    image: Optional[np.ndarray],
) -> None:
    """
    Publish PointCloud2 with per-point RGB sampled from 'image' at (u,v).
    Fields: x(float32), y(float32), z(float32), rgb(float32).
    """
    if xyz is None or uvz is None or xyz.size == 0 or uvz.size == 0:
        return
    n = min(xyz.shape[0], uvz.shape[0])
    xyz = xyz[:n].astype(np.float32, copy=False)
    uvz = uvz[:n].astype(np.float32, copy=False)

    # sample colors
    if image is not None and image.size != 0:
        h, w = image.shape[:2]
        u = np.clip(np.round(uvz[:, 0]).astype(np.int32), 0, w - 1)
        v = np.clip(np.round(uvz[:, 1]).astype(np.int32), 0, h - 1)
        bgr = image[v, u]                           # (N,3), uint8
        if bgr.dtype != np.uint8:
            bgr = bgr.astype(np.uint8, copy=False)
        # BGR -> RGB
        r = bgr[:, 2].astype(np.uint32)
        g = bgr[:, 1].astype(np.uint32)
        b = bgr[:, 0].astype(np.uint32)
    else:
        # fallback: white
        r = np.full(n, 255, dtype=np.uint32)
        g = np.full(n, 255, dtype=np.uint32)
        b = np.full(n, 255, dtype=np.uint32)

    # pack into single uint32, then view as float32
    rgb_uint32 = (r << 16) | (g << 8) | b
    rgb_float = rgb_uint32.view(np.float32)

    # interleave into N×4 float32 [x,y,z,rgb]
    data = np.zeros((n, 4), dtype=np.float32)
    data[:, 0:3] = xyz
    data[:, 3] = rgb_float

    publish_xyzr_cloud(pub, data, header)



# ----------------------- COCO names (ultralytics) -----------------------
from pathlib import Path

_ID2NAME: dict[int, str] = {}
_NAME2ID: dict[str, int] = {}

def load_labels_from_yaml(yaml_path: Path) -> None:
    """yaml_source: path(str/Path) or dict with key 'names'."""
    with open(os.path.expanduser(yaml_path), "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    items = sorted((int(k), str(v)) for k, v in names.items())
    id2name = {i: n for i, n in items}
    _ID2NAME.clear(); _ID2NAME.update(id2name)
    _NAME2ID.clear(); _NAME2ID.update({str(n).strip().lower(): i for i, n in _ID2NAME.items()})

def coco_id2name(class_id):
    return _ID2NAME.get(int(class_id), class_id)

def coco_name2id(name: str):
    return _NAME2ID.get(name.strip().lower())


# -------------------- From Semantic Perception -----------------------

def time_to_secs(stamp) -> Optional[float]:
    try: return float(stamp.sec) + 1e-9 * float(stamp.nanosec)
    except Exception: return None


def robust_filter(pts: np.ndarray, method: str, k: float) -> np.ndarray:
    if pts.shape[0] < 5 or k <= 0: return np.ones(pts.shape[0], dtype=bool)
    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    if method == "mad":
        med = np.median(d); mad = np.median(np.abs(d - med))
        sigma = 1.4826 * mad
        return d <= (med + k * sigma)
    if method == "sigma":
        mu = d.mean(); sd = d.std()
        return d <= (mu + k * sd)
    raise ValueError(f"Method {method} not implemented!")
 
def eig_sorted_cov(cov: np.ndarray):
    cov = np.asarray(cov, dtype=np.float32)
    cov = cov + 1e-9 * np.eye(3, dtype=np.float32)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]

def ellipsoid_axes_from_cov(cov: np.ndarray, k_std: float) -> np.ndarray:
    evals, _ = eig_sorted_cov(cov)
    return k_std * np.sqrt(np.maximum(evals, 1e-12))

def ellipsoid_volume(axes: np.ndarray) -> float:
    return (4.0 / 3.0) * pi * float(axes[0]) * float(axes[1]) * float(axes[2])

def yaw_from_quat(world_quat_xyzw, optical_vec_lidar, wrap_0_2pi=True, eps=1e-12):
    """
    world_quat_xyzw: [x,y,z,w] quaternion of LiDAR in world frame (ROS ordering).
    optical_vec_lidar: 3D forward axis of the camera expressed in LiDAR frame (need not be unit).
    Returns yaw in radians w.r.t. world Z-up: [0, 2pi) by default; (-pi, pi] if not wrap_0_2pi.
    """
    Rw = quat_to_R_tf(world_quat_xyzw)

    vL = np.asarray(optical_vec_lidar, dtype=np.float64)
    n = np.linalg.norm(vL)
    if n < eps:
        raise ValueError("optical_vec_lidar has near-zero length")
    vL /= n

    vW = Rw @ vL
    r_xy = np.hypot(vW[0], vW[1])
    if r_xy < 1e-8:
        raise ValueError("Projected direction is near vertical; yaw undefined")

    yaw = np.arctan2(vW[1], vW[0])
    if wrap_0_2pi and yaw < 0:
        yaw += 2*np.pi
    return float(yaw)

def sigmoid(z: float) -> float: 
    return float(1.0 / (1.0 + np.exp(-z)))

def union_length_on_circle(arcs: List[Tuple[float, float]]) -> float:
    if not arcs: return 0.0
    spans: List[Tuple[float, float]] = []
    for a, b in arcs:
        a = a % (2*pi); b = b % (2*pi)
        if a <= b: spans.append((a, b))
        else:
            spans.append((a, 2*pi))
            spans.append((0.0, b))
    spans.sort(key=lambda t: t[0])
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + 1e-9: cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(e - s for s, e in merged)
