# semantic_map/common/semantic.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from math import pi
import yaml
from scipy.spatial import cKDTree

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Quaternion
from vision_msgs.msg import Detection3DArray, Detection3D
from visualization_msgs.msg import Marker, MarkerArray

from semantic_map.common import common_utils


# ---------------- config ----------------
@dataclass
class SemanticPerceptionConfig:
    # accumulation
    keep_max_points_per_obj: int = 20000
    outlier_method: str = "mad"             # "mad" | "sigma"
    outlier_k: float = 2.0

    # clustering (on-demand)
    cluster_eps_m: float = 0.22
    cluster_min_pts: int = 40
    cluster_voxel_m: float = 0.03
    cluster_max_points: int = 250_000   # safety cap per class before clustering

    # ellipsoid + confidence
    ellipsoid_std_scale: float = 2.0
    density_k_std: float = 1.5
    density_ref: float = 200.0
    count_ref: float = 80.0
    coverage_theta_deg: float = 15.0
    optical_vector: Tuple[float,float,float] = (0.0, 1.0, 0.0)

    prob_k_density: float = 2.0
    prob_k_coverage: float = 2.0
    prob_k_count: float = 1.5
    prob_k_avgscore: float = 1.0
    prob_bias: float = -2.0


# ---------------- accumulation DB ----------------
@dataclass
class Observation:
    pts_w: np.ndarray                    # (Ni,3)
    yaw_rad: Optional[float] = None
    score: Optional[float] = None
    t_sec: Optional[float] = None

@dataclass
class ClassDB:
    cfg: SemanticPerceptionConfig
    observations: List[Observation] = field(default_factory=list)
    n_points_total: int = 0

    def add(self, pts_w: np.ndarray, yaw: Optional[float], score: Optional[float], t_sec: Optional[float]):
        if pts_w is None or pts_w.size == 0:
            return
        # per-observation robust trimming (light)
        mask = common_utils.robust_filter(pts_w, self.cfg.outlier_method, self.cfg.outlier_k)
        pts_filter = pts_w[mask]
        if pts_filter.shape[0] == 0:
            return
        self.observations.append(Observation(pts_filter.astype(np.float32), yaw, score, t_sec))
        self.n_points_total += int(pts_filter.shape[0])

    def clear(self):
        self.observations.clear()
        self.n_points_total = 0

    def aggregate(self, voxel_m: float | None = None, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.observations:
            return np.zeros((0,3), np.float32), np.zeros((0,), np.int32)
        pts = []
        obs_ids = []
        for i, ob in enumerate(self.observations):
            pts.append(ob.pts_w)
            obs_ids.append(np.full((ob.pts_w.shape[0],), i, dtype=np.int32))
        P = np.vstack(pts).astype(np.float32)
        I = np.concatenate(obs_ids, axis=0)

        # voxel downsample
        if voxel_m and voxel_m > 0.0 and P.shape[0] > 0:
            q = np.floor(P / float(voxel_m)).astype(np.int32)
            _, idx = np.unique(q, axis=0, return_index=True)
            P = P[idx]; I = I[idx]

        # spill cap
        if max_points and P.shape[0] > max_points:
            sel = np.random.choice(P.shape[0], size=max_points, replace=False)
            P = P[sel]; I = I[sel]
        return P, I


# ---------------- tracks ----------------
@dataclass
class ObjectTrack:
    track_id: int
    class_id: int | str
    cfg: SemanticPerceptionConfig

    points: np.ndarray
    yaw_angles: List[float]
    det_score_list: List[float]
    first_ts: Optional[float]
    last_ts: Optional[float]

    _mean: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    _cov:  np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32)*1e-4)
    _obs_p: int = 0

    def compute_stats(self):
        if self.points.size == 0:
            self._mean[:] = 0.0
            self._cov[:] = np.eye(3, dtype=np.float32)*1e-4
            self._obs_p = 0
            return
        P = self.points.astype(np.float32, copy=False)
        mask = common_utils.robust_filter(P, self.cfg.outlier_method, self.cfg.outlier_k)
        P = P[mask]
        self._obs_p = int(P.shape[0])
        if self._obs_p < 2:
            self._mean = P.mean(axis=0).astype(np.float32) if self._obs_p > 0 else np.zeros(3, np.float32)
            self._cov  = np.eye(3, dtype=np.float32)*1e-4
            return
        mu = P.mean(axis=0).astype(np.float32)
        X  = (P - mu).astype(np.float32)
        M2 = (X.T @ X).astype(np.float32)
        cov = M2 / max(self._obs_p - 1, 1)
        self._mean = mu
        self._cov  = cov + 1e-6*np.eye(3, dtype=np.float32)

    def get_mean(self) -> np.ndarray:
        return self._mean

    def get_cov(self) -> np.ndarray:
        return self._cov

    def _score_term(self) -> float:
        return float(np.clip(np.mean(self.det_score_list), 0.0, 1.0)) if self.det_score_list else 0.0

    def _density_term(self) -> float:
        axes = common_utils.ellipsoid_axes_from_cov(self._cov, self.cfg.density_k_std)
        V = common_utils.ellipsoid_volume(axes) + 1e-9
        rho = float(self.points.shape[0]) / V
        return float(1.0 - np.exp(-rho / max(self.cfg.density_ref, 1e-9)))

    def _coverage_term(self) -> float:
        if not self.yaw_angles: return 0.0
        theta0 = np.deg2rad(self.cfg.coverage_theta_deg)
        arcs = [(ang - theta0, ang + theta0) for ang in self.yaw_angles]
        L = common_utils.union_length_on_circle(arcs)
        return float(np.clip(L / (2.0 * pi), 0.0, 1.0))

    def _count_term(self) -> float:
        N = float(self._obs_p)
        N0 = float(self.cfg.count_ref)
        return float(1.0 - np.exp(-N / max(N0, 1e-9)))

    def get_confidence(self) -> float:
        d = self._density_term()
        c = self._coverage_term()
        n = self._count_term()
        s = self._score_term()
        x = (self.cfg.prob_k_density * d +
             self.cfg.prob_k_coverage * c +
             self.cfg.prob_k_count * n +
             self.cfg.prob_k_avgscore * s +
             self.cfg.prob_bias)
        return common_utils.sigmoid(x)

    def export_summary(self) -> Dict:
        axes = common_utils.ellipsoid_axes_from_cov(self._cov, self.cfg.density_k_std)
        V = common_utils.ellipsoid_volume(axes) + 1e-9
        rho = float(self.points.shape[0]) / V
        theta0 = np.deg2rad(self.cfg.coverage_theta_deg)
        arcs = [(ang - theta0, ang + theta0) for ang in self.yaw_angles]
        L = common_utils.union_length_on_circle(arcs)
        return {
            "track_id": int(self.track_id),
            "class_id": int(self.class_id) if str(self.class_id).isdigit() else str(self.class_id),
            "class_name": str(common_utils.coco_id2name(self.class_id)),
            "n_points": int(self.points.shape[0]),
            "first_ts": None if self.first_ts is None else float(self.first_ts),
            "last_ts": None if self.last_ts is None else float(self.last_ts),
            "mean": self._mean.astype(float).tolist(),
            "covariance": self._cov.astype(float).tolist(),
            "confidence_prob": float(self.get_confidence()),
            "components": {
                "density": self._density_term(),
                "coverage": self._coverage_term(),
                "count": self._count_term(),
                "avg_score": self._score_term(),
            },
            "density_meta": {"axes_std": [float(a) for a in axes], "volume_m3": float(V), "rho": float(rho)},
            "coverage_meta": {"theta_deg": float(self.cfg.coverage_theta_deg), "union_len_rad": float(L),
                              "yaw_angles_deg": [float(np.degrees(a)) for a in self.yaw_angles]},
            "count_meta": {"inlier_points": int(self._obs_p), "count_ref": float(self.cfg.count_ref)},
        }


# ---------------- main module ----------------
class SemanticPerception:
    """
    Online: accumulate per-class observations only.
    On-demand: cluster accumulated points class-wise, build ObjectTrack list.
    """
    def __init__(self, world_frame: str, **cfg_kwargs):
        self.world_frame = world_frame
        self.cfg = SemanticPerceptionConfig(**cfg_kwargs)
        self._db_by_class: Dict[int|str, ClassDB] = {}
        self._tracks_by_class: Dict[int|str, List[ObjectTrack]] = {}
        self._next_id = 1

    # --------- online accumulation ---------
    def update(
        self,
        detections3d: Detection3DArray,
        point_clouds: List[np.ndarray],
        T_src_to_world: np.ndarray,
    ) -> None:
        t_sec = common_utils.time_to_secs(detections3d.header.stamp)
        view_q = common_utils.R_to_quat(T_src_to_world[:3, :3])
        view_yaw = common_utils.yaw_from_quat(view_q, self.cfg.optical_vector)

        for det3d, pts_src in zip(detections3d.detections, point_clouds, strict=True):
            cls_id, score = self._top_class(det3d)
            if cls_id is None or pts_src is None or pts_src.size == 0:
                continue
            pts_w = common_utils.transform_points_matrix(pts_src, T_src_to_world)
            if pts_w.shape[0] == 0:
                continue
            db = self._db_by_class.setdefault(cls_id, ClassDB(self.cfg))
            db.add(pts_w, view_yaw, score, t_sec)

    # --------- on-demand clustering ---------
    def build_objects(self, class_id: int|str|None = None) -> None:
        self._next_id = 1
        targets = [class_id] if class_id is not None else list(self._db_by_class.keys())
        for cid in targets:
            db = self._db_by_class.get(cid)
            if db is None or not db.observations:
                self._tracks_by_class[cid] = []
                continue
            P, obs_ids = db.aggregate(self.cfg.cluster_voxel_m, self.cfg.cluster_max_points)
            if P.shape[0] == 0:
                self._tracks_by_class[cid] = []
                continue
            labels = _dbscan_radius(P, eps=self.cfg.cluster_eps_m, min_pts=self.cfg.cluster_min_pts)
            ncl = int(labels.max()) + 1 if labels.size else 0
            tracks: List[ObjectTrack] = []

            for k in range(ncl):
                idx = np.flatnonzero(labels == k)
                if idx.size == 0: continue
                pts_k = P[idx]

                # per-cluster yaw/score/time from contributing observations
                contrib, counts = np.unique(obs_ids[idx], return_counts=True)
                yaw_list, score_list, t_list = [], [], []
                first_ts, last_ts = None, None
                for ob_id, c in zip(contrib, counts):
                    ob = db.observations[int(ob_id)]
                    if ob.yaw_rad is not None and c >= 3:
                        yaw_list.append(float(ob.yaw_rad))
                    if ob.score is not None:
                        score_list.extend([float(ob.score)] * min(int(c), 5))
                    if ob.t_sec is not None:
                        t_list.append(float(ob.t_sec))
                if t_list:
                    first_ts, last_ts = float(min(t_list)), float(max(t_list))

                tr = ObjectTrack(
                    track_id=self._next_id, class_id=cid, cfg=self.cfg,
                    points=pts_k.astype(np.float32), yaw_angles=yaw_list,
                    det_score_list=score_list, first_ts=first_ts, last_ts=last_ts
                )
                self._next_id += 1
                tr.compute_stats()
                tracks.append(tr)

            self._tracks_by_class[cid] = tracks

    # --------- consumers ---------
    def get_objects(self, id: str | int = "all") -> List[ObjectTrack]:
        if id == "all":
            return [t for L in self._tracks_by_class.values() for t in L]
        for L in self._tracks_by_class.values():
            for t in L:
                if t.track_id == id:
                    return [t]
        return []

    def to_marker_array(self) -> MarkerArray:
        out = MarkerArray()
        idx = 0
        for cid, tracks in self._tracks_by_class.items():
            color = common_utils.color_from_id(str(cid))
            color_np = (np.array(color, dtype=np.float32) / 350.0).astype(float)
            cls_name = common_utils.coco_id2name(cid)
            for t in tracks:
                cov = t.get_cov()
                mean = t.get_mean()
                axes = common_utils.ellipsoid_axes_from_cov(cov, self.cfg.ellipsoid_std_scale)
                _, evecs = common_utils.eig_sorted_cov(cov)
                q = common_utils.R_to_quat(evecs)

                conf = t.get_confidence()

                mk = Marker()
                mk.header.frame_id = self.world_frame
                mk.ns = f"obj/{cid}"; mk.id = idx; idx += 1
                mk.type = Marker.SPHERE; mk.action = Marker.ADD
                mk.pose = common_utils.pose(mean, q)
                mk.scale = Vector3(x=float(2*axes[0]), y=float(2*axes[1]), z=float(2*axes[2]))
                mk.color = ColorRGBA(r=color_np[0], g=color_np[1], b=color_np[2], a=0.35)
                out.markers.append(mk)

                txt = Marker()
                txt.header.frame_id = self.world_frame
                txt.ns = f"lbl/{cid}"; txt.id = idx; idx += 1
                txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
                txt.pose = common_utils.pose(mean + evecs @ np.array([0, 0, max(axes[2], 0.1)], np.float32))
                txt.scale.z = 0.18
                txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
                txt.text = f"{cid}:{cls_name} #{t.track_id} ({int(100*conf)}%)"
                out.markers.append(txt)
        return out

    def save_to_file(self, path: str) -> None:
        data = {
            "world_frame": self.world_frame,
            "num_tracks": int(sum(len(v) for v in self._tracks_by_class.values())),
            "config": {
                "cluster_eps_m": float(self.cfg.cluster_eps_m),
                "cluster_min_pts": int(self.cfg.cluster_min_pts),
                "cluster_voxel_m": float(self.cfg.cluster_voxel_m),
                "ellipsoid_std_scale": float(self.cfg.ellipsoid_std_scale),
                "density_k_std": float(self.cfg.density_k_std),
                "density_ref": float(self.cfg.density_ref),
                "count_ref": float(self.cfg.count_ref),
                "coverage_theta_deg": float(self.cfg.coverage_theta_deg),
                "w_density": float(self.cfg.prob_k_density),
                "w_coverage": float(self.cfg.prob_k_coverage),
                "w_count": float(self.cfg.prob_k_count),
                "w_avgscore": float(self.cfg.prob_k_avgscore),
                "bias": float(self.cfg.prob_bias),
            },
            "objects": [],
        }
        for tracks in self._tracks_by_class.values():
            for t in tracks:
                data["objects"].append(t.export_summary())
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    # --------- helpers ---------
    def _top_class(self, det: Detection3D) -> Tuple[Optional[int | str], float]:
        if not det.results: return None, 0.0
        cid = det.results[0].hypothesis.class_id
        try:
            cid = int(cid)
        except Exception:
            pass
        score = float(det.results[0].hypothesis.score)
        return cid, score


# ---------------- clustering utils ----------------
def _dbscan_radius(P: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    if P.shape[0] == 0:
        return np.zeros((0,), np.int32)
    tree = cKDTree(P)
    N = P.shape[0]
    labels = -np.ones(N, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    cluster_id = 0

    for i in range(N):
        if visited[i]: continue
        visited[i] = True
        neigh = tree.query_ball_point(P[i], r=eps)
        if len(neigh) < min_pts:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = [j for j in neigh if j != i]
        k = 0
        while k < len(seeds):
            j = seeds[k]; k += 1
            if not visited[j]:
                visited[j] = True
                neigh_j = tree.query_ball_point(P[j], r=eps)
                if len(neigh_j) >= min_pts:
                    for n in neigh_j:
                        if n not in seeds:
                            seeds.append(n)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels
