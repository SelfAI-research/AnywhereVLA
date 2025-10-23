"""Adapters implementing strategy interfaces with current functions; light logging inside submodules."""
from typing import Iterable, List, Optional, Tuple
from scipy.ndimage import distance_transform_edt

try:
    from .interfaces import NormalEstimator, EdgeFinder, CandidateGenerator, ClearanceValidator, PlannerInterface
    from .normal_estimators import estimate_normal_frontier, estimate_normal_distance, estimate_normal_pca, estimate_normal_raycast
    from .candidates import find_edge_distance, generate_candidates
    from .coord_utils import world_to_map
    from .planner_client import PlannerChecker
except ImportError:  # pragma: no cover
    from interfaces import NormalEstimator, EdgeFinder, CandidateGenerator, ClearanceValidator, PlannerInterface  # type: ignore
    from normal_estimators import estimate_normal_frontier, estimate_normal_distance, estimate_normal_pca, estimate_normal_raycast  # type: ignore
    from candidates import find_edge_distance, generate_candidates  # type: ignore
    from coord_utils import world_to_map  # type: ignore
    from planner_client import PlannerChecker  # type: ignore
import logging

log = logging.getLogger(__name__)

class FrontierEstimator(NormalEstimator):
    def __init__(self, boundary: List[Tuple[int,int]]) -> None: self.boundary = boundary
    def estimate(self, region_mask, free_mask, obj_idx, ox, oy, oyaw, res):
        return estimate_normal_frontier(free_mask, self.boundary, obj_idx)

class DistanceGradEstimator(NormalEstimator):
    def estimate(self, region_mask, free_mask, obj_idx, ox, oy, oyaw, res):
        return estimate_normal_distance(region_mask, free_mask, obj_idx, ox, oy, oyaw, res)

class PcaBoundaryEstimator(NormalEstimator):
    def __init__(self, boundary: List[Tuple[int,int]]) -> None: self.boundary = boundary
    def estimate(self, region_mask, free_mask, obj_idx, ox, oy, oyaw, res):
        return estimate_normal_pca(self.boundary, ox, oy, oyaw, res, region_mask, free_mask, obj_idx)

class RaycastEstimator(NormalEstimator):
    def __init__(self, max_radius: float, step: float) -> None:
        self.max_radius, self.step = max_radius, step
    def estimate(self, region_mask, free_mask, obj_idx, ox, oy, oyaw, res):
        return estimate_normal_raycast(free_mask, obj_idx, res, self.max_radius, self.step)

class CompositeNormal(NormalEstimator):
    def __init__(self, estimators: List[NormalEstimator]) -> None: self.estimators = estimators
    def estimate(self, *args, **kwargs):
        log.error(f"CompositeNormal begin: the ests size:" + str(len(self.estimators)))
        i = 0
        for e in self.estimators:
            log.error(f"CompositeNormal check " + str(i) + str(type(e)))
            i+=1
            n = e.estimate(*args, **kwargs)
            if n is not None: return n
        return None

class EdgeFinderAdapter(EdgeFinder):
    def distance_to_edge(self, ox, oy, nx, ny, free_mask, mx, my, myaw, res, max_search):
        return find_edge_distance(ox, oy, nx, ny, free_mask, mx, my, myaw, res, max_search)

class CandidateGeneratorAdapter(CandidateGenerator):
    def __init__(self, max_search, sample_step, num_angles, num_offsets, offset_step, max_slide, slide_step) -> None:
        self.args = (max_search, sample_step, num_angles, num_offsets, offset_step, max_slide, slide_step)
    def generate(self, ex, ey, nx, ny, base_standoff):
        ms, ss, na, no, os, msl, sl = self.args
        return generate_candidates(ex, ey, nx, ny, base_standoff, ms, ss, na, no, os, msl, sl)

class DTMClearance(ClearanceValidator):
    def __init__(self) -> None:
        self.free_mask = None; self.cost = None; self.dist_map = None
        self.res = self.mx = self.my = self.myaw = 0.0
        self.min_clear = 0.0; self.disallow_inscribed = False; self.inscribed = 253
    def configure_grid(self, free_mask, cost, res, mx, my, myaw, min_clear, disallow_inscribed, inscribed_thresh):
        self.free_mask, self.cost = free_mask, cost
        self.dist_map = distance_transform_edt(free_mask)
        self.res, self.mx, self.my, self.myaw = res, mx, my, myaw
        self.min_clear = float(min_clear); self.disallow_inscribed = bool(disallow_inscribed)
        self.inscribed = int(inscribed_thresh)
    def ok(self, x: float, y: float) -> bool:
        i, j = world_to_map(x, y, self.mx, self.my, self.myaw, self.res)
        h, w = self.free_mask.shape  # type: ignore
        if i<0 or j<0 or i>=w or j>=h: return False
        if (self.dist_map[j, i] * self.res) < self.min_clear: return False  # type: ignore
        if self.disallow_inscribed and self.cost[j, i] >= self.inscribed: return False  # type: ignore
        return True

class PlannerAdapter(PlannerInterface):
    def __init__(self, checker: PlannerChecker) -> None: self.checker = checker
    def feasible(self, goal, start, mode: str) -> bool:
        log.error(f"PlannerAdapter feasible begin")
        try:
            return self.checker.check_action(goal, start)
        except Exception as e:
            log.error(f"PlannerAdapter error: {e}")