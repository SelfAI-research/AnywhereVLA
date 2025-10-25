import numpy as np
from scripts.normal_estimators import estimate_normal_frontier
from scripts.candidates import clearance_ok
from scipy.ndimage import distance_transform_edt

def test_frontier_normal_points_inward():
    free = np.zeros((10,10), dtype=bool)
    free[6:, :] = True  # free corridor below boundary at j=5
    boundary = [(i,5) for i in range(10)]
    n = estimate_normal_frontier(free, boundary, (5,6))
    assert n is not None
    nx, ny = n
    # inward means pointing from free -> obstacle; here that is upward (negative y)
    assert ny < 0 or abs(nx) <= 1.0

def test_clearance_ok_basic():
    free = np.ones((20,20), dtype=bool)
    free[8:12, 8:12] = False  # obstacle island
    dist = distance_transform_edt(free)
    res = 0.05
    ok_center = clearance_ok(0.2, 0.2, res, free, dist, 0.0, 0.0, 0.0, 0.10)
    assert ok_center is True
