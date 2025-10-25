"""Costmap/occupancy conversion and mask building."""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid
import tf_transformations as tft
import logging

log = logging.getLogger(__name__)

def costmap_to_array(msg: Costmap) -> Tuple[NDArray[np.uint8], float, float, float, float]:
    md = msg.info
    w, h = int(md.width), int(md.height)
    res = float(md.resolution)
    ox, oy = md.origin.position.x, md.origin.position.y
    q = md.origin.orientation
    _, _, oyaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
    data = np.asarray(msg.data, dtype=np.uint8).reshape((h, w))
    log.debug(f"Costmap array: {w}x{h}, res={res:.3f}, frame={msg.header.frame_id}")
    return data, res, ox, oy, oyaw

def occupancy_to_array(og: OccupancyGrid) -> Tuple[NDArray[np.uint8], float, float, float, float]:
    info = og.info
    w, h = int(info.width), int(info.height)
    res = float(info.resolution)
    ox, oy = info.origin.position.x, info.origin.position.y
    q = info.origin.orientation
    _, _, oyaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
    arr = np.asarray(og.data, dtype=np.int8).reshape((h, w))
    cost = np.zeros_like(arr, dtype=np.uint8)
    cost[arr == -1] = 255
    cost[arr >= 65] = 254
    log.debug(f"Occupancy array: {w}x{h}, res={res:.3f}, frame={og.header.frame_id}")
    return cost, res, ox, oy, oyaw

def build_free_mask(cost: NDArray[np.uint8], lethal: int, inscribed: int,
                    unknown: int, unknown_as_obstacle: bool, disallow_inscribed: bool
                    ) -> NDArray[np.bool_]:
    occ = cost >= lethal
    if unknown_as_obstacle:
        occ |= (cost == unknown)
    if disallow_inscribed:
        occ |= (cost >= inscribed)
    free = ~occ
    log.debug(f"Free mask computed: free_ratio={free.mean():.3f}")
    return free
