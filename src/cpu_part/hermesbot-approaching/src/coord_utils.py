"""Coordinate helpers with rotated origins."""
import math
from typing import Tuple
from geometry_msgs.msg import PoseStamped, Quaternion

def world_to_map(x: float, y: float, ox: float, oy: float, oyaw: float, res: float) -> Tuple[int,int]:
    dx, dy = x - ox, y - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    lx, ly = c*dx - s*dy, s*dx + c*dy
    return int(lx / res), int(ly / res)

def map_to_world(i: int, j: int, ox: float, oy: float, oyaw: float, res: float) -> Tuple[float,float]:
    lx, ly = (i + 0.5) * res, (j + 0.5) * res
    c, s = math.cos(oyaw), math.sin(oyaw)
    return ox + c*lx - s*ly, oy + s*lx + c*ly

def yaw_to_quat(yaw: float) -> Quaternion:
    qz, qw = math.sin(yaw * 0.5), math.cos(yaw * 0.5)
    return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

def pose_xyyaw(x: float, y: float, yaw: float, frame: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x, ps.pose.position.y = x, y
    ps.pose.orientation = yaw_to_quat(yaw)
    return ps
