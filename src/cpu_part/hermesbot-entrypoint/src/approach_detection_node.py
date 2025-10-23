"""
Approach Arrival Monitor:
- Subscribes:  /goal_pose (geometry_msgs/PoseStamped)
- Checks TF:   T(goal_frame <- base_link)
- Publishes:   /approaching/done (std_msgs/Bool) when distance <= threshold_m
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import Bool as BoolMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R

from tf2_ros import Buffer, TransformListener


# --------------------------- Config & Helpers ---------------------------

@dataclass
class TFConfig:
    wait_sec: float = 0.02
    max_staleness_sec: float = 0.20
    reuse_last_sec: float = 0.50

@dataclass
class Config:
    goal_topic: str = "/goal_pose"
    done_topic: str = "/approaching/done"
    world_frame: str = "camera_init"
    robot_frame: str = "base_link"
    check_rate_hz: float = 10.0
    threshold_m: float = 0.5
    tf: TFConfig = field(default_factory=TFConfig)

@dataclass
class Transform:
    translation: np.ndarray  # shape (3,)
    rotation: np.ndarray     # shape (3,3)

    @property
    def x(self) -> float: return float(self.translation[0])
    @property
    def y(self) -> float: return float(self.translation[1])
    @property
    def z(self) -> float: return float(self.translation[2])

    def to_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    def yaw(self) -> float:
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

    @classmethod
    def from_msg(cls, msg: TransformStamped) -> "Transform":
        # extract translation
        t = msg.transform.translation
        translation = np.array([t.x, t.y, t.z], dtype=float)
    
        # extract quaternion and convert to rotation matrix
        q = msg.transform.rotation
        quat = [q.x, q.y, q.z, q.w]
        rotation = R.from_quat(quat).as_matrix()

        return cls(translation=translation, rotation=rotation)



# ------------------------------ Node ------------------------------

class ApproachArrivalMonitor(Node):
    """Publishes /approaching/done when base_link is within threshold of /goal_pose."""

    def __init__(self) -> None:
        super().__init__("approach_arrival_monitor")
        self.cfg = Config()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_done = self.create_publisher(BoolMsg, self.cfg.done_topic, 1)
        self.sub_goal = self.create_subscription(PoseStamped, self.cfg.goal_topic, self._on_goal, 10)

        self.timer = self.create_timer(1.0 / self.cfg.check_rate_hz, self._on_tick)

        self._goal: Optional[PoseStamped] = None
        self._armed: bool = False
        self._last_pose: Optional[Transform] = None
        self._last_pose_time: Time = Time()

        self.get_logger().info(
            f"Monitoring {self.cfg.goal_topic} (frame={self.cfg.world_frame}) → publish {self.cfg.done_topic} "
            f"when distance ≤ {self.cfg.threshold_m:.2f} m."
        )

    # --------------------------- TF lookup (your logic) ---------------------------

    def _lookup_position(self, target: str, source: str, stamp: Time) -> Transform | None:
        """Get T(target<-source) with sensible fallbacks."""
        wait_dur = Duration(seconds=float(self.cfg.tf.wait_sec))
        max_stale = Duration(seconds=float(self.cfg.tf.max_staleness_sec))
        reuse_for = Duration(seconds=float(self.cfg.tf.reuse_last_sec))

        # 1) Exact at goal stamp (short wait)
        try:
            t = self.tf_buffer.lookup_transform(target, source, stamp, timeout=wait_dur)
            tf_time = Time.from_msg(t.header.stamp)
            self._last_pose = Transform.from_msg(t)
            self._last_pose_time = stamp
            self.get_logger().debug(f"[Exact lookup] ok, diff={(stamp - tf_time).nanoseconds*1e-9:.3f}s")
            return self._last_pose
        except Exception as e:
            self.get_logger().debug(f"[Exact lookup] {e}")

        # 2) Latest TF, accept if not too stale
        try:
            t = self.tf_buffer.lookup_transform(target, source, Time())  # latest
            tf_time = Time.from_msg(t.header.stamp)
            diff = stamp - tf_time
            if diff <= max_stale:
                self._last_pose = Transform.from_msg(t)
                self._last_pose_time = stamp
                self.get_logger().debug(f"[Latest ok] diff={diff.nanoseconds*1e-9:.3f}s")
                return self._last_pose
        except Exception as e:
            self.get_logger().debug(f"[Latest lookup] {e}")

        # 3) Reuse last good briefly
        if self._last_pose:
            diff = stamp - self._last_pose_time
            if diff <= reuse_for:
                self.get_logger().warn(f"TF missing; reusing last pose, age={diff.nanoseconds*1e-9:.3f}s")
                return self._last_pose

        self.get_logger().warn("TF missing; no reusable pose.")
        return None

    # --------------------------- Callbacks ---------------------------

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._armed = True
        self.get_logger().info(
            f"New goal: frame={msg.header.frame_id} pos=({msg.pose.position.x:.3f},{msg.pose.position.y:.3f})"
        )

    def _on_tick(self) -> None:
        if not (self._armed and self._goal):
            return

        goal = self._goal
        goal_frame = goal.header.frame_id or self.cfg.world_frame

        tf = self._lookup_position(goal_frame, self.cfg.robot_frame, Time.from_msg(goal.header.stamp))
        if tf is None:
            return

        dx = tf.x - goal.pose.position.x
        dy = tf.y - goal.pose.position.y
        dist = math.hypot(dx, dy)

        self.get_logger().debug(f"Distance to goal: {dist:.3f} m")
        if dist <= self.cfg.threshold_m:
            self.pub_done.publish(BoolMsg(data=True))
            self.get_logger().info(f"Arrived within {self.cfg.threshold_m:.2f} m → published {self.cfg.done_topic}")
            self._armed = False  # arm again on next goal


# ------------------------------ Main ------------------------------

def main() -> None:
    rclpy.init()
    node = ApproachArrivalMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
