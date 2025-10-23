import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray

from tf2_ros import Buffer, TransformListener

from utils import Transform, load_cfg, quat_from_yaw
from frontier import FrontierFinder, FrontierResult
from nav2_iface import Nav2Client


class ExplorerNode(Node):
    """Explores an area by iterating over frontiers until detector reports True or space is fully explored."""

    def __init__(self, cfg_path: str) -> None:
        super().__init__("explorer_supervisor")
        self.cfg = load_cfg(cfg_path)

        # Modules
        self.frontier = FrontierFinder(
            occ_threshold_free=int(self.cfg.frontier.free_threshold),
            min_cluster_px=int(self.cfg.frontier.min_cluster_px),
            max_chunk_px=int(self.cfg.frontier.max_chunk_px),
            min_goal_separation_m=float(self.cfg.frontier.min_goal_separation_m),
            offset_m=float(self.cfg.frontier.offset_distance),
            max_candidates=int(self.cfg.frontier.max_candidates),
            angle_samples=int(self.cfg.frontier.angle_samples),
            gain_stride=int(self.cfg.frontier.gain_stride),
        )
        self.nav2 = Nav2Client(self)

        # TF
        self.tf_buffer = Buffer(cache_time=Duration(seconds=float(self.cfg.tf.cache_sec)))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.map_sub = self.create_subscription(OccupancyGrid, self.cfg.input.map_topic, self.map_callback, qos)
        self.det_sub = self.create_subscription(Bool, self.cfg.input.detector_topic, self.det_callback, 1)

        # External exploration enable/disable
        self._enabled: bool = False
        self.enable_sub = self.create_subscription(Bool, 'exploration/enable', self.enable_callback, 1)

        # Publishers
        self.pub_markers = self.create_publisher(MarkerArray, self.cfg.output.candidate_markers_topic, 1)
        self.pub_stop = self.create_publisher(Bool, self.cfg.output.stop_message_topic, 1)

        # State
        self._last_pose: Transform | None = None
        self._last_pose_time = Time(seconds=0)
        self._detected: bool = False
        self._active_goal_xy: tuple[float, float] | None = None

        # Periodic replan trigger. Timer only sets a flag; we cancel inside map_callback.
        self._replan_requested = False
        self._replan_period_sec = self.cfg.navigation.update_goal_period
        self._replan_timer = self.create_timer(self._replan_period_sec, self._on_replan_timer)

        self.throttling = 0

        if bool(self.cfg.logging.debug):
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.get_logger().info("ExplorerNode up.")

    def _on_replan_timer(self) -> None:
        # Set a flag; the actual cancel happens in map_callback to avoid nested spinning in timer context.
        self._replan_requested = True

    def _lookup_position(self, target: str, source: str, stamp: Time) -> Transform | None:
        """Get T(target<-source) with sensible fallbacks."""
        wait_dur = Duration(seconds=float(self.cfg.tf.wait_sec))
        max_stale = Duration(seconds=float(self.cfg.tf.max_staleness_sec))
        reuse_for = Duration(seconds=float(self.cfg.tf.reuse_last_sec))

        # 1) Exact at cloud stamp (short wait)
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

    def _stop_exploration(self, reason: str) -> None:
        self.get_logger().info(f"Exploration stop: {reason}")
        self.nav2.cancel_navigation()
        msg = Bool()
        msg.data = True
        self.pub_stop.publish(msg)

    def enable_callback(self, msg: Bool) -> None:
        new_state = bool(msg.data)
        if new_state == self._enabled:
            return
        self._enabled = new_state
        if not self._enabled:
            # Stop immediately on disable
            self._active_goal_xy = None
            self._stop_exploration("disabled_by_topic")
        else:
            self.get_logger().info("Exploration enabled by topic.")

    def det_callback(self, det_msg: Bool) -> None:
        self._detected = bool(det_msg.data)
        if self._detected:
            self._stop_exploration("target_detected")

    def map_callback(self, map_msg: OccupancyGrid) -> None:
        t0 = time.perf_counter()

        # Gate navigation by external enable flag
        if not self._enabled:
            return None

        if self.throttling < 5:
            self.throttling += 1
            return
        self.throttling = 0

        # Periodic replan: if requested and navigating, cancel current goal to refresh.
        if self._replan_requested and not self._detected:
            self._replan_requested = False
            if self.nav2.is_busy():
                self.get_logger().debug("Periodic replan: canceling current NavigateToPose goal.")
                self.nav2.cancel_navigation()
                self._active_goal_xy = None  # force re-select

        # Pose: we need base in map -> T(map<-base)
        map_time = Time().from_msg(map_msg.header.stamp)
        pose_tf = self._lookup_position(
            target=self.cfg.frames.map,
            source=self.cfg.frames.base,
            stamp=map_time,
        )
        if not pose_tf:
            return

        # Frontier extraction
        t1 = time.perf_counter()
        fr: FrontierResult = self.frontier.find(
            map_msg=map_msg,
            explore_range=self.cfg.frontier.explore_range,
            robot_xy=pose_tf.translation[:2],
            robot_yaw=pose_tf.yaw(),
            camera_fov=self.cfg.camera.fov_deg,
            camera_range=self.cfg.camera.max_range_m,
        )
        self.get_logger().debug(f"Found {len(fr.world_points)} goal points:")
        for point in fr.world_points:
            self.get_logger().debug(f" - ({point[0]}), ({point[1]})")

        # Decide goal (only if not detected and not already navigating)
        t2 = time.perf_counter()
        if self._detected:
            pass
        elif fr.world_points:
            pos_xy = pose_tf.translation[:2]
            chosen = self._choose_and_send_goal(fr.world_points, fr.yaws, pos_xy, prefer_shortest=True)
            self._active_goal_xy = chosen
        else:
            # No frontiers -> fully explored (within bounds)
            self._stop_exploration("fully_explored")

        # Publish markers
        t3 = time.perf_counter()
        self.pub_markers.publish(fr.to_markers_msg(frame_id=map_msg.header.frame_id))

        # print timings
        t3 = time.perf_counter()
        self.get_logger().debug(
            f"Total: {1000 * (t3 - t0):.1f} ms | "
            f"read_pose: {1000 * (t1 - t0):.1f} ms | "
            f"frontier: {1000 * (t2 - t1):.1f} ms | "
            f"set goal: {1000 * (t3 - t2):.1f} ms"
        )

    def _choose_and_send_goal(
        self,
        candidates_xy: list[tuple[float, float]],
        candidate_yaws: list[float],
        robot_xy: np.ndarray | tuple[float, float],
        prefer_shortest: bool,
    ) -> tuple[float, float] | None:
        """Pick first candidate with a feasible path. Prefers shortest path length. Uses FoV-optimized yaw."""
        if self.nav2.is_busy():
            return self._active_goal_xy

        # Order by heuristic (euclidean distance)
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        paired = list(zip(candidates_xy, candidate_yaws))
        ordered = sorted(paired, key=lambda p: np.hypot(p[0][0] - rx, p[0][1] - ry))

        best_xy: tuple[float, float] | None = None
        best_yaw: float | None = None
        best_len = float("inf")

        for (gx, gy), yaw in ordered:
            ok, path_len = self.nav2.check_path_feasible((rx, ry), (gx, gy))
            if ok:
                if prefer_shortest:
                    if path_len < best_len:
                        best_xy, best_yaw, best_len = (gx, gy), float(yaw), path_len
                else:
                    best_xy, best_yaw = (gx, gy), float(yaw)
                    break

        if best_xy is not None:
            q = quat_from_yaw(best_yaw)  # [x,y,z,w]
            goal = PoseStamped()
            goal.header.frame_id = self.cfg.frames.map
            goal.pose.position.x = best_xy[0]
            goal.pose.position.y = best_xy[1]
            goal.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
            self.nav2.navigate_to(goal)
        else:
            self.get_logger().info("No reachable frontier found.")

        return best_xy


def main() -> None:
    rclpy.init()
    node = ExplorerNode(cfg_path="configs/config.yaml")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
