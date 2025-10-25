import math, time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose


class Nav2Client:
    """Thin wrapper around Nav2 actions with a path-feasibility check."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._cbg = ReentrantCallbackGroup()  # allow callbacks while inside other callbacks
        self._compute = ActionClient(node, ComputePathToPose, "compute_path_to_pose", callback_group=self._cbg)
        self._navigate = ActionClient(node, NavigateToPose, "navigate_to_pose", callback_group=self._cbg)
        self._nav_goal_handle = None
        self._map_frame = str(node.cfg.frames.map)
        self.timeout = 3.0

    def is_busy(self) -> bool:
        return self._nav_goal_handle is not None

    def _wait_server(self, client: ActionClient, name: str, total_timeout: float = 5.0) -> bool:
        start = time.time()
        while not client.wait_for_server(timeout_sec=0.25):
            if time.time() - start > total_timeout:
                self._node.get_logger().warn(f"Timeout waiting for '{name}' action server.")
                return False
        return True

    def _spin_until(self, fut, timeout_sec: float) -> bool:
        end = time.time() + timeout_sec
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self._node, timeout_sec=0.05)
            if time.time() > end:
                return False
        return True

    def check_path_feasible(self, robot_xy: tuple[float, float], goal_xy: tuple[float, float]) -> tuple[bool, float]:
        """Returns (is_feasible, path_length_m)."""
        if not self._wait_server(self._compute, "compute_path_to_pose"):
            return False, math.inf

        g = ComputePathToPose.Goal()

        g.goal = PoseStamped()
        g.goal.header.frame_id = self._map_frame
        g.goal.header.stamp = self._node.get_clock().now().to_msg()
        g.goal.pose.position.x = float(goal_xy[0])
        g.goal.pose.position.y = float(goal_xy[1])
        g.goal.pose.orientation.w = 1.0

        g.start = PoseStamped()
        g.start.header.frame_id = self._map_frame
        g.start.header.stamp = self._node.get_clock().now().to_msg()
        g.start.pose.position.x = float(robot_xy[0])
        g.start.pose.position.y = float(robot_xy[1])
        g.start.pose.orientation.w = 1.0
        g.use_start = True

        fut = self._compute.send_goal_async(g)
        if not self._spin_until(fut, self.timeout):
            self._node.get_logger().warn("Planner goal acceptance timed out.")
            return False, math.inf

        goal_handle = fut.result()
        if not goal_handle or not goal_handle.accepted:
            return False, math.inf

        res_fut = goal_handle.get_result_async()
        if not self._spin_until(res_fut, self.timeout):
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._node.get_logger().warn("Planner result timed out; canceled.")
            return False, math.inf

        rr = res_fut.result()
        res = getattr(rr, "result", rr)
        path = getattr(res, "path", None)
        poses = path.poses if (path and hasattr(path, "poses")) else []
        if not poses:
            return False, math.inf

        total = 0.0
        for a, b in zip(poses[:-1], poses[1:]):
            dx = b.pose.position.x - a.pose.position.x
            dy = b.pose.position.y - a.pose.position.y
            total += math.hypot(dx, dy)
        return True, total

    def navigate_to(self, goal: PoseStamped) -> None:
        if not self._wait_server(self._navigate, "navigate_to_pose"):
            self._node.get_logger().warn("navigate_to_pose server not ready.")
            return

        fut = self._navigate.send_goal_async(NavigateToPose.Goal(pose=goal))
        if not self._spin_until(fut, self.timeout):
            self._node.get_logger().warn("Navigate goal acceptance timed out.")
            return

        self._nav_goal_handle = fut.result()
        if not self._nav_goal_handle or not self._nav_goal_handle.accepted:
            self._node.get_logger().warn("NavigateToPose goal not accepted.")
            self._nav_goal_handle = None
            return

        res_fut = self._nav_goal_handle.get_result_async()

        def _done_cb(_):
            try:
                st = res_fut.result().status
                self._node.get_logger().info(f"NavigateToPose finished with status={st}")
            except Exception as e:
                self._node.get_logger().warn(f"Navigate result exception: {e}")
            finally:
                self._nav_goal_handle = None

        res_fut.add_done_callback(_done_cb)

    def cancel_navigation(self) -> None:
        if self._nav_goal_handle:
            cancel_future = self._nav_goal_handle.cancel_goal_async()
            self._spin_until(cancel_future, 2.0)
            self._nav_goal_handle = None
