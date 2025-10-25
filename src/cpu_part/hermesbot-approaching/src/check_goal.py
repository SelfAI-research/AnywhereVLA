#!/usr/bin/env python3
"""
Send a ComputePathToPose goal and print all feedback + the final result.

Usage:
  python3 send_compute_path.py  <x> <y> <w>  [--frame FRAME] [--server /compute_path_to_pose] [--timeout 5.0]

Notes:
  - <w> is the quaternion w (z=0), so w=1.0 => yaw=0.
  - Set --frame to your planner's global frame (often 'map' or 'camera_init').
"""

import sys
import argparse
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose


class PathClient(Node):
    def __init__(self, server: str) -> None:
        super().__init__("path_client")
        self.cli = ActionClient(self, ComputePathToPose, server)

    def send(self, x: float, y: float, w: float, frame: str, timeout: float) -> None:
        if not self.cli.wait_for_server(timeout_sec=timeout):
            self.get_logger().error(f"Action server '{self.cli._action_name}' not available")
            rclpy.shutdown()
            return

        goal = ComputePathToPose.Goal()
        goal.goal = PoseStamped()
        goal.goal.header.frame_id = frame
        goal.goal.pose.position.x = x
        goal.goal.pose.position.y = y
        goal.goal.pose.orientation.z = 0.0
        goal.goal.pose.orientation.w = w

        self.get_logger().info(f"Sending goal → frame='{frame}', x={x:.3f}, y={y:.3f}, w={w:.3f}")
        send_future = self.cli.send_goal_async(goal, feedback_callback=self._on_feedback)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, fut) -> None:
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().warn("Goal rejected")
            rclpy.shutdown()
            return
        self.get_logger().info("Goal accepted")
        res_future = gh.get_result_async()
        res_future.add_done_callback(self._on_result)

    def _on_feedback(self, fb_msg) -> None:
        # Feedback type can vary by plugin; print as-is to see all fields
        self.get_logger().info(f"Feedback: {fb_msg.feedback}")

    def _on_result(self, fut) -> None:
        try:
            res = fut.result().result
        except Exception as e:
            self.get_logger().error(f"Result error: {e}")
            rclpy.shutdown()
            return
        poses = len(res.path.poses) if res and res.path else 0
        self.get_logger().info(f"Result: error_code={getattr(res, 'error_code', 'N/A')}, path_poses={poses}")
        if poses:
            p0 = res.path.poses[0].pose.position
            pN = res.path.poses[-1].pose.position
            self.get_logger().info(f"Path start=({p0.x:.3f},{p0.y:.3f}) end=({pN.x:.3f},{pN.y:.3f})")
        rclpy.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="Send Nav2 ComputePathToPose goal and stream feedback.")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("w", type=float, help="Quaternion w (z=0). Use 1.0 for yaw=0.")
    ap.add_argument("--frame", default="camera_init", help="Goal frame_id (e.g., 'map' or 'camera_init').")
    ap.add_argument("--server", default="/compute_path_to_pose", help="Action server name.")
    ap.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for server.")
    args = ap.parse_args()

    rclpy.init()
    node = PathClient(args.server)
    node.send(args.x, args.y, args.w, args.frame, args.timeout)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
