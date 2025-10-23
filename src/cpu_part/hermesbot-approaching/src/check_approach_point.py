#!/usr/bin/env python3
"""
Publish one object point, take the first /goal_pose, send ONE ComputePathToPose action,
and print whether the action goal was ACCEPTED or REJECTED.

Usage:
  python3 send_one_action_status.py <x> <y> <z> [--obj-frame camera_init]
                                    [--object-topic /approaching_object_point]
                                    [--goal-topic /goal_pose]
                                    [--server /compute_path_to_pose]
                                    [--wait-goal 30.0] [--timeout 5.0]
"""

import sys
import argparse
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import ComputePathToPose


def make_point(frame: str, x: float, y: float, z: float) -> PointStamped:
    msg = PointStamped()
    msg.header.frame_id = frame
    msg.header.stamp = Time(sec=0, nanosec=0)  # "latest" TF semantics in many stacks
    msg.point.x, msg.point.y, msg.point.z = x, y, z
    return msg


class OneShotStatus(Node):
    """Publishes object, waits for goal_pose, sends ONE action goal, prints ACCEPTED/REJECTED."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("one_shot_action_status")
        self.args = args
        self.sent = False
        self.goal_seen = False

        self.obj_pub = self.create_publisher(PointStamped, args.object_topic, 1)
        self.goal_sub = self.create_subscription(PoseStamped, args.goal_topic, self._on_goal, 10)
        self.client = ActionClient(self, ComputePathToPose, args.server)

        self.obj_pub.publish(make_point(args.obj_frame, args.x, args.y, args.z))
        self.get_logger().info(
            f"Published object to {args.object_topic} in '{args.obj_frame}': "
            f"({args.x:.3f}, {args.y:.3f}, {args.z:.3f})"
        )

        self.guard = self.create_timer(float(args.wait_goal), self._timeout_no_goal)

    # --- callbacks

    def _on_goal(self, msg: PoseStamped) -> None:
        if self.sent:
            return
        self.goal_seen = True
        self.guard.cancel()
        self.get_logger().info(
            f"Got /goal_pose (frame={msg.header.frame_id}). Sending ONE action request..."
        )

        if not self.client.wait_for_server(timeout_sec=self.args.timeout):
            print("STATUS: SERVER_UNAVAILABLE")
            self._quit()
            return

        goal = ComputePathToPose.Goal()
        goal.goal = msg

        fut = self.client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)
        self.sent = True

    def _on_goal_response(self, future) -> None:
        try:
            gh = future.result()
        except Exception as e:
            self.get_logger().error(f"Send goal error: {e}")
            print("STATUS: ERROR")
            self._quit()
            return
        if not gh or not gh.accepted:
            print("STATUS: REJECTED")
        else:
            print("STATUS: ACCEPTED")
        self._quit()

    def _timeout_no_goal(self) -> None:
        if self.goal_seen:
            return
        self.get_logger().error(
            f"Timed out waiting for {self.args.goal_topic}. Is the approach planner publishing?"
        )
        print("STATUS: NO_GOAL")
        self._quit()

    def _quit(self) -> None:
        
        self.destroy_node()
        rclpy.shutdown()
        exit(0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Send ONE ComputePathToPose and print ACCEPTED/REJECTED.")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("z", type=float)
    ap.add_argument("--obj-frame", default="camera_init")
    ap.add_argument("--object-topic", default="/approaching_object_point")
    ap.add_argument("--goal-topic", default="/goal_pose")
    ap.add_argument("--server", default="/compute_path_to_pose")
    ap.add_argument("--wait-goal", type=float, default=10.0, help="Seconds to wait for /goal_pose.")
    ap.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for action server.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = OneShotStatus(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
