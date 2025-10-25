"""Nav2 ComputePathToPose feasibility checker with logging (Option A: non-deadlocking wait)."""
from typing import Optional
import time

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose

class PlannerChecker:
    def __init__(
        self,
        node,
        action_name: str,
        timeout_sec: float,
        allow_partial: bool,
        callback_group: Optional[ReentrantCallbackGroup] = None,
    ) -> None:
        self.node = node
        self.timeout = float(timeout_sec)
        self.allow_partial = bool(allow_partial)
        self.cb_group = callback_group
        self.action = ActionClient(
            node, ComputePathToPose, action_name,
            callback_group=self.cb_group
        )

    def check_action(self, goal: PoseStamped, start: Optional[PoseStamped]) -> bool:
        lg = self.node.get_logger()
        lg.info("Planner check: start")

        if not self.action.wait_for_server(timeout_sec=0.2):
            lg.warn("Planner action server not available")
            return False

        g = ComputePathToPose.Goal()
        g.goal = goal
        if start is not None:
            g.start = start
            g.use_start = True

        # Send goal, poll the future with a tight sleep; another executor thread keeps spinning.
        gh_fut = self.action.send_goal_async(g)
        t0 = time.time()
        while not gh_fut.done():
            if time.time() - t0 > self.timeout:
                lg.warn("Planner action send_goal timeout")
                return False
            time.sleep(0.005)

        gh = gh_fut.result()
        if not gh or not getattr(gh, "accepted", False):
            lg.info("Planner action rejected goal")
            return False

        res_fut = gh.get_result_async()
        t0 = time.time()
        while not res_fut.done():
            if time.time() - t0 > self.timeout:
                lg.warn("Planner action result timeout")
                return False
            time.sleep(0.005)

        # rclpy may return either:
        #   (A) a wrapper with fields {status, result}, or
        #   (B) the result message directly.
        rr = res_fut.result()
        res = getattr(rr, "result", rr)           # handle both cases
        status = getattr(rr, "status", None)      # optional

        # ComputePathToPose.Result commonly has at least 'path'
        path = getattr(res, "path", None)
        num_poses = len(path.poses) if (path and hasattr(path, "poses")) else 0

        need = 1 if self.allow_partial else 2
        ok = (num_poses >= need)

        if status is not None:
            lg.info(f"Planner action result: ok={ok}, poses={num_poses}, status={status}")
        else:
            lg.info(f"Planner action result: ok={ok}, poses={num_poses}")

        return ok
