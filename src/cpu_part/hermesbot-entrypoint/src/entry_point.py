"""
Main Interpreter Node for SLAM-VLA integration (ROS 2 Humble).

Pipeline (per object):
  0) Enable Semantic Map (Bool)
  1) Read MarkerArray of detected objects & poses
  2) If not found: enable Exploration (Bool, first time), sleep, re-check
  3) If found: take first matching pose
  4) Disable Exploration (Bool)
  5) Publish Approach goal (PoseStamped)
  6) Wait for Approach "done" (Bool)
  7) Disable Semantic Map (Bool)
  8) Publish VLA prompt (String)
  9) Wait for VLA "done" (Bool)
 10) After all objects: publish Nav2 goal (PoseStamped) if configured

Input command format:
- String topic whose data is a JSON array.
  Each element can be either:
    {"object": "...", "prompt": "..."}    # preferred
  or {"cup": "pick up the cup"}           # legacy single-key form
"""

import json
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import String as StringMsg, Bool as BoolMsg
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, PoseStamped


# ------------------------------- Data Models ---------------------------------


@dataclass(frozen=True)
class TaskItem:
    """One VLA task for a specific object."""
    object_name: str
    prompt: str

@dataclass
class InterpreterConfig:
    """
    Runtime configuration for the interpreter. All values live here to allow
    quick tuning.
    """
    # Topics
    input_command_topic: str
    semantic_map_enable_topic: str
    semantic_map_markers_topic: str
    exploration_enable_topic: str
    approach_goal_topic: str
    approach_done_topic: str
    vla_prompt_topic: str
    vla_done_topic: str
    nav_goal_topic: str

    # Frames
    world_frame_id: str

    # Matching behavior against Marker fields
    # Any of {"ns", "text", "id"}; checked in given order.
    marker_match_fields: list[str]

    # Timing (seconds)
    search_timeout_sec: float
    exploration_check_interval_sec: float
    approach_timeout_sec: float
    vla_timeout_sec: float

    # Optional final NAV goal after all tasks
    publish_final_nav_goal: bool
    final_goal_xyz: tuple[float, float, float]          # (x, y, z)
    final_goal_xyzw: tuple[float, float, float, float]  # (qx, qy, qz, qw)


class Step(IntEnum):
    ENABLE_SEMANTIC = 0
    READ_MARKERS = 1
    ENABLE_EXPLORATION_IF_NEEDED = 2
    SELECT_FIRST_MATCH = 3
    DISABLE_EXPLORATION = 4
    PUBLISH_APPROACH_GOAL = 5
    WAIT_APPROACH_DONE = 6
    DISABLE_SEMANTIC = 7
    PUBLISH_VLA_PROMPT = 8
    WAIT_VLA_DONE = 9
    FINAL_NAV_GOAL = 10

# ------------------------------- Main  Node ----------------------------------


class MainInterpreterNode(Node):
    """
    Orchestrates SLAM-VLA flow:
    - Consumes structured text commands
    - Coordinates Semantic Map, Exploration, Approach, VLA, and final Nav goal
    """

    def __init__(self) -> None:
        super().__init__("slam_vla_main_interpreter")

        # =========================== CONFIGURATION ============================
        # Declare all parameters here so you can flip them in one place.
        self.cfg = InterpreterConfig(
            # Topics
            input_command_topic="/interpreter/commands",  # Done
            semantic_map_enable_topic="/semantic_map/enable",  # Done
            semantic_map_markers_topic="/markers_topic",  # Done
            exploration_enable_topic="/exploration/enable",  # Done
            approach_goal_topic="/approaching_object_point",  # Done
            approach_done_topic="/approaching/done",  # Done
            vla_prompt_topic="/VLA_start_control",  # Done
            vla_done_topic="/vla/done",
            nav_goal_topic="/goal_pose",  # Done

            # Frames
            world_frame_id="map",

            # Matching behavior against Marker fields
            # Any of {"ns", "text", "id"}; checked in given order.
            marker_match_fields=["ns", "text", "id"],

            # Timing (seconds)
            search_timeout_sec=120.0,
            exploration_check_interval_sec=2.0,
            approach_timeout_sec=60.0,
            vla_timeout_sec=120.0,

            # Optional final NAV goal after all tasks
            publish_final_nav_goal=True,
            final_goal_xyz=(0.0, 0.0, 0.0),
            final_goal_xyzw=(0.0, 0.0, 0.0, 1.0)
        )
        # =====================================================================

        # Publishers
        self._pub_semantic_enable = self.create_publisher(BoolMsg, self.cfg.semantic_map_enable_topic, 1)
        self._pub_exploration_enable = self.create_publisher(BoolMsg, self.cfg.exploration_enable_topic, 1)
        self._pub_approach_goal = self.create_publisher(PoseStamped, self.cfg.approach_goal_topic, 1)
        self._pub_vla_prompt = self.create_publisher(StringMsg, self.cfg.vla_prompt_topic, 1)
        self._pub_nav_goal = self.create_publisher(PoseStamped, self.cfg.nav_goal_topic, 1)

        # Subscriptions
        self.create_subscription(StringMsg, self.cfg.input_command_topic, self._on_command, 10)

        # MarkerArray subscriber (default QoS here)
        markers_qos = QoSProfile(depth=10)
        self.create_subscription(MarkerArray, self.cfg.semantic_map_markers_topic, self._on_markers, markers_qos)

        self.create_subscription(BoolMsg, self.cfg.approach_done_topic, self._on_approach_done, 10)
        self.create_subscription(BoolMsg, self.cfg.vla_done_topic, self._on_vla_done, 10)

        # State
        self._last_markers: Optional[MarkerArray] = None
        self._approach_done = threading.Event()
        self._vla_done = threading.Event()
        self._worker_lock = threading.Lock()  # ensure one worker at a time

        # Startup logs
        self.get_logger().info("Main Interpreter node initialized.")
        self._log_cfg_summary()
        self.get_logger().info(f"Waiting for commands on: {self.cfg.input_command_topic}")

    # ------------------------------ Callbacks --------------------------------

    def _on_command(self, msg: StringMsg) -> None:
        """Handle incoming JSON array of tasks and kick off a worker thread."""
        try:
            tasks = self._parse_tasks(msg.data)
        except Exception as exc:
            self.get_logger().error(f"[SYS] Failed to parse command JSON: {exc}")
            return

        self.get_logger().info(f"[SYS] Received {len(tasks)} task(s): {tasks}")

        if not tasks:
            self.get_logger().warn("[SYS] Empty task list. Ignoring.")
            return

        if not self._worker_lock.acquire(blocking=False):
            self.get_logger().warn("[SYS] Interpreter is busy. Ignoring new command.")
            return

        def _runner() -> None:
            try:
                self._process_tasks(tasks)
            finally:
                self._worker_lock.release()

        threading.Thread(target=_runner, daemon=True).start()

    def _on_markers(self, msg: MarkerArray) -> None:
        self._last_markers = msg
        self.get_logger().debug(f"[S{Step.READ_MARKERS}] Updated markers: count={len(msg.markers)}")

    def _on_approach_done(self, msg: BoolMsg) -> None:
        if msg.data:
            self._approach_done.set()
            self.get_logger().info(f"[S{Step.WAIT_APPROACH_DONE}] Approach DONE signal received.")

    def _on_vla_done(self, msg: BoolMsg) -> None:
        if msg.data:
            self._vla_done.set()
            self.get_logger().info(f"[S{Step.WAIT_VLA_DONE}] VLA DONE signal received.")

    # ----------------------------- Core Logic --------------------------------

    def _process_tasks(self, tasks: list[TaskItem]) -> None:
        self.get_logger().info(f"[SYS] Starting task batch with {len(tasks)} item(s).")

        exploration_enabled = False

        for idx, task in enumerate(tasks, start=1):
            hdr = f"[T{idx}/{len(tasks)}]"
            self.get_logger().info(f"{hdr} Object='{task.object_name}', prompt='{task.prompt}'")

            # S0) Enable Semantic Map
            self._log_step(Step.ENABLE_SEMANTIC, hdr, "Enabling Semantic Map...")
            self._publish_bool(self._pub_semantic_enable, True)

            # S1–S2) Search loop with optional Exploration enable
            pose: Optional[Pose] = None
            start_time = time.monotonic()
            attempts = 0

            while True:
                attempts += 1
                pose = self._find_object_pose(task.object_name, self._last_markers, hdr)
                if pose is not None:
                    break

                elapsed = time.monotonic() - start_time
                if elapsed > self.cfg.search_timeout_sec:
                    self.get_logger().warn(f"{hdr} [S{Step.READ_MARKERS}] Timeout ({elapsed:.1f}s) searching for '{task.object_name}'. Skipping.")
                    break

                if not exploration_enabled:
                    self._log_step(Step.ENABLE_EXPLORATION_IF_NEEDED, hdr, "Enabling Exploration...")
                    self._publish_bool(self._pub_exploration_enable, True)
                    exploration_enabled = True

                self.get_logger().info(f"{hdr} [S{Step.READ_MARKERS}] Not found yet (attempt {attempts}, t={elapsed:.1f}s). Rechecking in {self.cfg.exploration_check_interval_sec:.1f}s.")
                time.sleep(self.cfg.exploration_check_interval_sec)

            if pose is None:
                self.get_logger().warn(f"{hdr} Could not find object '{task.object_name}'. Moving to next.")
                continue

            self._log_step(Step.SELECT_FIRST_MATCH, hdr, f"Found '{task.object_name}' at pose: {self._pose_str(pose)}")

            # S4) Disable Exploration
            if exploration_enabled:
                self._log_step(Step.DISABLE_EXPLORATION, hdr, "Disabling Exploration...")
                self._publish_bool(self._pub_exploration_enable, False)
                exploration_enabled = False

            # S5) Send Approach goal
            approach_goal = self._pose_to_stamped(pose, frame_id=self.cfg.world_frame_id)
            self._approach_done.clear()
            self._pub_approach_goal.publish(approach_goal)
            self._log_step(Step.PUBLISH_APPROACH_GOAL, hdr, f"Approach goal published (frame={self.cfg.world_frame_id}).")

            # S6) Wait for approach completion
            self._log_step(Step.WAIT_APPROACH_DONE, hdr, f"Waiting up to {self.cfg.approach_timeout_sec:.1f}s for approach DONE...")
            if not self._wait_event(self._approach_done, self.cfg.approach_timeout_sec):
                self.get_logger().warn(f"{hdr} [S{Step.WAIT_APPROACH_DONE}] Approach timeout. Continuing workflow.")

            # S7) Disable Semantic Map
            self._log_step(Step.DISABLE_SEMANTIC, hdr, "Disabling Semantic Map...")
            self._publish_bool(self._pub_semantic_enable, False)

            # S8) Send VLA prompt
            self._vla_done.clear()
            self._pub_vla_prompt.publish(StringMsg(data=task.prompt))
            self._log_step(Step.PUBLISH_VLA_PROMPT, hdr, "VLA prompt published.")

            # S9) Wait for VLA completion
            self._log_step(Step.WAIT_VLA_DONE, hdr, f"Waiting up to {self.cfg.vla_timeout_sec:.1f}s for VLA DONE...")
            if not self._wait_event(self._vla_done, self.cfg.vla_timeout_sec):
                self.get_logger().warn(f"{hdr} [S{Step.WAIT_VLA_DONE}] VLA timeout. Continuing to next task.")

        # S10) Final Nav2 goal (optional)
        if self.cfg.publish_final_nav_goal:
            nav_goal = self._make_stamped_pose(
                xyz=self.cfg.final_goal_xyz,
                xyzw=self.cfg.final_goal_xyzw,
                frame_id=self.cfg.world_frame_id,
            )
            self._pub_nav_goal.publish(nav_goal)
            self._log_step(Step.FINAL_NAV_GOAL, "[FINAL]", f"Final Nav2 goal published: pos={self.cfg.final_goal_xyz}, quat={self.cfg.final_goal_xyzw}.")

        self.get_logger().info("[SYS] Task batch complete.")

    # ---------------------------- Helper Methods -----------------------------

    def _parse_tasks(self, raw: str) -> list[TaskItem]:
        """
        Accepts:
          [{"object": "bottle", "prompt": "pick up the bottle"}, ...]
        or [{"bottle": "pick up the bottle"}, ...]
        """
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be a list.")

        tasks: list[TaskItem] = []
        for i, entry in enumerate(data):
            if isinstance(entry, dict) and "object" in entry and "prompt" in entry:
                obj = str(entry["object"]).strip()
                prm = str(entry["prompt"]).strip()
            elif isinstance(entry, dict) and len(entry) == 1:
                (obj, prm), = entry.items()
                obj = str(obj).strip()
                prm = str(prm).strip()
            else:
                raise ValueError(f"Invalid task at index {i}: {entry!r}")

            if not obj or not prm:
                raise ValueError(f"Empty object or prompt at index {i}")

            tasks.append(TaskItem(object_name=obj, prompt=prm))
        return tasks

    def _find_object_pose(self, object_name: str, markers: Optional[MarkerArray], hdr: str) -> Optional[Pose]:
        """
        Match strategy:
        1) Prefer label text that contains the requested class name (case-insensitive),
            using 'lbl/<cid>' → pose from paired 'obj/<cid>'.
        2) Accept numeric inputs ('39') or 'obj/39' to target a specific class id.
        3) Fallback to exact equality on ns/text/id (legacy behavior).
        """
        count = len(markers.markers) if markers and markers.markers else 0
        self.get_logger().info(f"{hdr} [S{Step.READ_MARKERS}] Looking for '{object_name}'. Semantic map has {count} marker(s).")
        if markers is None or not markers.markers:
            return None

        target = object_name.strip().lower()

        # Build cid → pose (from obj/...) and cid → label text (from lbl/...)
        obj_pose_by_cid: dict[str, Pose] = {}
        label_text_by_cid: dict[str, str] = {}
        for m in markers.markers:
            ns = (m.ns or "")
            if ns.startswith("obj/"):
                cid = ns.split("/", 1)[1]
                obj_pose_by_cid[cid] = m.pose
            elif ns.startswith("lbl/"):
                cid = ns.split("/", 1)[1]
                label_text_by_cid[cid] = (getattr(m, "text", "") or "")

        # Direct class-id addressing: "obj/39" or "39"
        if target.startswith("obj/"):
            cid = target.split("/", 1)[1]
            if cid in obj_pose_by_cid:
                self.get_logger().info(f"{hdr} [S{Step.SELECT_FIRST_MATCH}] Matched by explicit cid 'obj/{cid}'.")
                return obj_pose_by_cid[cid]
        if target.isdigit() and target in obj_pose_by_cid:
            self.get_logger().info(f"{hdr} [S{Step.SELECT_FIRST_MATCH}] Matched by numeric cid '{target}'.")
            return obj_pose_by_cid[target]

        # Label substring match: "bottle" matches "39:bottle #12 (85%)"
        for cid, text in label_text_by_cid.items():
            if target and target in text.lower():
                pose = obj_pose_by_cid.get(cid)
                if pose is not None:
                    self.get_logger().info(f"{hdr} [S{Step.SELECT_FIRST_MATCH}] Matched by label substring: cid={cid}, text='{text}'.")
                    return pose

        # Fallback: exact equals on configured fields (ns/text/id)
        for m in markers.markers:
            for field in self.cfg.marker_match_fields:
                if field == "ns":
                    val = (m.ns or "").lower()
                elif field == "text":
                    val = (getattr(m, "text", "") or "").lower()
                elif field == "id":
                    val = str(m.id).lower()
                else:
                    continue
                if val == target:
                    self.get_logger().info(f"{hdr} [S{Step.SELECT_FIRST_MATCH}] Matched by exact {field} == '{target}'.")
                    return m.pose

        return None

    def _publish_bool(self, pub, value: bool) -> None:
        topic = getattr(pub, "topic_name", "<unknown_topic>")
        self.get_logger().info(f"--- Publishing Bool={value} to '{topic}'")
        pub.publish(BoolMsg(data=value))

    def _pose_to_stamped(self, pose: Pose, frame_id: str) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        return ps

    def _make_stamped_pose(
        self,
        xyz: tuple[float, float, float],
        xyzw: tuple[float, float, float, float],
        frame_id: str,
    ) -> PoseStamped:
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = xyzw
        return self._pose_to_stamped(p, frame_id)

    def _wait_event(self, ev: threading.Event, timeout_sec: float) -> bool:
        return ev.wait(timeout=timeout_sec)

    def _pose_str(self, p: Pose) -> str:
        return (f"(pos=({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f}), "
                f"quat=({p.orientation.x:.3f}, {p.orientation.y:.3f}, {p.orientation.z:.3f}, {p.orientation.w:.3f}))")

    def _log_step(self, step: Step, hdr: str, msg: str) -> None:
        self.get_logger().info(f"{hdr} [S{int(step)}] {msg}")

    def _log_cfg_summary(self) -> None:
        """Print key configuration at startup for quick validation."""
        c = self.cfg
        self.get_logger().info(
            "[CFG] Topics:\n"
            f"  commands:      {c.input_command_topic}\n"
            f"  sem.enable:    {c.semantic_map_enable_topic}\n"
            f"  sem.markers:   {c.semantic_map_markers_topic}\n"
            f"  explore.enable:{c.exploration_enable_topic}\n"
            f"  approach.goal: {c.approach_goal_topic}\n"
            f"  approach.done: {c.approach_done_topic}\n"
            f"  vla.prompt:    {c.vla_prompt_topic}\n"
            f"  vla.done:      {c.vla_done_topic}\n"
            f"  nav.goal:      {c.nav_goal_topic}\n"
            f"[CFG] Frames:\n"
            f"  world:         {c.world_frame_id}\n"
            f"[CFG] Timing:\n"
            f"  search_timeout={c.search_timeout_sec}s, "
            f"explore_interval={c.exploration_check_interval_sec}s, "
            f"approach_timeout={c.approach_timeout_sec}s, "
            f"vla_timeout={c.vla_timeout_sec}s\n"
            f"[CFG] Final NAV:\n"
            f"  publish={c.publish_final_nav_goal}, pos={c.final_goal_xyz}, quat={c.final_goal_xyzw}"
        )


# --------------------------------- Main --------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Standard ROS 2 Python entrypoint."""
    rclpy.init(args=argv)
    node = MainInterpreterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down (Ctrl-C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
