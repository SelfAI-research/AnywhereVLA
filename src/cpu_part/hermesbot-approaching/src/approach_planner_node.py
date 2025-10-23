"""ROS 2 node: publishes a Nav2-validated goal pose normal to the table face nearest the object, with step-by-step logs."""
from typing import List, Optional, Tuple
import math, rclpy

# dual import style
try:
    from .types_ex import NoMapError, InvalidObjectPointError, NoFaceDetectedError
    from .costmap_utils import costmap_to_array, occupancy_to_array, build_free_mask
    from .coord_utils import world_to_map, map_to_world, pose_xyyaw
    from .region_utils import flood_fill_local, boundary_points_toward_free
    from .gap_tools import close_small_gaps
    from .adapters import (
        FrontierEstimator, DistanceGradEstimator, PcaBoundaryEstimator, RaycastEstimator,
        CompositeNormal, EdgeFinderAdapter, CandidateGeneratorAdapter, DTMClearance, PlannerAdapter
    )
    from .planner_client import PlannerChecker
    from .markers import MarkersPublisher
except ImportError:  # pragma: no cover
    from types_ex import NoMapError, InvalidObjectPointError, NoFaceDetectedError  # type: ignore
    from costmap_utils import costmap_to_array, occupancy_to_array, build_free_mask  # type: ignore
    from coord_utils import world_to_map, map_to_world, pose_xyyaw  # type: ignore
    from region_utils import flood_fill_local, boundary_points_toward_free  # type: ignore
    from gap_tools import close_small_gaps  # type: ignore
    from adapters import (  # type: ignore
        FrontierEstimator, DistanceGradEstimator, PcaBoundaryEstimator, RaycastEstimator,
        CompositeNormal, EdgeFinderAdapter, CandidateGeneratorAdapter, DTMClearance, PlannerAdapter
    )
    from planner_client import PlannerChecker  # type: ignore
    from markers import MarkersPublisher  # type: ignore

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
import tf_transformations as tft
from tf2_geometry_msgs import do_transform_point, do_transform_pose

class ApproachingPlannerNode(Node):
    """Orchestrates TF, costmap parsing, face normal estimation, candidate generation, planner validation, and goal publish."""

    def __init__(self) -> None:
        super().__init__("approaching_planner")
        self._load_params()

        # Set logger level
        level_name = self._get_or_declare("logging.level", "INFO", "str")
        level = getattr(rclpy.logging.LoggingSeverity, level_name)
        self.get_logger().set_level(level)

        # Reentrant group so action callbacks can execute while this callback waits
        self.cbgroup = ReentrantCallbackGroup()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # IO
        self._subs()
        self.goal_pub = self.create_publisher(PoseStamped, self.topic_goal, 10)

        # Components
        self.planner = PlannerAdapter(
            PlannerChecker(
                self, self.nav2_action, self.nav2_timeout, self.allow_partial,
                callback_group=self.cbgroup
            )
        )
        self.clearing = DTMClearance()
        self.edge_finder = EdgeFinderAdapter()
        self.cand_gen = CandidateGeneratorAdapter(
            self.max_search, self.sample_step, self.num_angles, self.num_offsets,
            self.offset_step, self.max_slide, self.slide_step
        )
        self.markers = MarkersPublisher(self, self.topic_marker_edge, self.topic_marker_cands, self.topic_marker_goal)

        # State
        self.costmap_msg: Optional[OccupancyGrid] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.get_logger().info("ApproachingPlannerNode ready.")

    # -------- parameters & wiring --------
    def _get_or_declare(self, name: str, default, t: str = "float"):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        pv = self.get_parameter(name).get_parameter_value()
        return {"float": pv.double_value, "int": pv.integer_value, "bool": pv.bool_value, "str": pv.string_value}[t]

    def _load_params(self) -> None:
        gp = self._get_or_declare
        self.standoff = gp("stop.standoff_m", 0.45); self.safety = gp("stop.safety_margin_m", 0.05)
        self.use_nav2_global = gp("costmap.use_nav2_global", True, t="bool")
        self.costmap_topic = gp("costmap.topic", "/global_costmap/costmap", t="str")
        self.fallback_topic = gp("costmap.fallback_map_topic", "/map", t="str")
        self.lethal = gp("costmap.lethal_threshold", 254, t="int"); self.inscribed = gp("costmap.inscribed_threshold", 253, t="int")
        self.free_cost = gp("costmap.free_cost", 0, t="int")
        self.unknown_cost = gp("costmap.unknown_cost", 255, t="int"); self.unknown_as_obs = gp("costmap.unknown_as_obstacle", True, t="bool")
        self.normal_method = gp("approach.normal_method", "distance_gradient", t="str")
        self.max_search = gp("approach.max_search_radius_m", 2.0); self.sample_step = gp("approach.sample_step_m", 0.02)
        self.min_face_span = gp("approach.min_face_span_m", 0.25); self.max_slide = gp("approach.max_edge_slide_m", 0.40)
        self.slide_step = gp("approach.edge_slide_step_m", 0.05); self.gap_close_m = gp("approach.gap_close_m", 0.08)
        self.min_free_ring = gp("validation.min_free_ring_m", 0.15); self.disallow_inscribed = gp("validation.disallow_inscribed", True, t="bool")
        self.num_angles = gp("retry.num_angles", 5, t="int"); self.num_offsets = gp("retry.num_offsets", 3, t="int")
        self.offset_step = gp("retry.offset_step_m", 0.05); self.max_candidates = gp("retry.max_candidates", 30, t="int")
        self.nav2_enabled = gp("nav2_check.enable", True, t="bool"); self.nav2_mode = gp("nav2_check.mode", "action", t="str")
        self.nav2_action = gp("nav2_check.action_name", "/compute_path_to_pose", t="str")
        self.nav2_service = gp("nav2_check.service_name", "/planner_server/compute_path_to_pose", t="str")
        self.nav2_timeout = gp("nav2_check.timeout_sec", 1.5); self.allow_partial = gp("nav2_check.allow_partial_path", False, t="bool")
        self.use_current_tf = gp("nav2_check.use_current_pose_from_tf", True, t="bool")
        self.start_pose_frame = gp("nav2_check.start_pose_frame", "", t="str")
        self.topic_obj = gp("topics.object_point", "/approaching_object_point", t="str")
        self.topic_goal = gp("topics.goal_pose", "/goal_pose", t="str")
        self.topic_marker_edge = gp("topics.marker_edge", "/dock_pose_edge_marker", t="str")
        self.topic_marker_cands = gp("topics.marker_candidates", "/dock_pose_candidates", t="str")
        self.topic_marker_goal = gp("topics.marker_goal", "/dock_pose_goal", t="str")
        self.frame_compute = gp("frames.compute_in", "camera_init", t="str")
        self.frame_publish = gp("frames.publish_goal_in", "camera_init", t="str")
        self.frame_base = gp("frames.base_link", "base_link", t="str"); self.tf_timeout = gp("frames.tf_timeout_sec", 0.2)

    def _subs(self) -> None:
        # Subscribe with the reentrant group
        self.create_subscription(OccupancyGrid, self.costmap_topic, self._costmap_cb, 10, callback_group=self.cbgroup)
        self.create_subscription(OccupancyGrid, self.fallback_topic, self._map_cb, 10, callback_group=self.cbgroup)
        self.create_subscription(PointStamped, self.topic_obj, self._obj_cb, 10, callback_group=self.cbgroup)

    # -------- callbacks --------
    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        try:
            self.costmap_msg = msg
            self.get_logger().debug(f"Received costmap: {msg.info.width}x{msg.info.height} @ {msg.info.resolution:.3f} m")
        except Exception as e:
            self.get_logger().error(f"Costmap callback error: {e}")

    def _map_cb(self, msg: OccupancyGrid) -> None:
        try:
            self.map_msg = msg
            self.get_logger().debug(f"Received occupancy grid: {msg.info.width}x{msg.info.height} @ {msg.info.resolution:.3f} m")
        except Exception as e:
            self.get_logger().error(f"Map callback error: {e}")

    def _log_object_cell(self, lg, cf: str, mf: str, ox: float, oy: float,
                        res: float, cost, mx: float, my: float, myaw: float) -> None:
        """Log world→map mapping and raw cost/class at the object position."""
        ci, cj, cval, ccls = self._cost_at_world(cost, ox, oy, mx, my, myaw, res)
        val = "NA" if cval is None else str(cval)
        lg.info(
            f"Object @ {cf}->{mf}: ({ox:.3f},{oy:.3f}) -> cell({ci},{cj}) "
            f"cost={val} [{ccls}] (res={res:.3f} m)"
        )

    def _publish_edge_markers(self, frame: str, boundary, mx: float, my: float, myaw: float, res: float) -> None:
        """Publish sparsified boundary points for RViz."""
        if not boundary:
            return
        step = max(1, len(boundary) // 300)
        pts = [map_to_world(i, j, mx, my, myaw, res) for (i, j) in boundary[::step]]
        self.markers.publish_edge(frame, pts)

    def _obj_cb(self, msg: PointStamped) -> None:
        """Main object callback: compute and publish a feasible, normal-aligned approach goal."""
        lg = self.get_logger()
        try:
            # 1) Object in, to compute frame
            lg.info(f"Object received in {msg.header.frame_id}: ({msg.point.x:.3f},{msg.point.y:.3f})")
            cf, ox, oy = self._transform_object(msg)
            lg.debug(f"Transformed to compute frame {cf}: ({ox:.3f},{oy:.3f})")

            # 2) Load costmap arrays; align frames
            cost, res, mx, my, myaw, mf = self._load_map_arrays()
            if cf != mf:
                lg.debug(f"Align frames: compute_in={cf} -> map={mf}")
            cf, ox, oy = self._align_frames(cf, mf, ox, oy)
            self._log_object_cell(lg, cf, mf, ox, oy, res, cost, mx, my, myaw)

            # 3) Masks, object index, region + boundary
            free = self._build_free(cost)
            oi, oj, w, h = self._obj_index(ox, oy, mx, my, myaw, res, free)
            lg.info(f"Object idx: ({oi},{oj}) in map {w}x{h}, res={res:.3f}")
            region, boundary = self._extract_region(free, (oi, oj), res)
            lg.info(f"Region cells: {int(region.sum())}; boundary: {len(boundary)}")

            # 4) Normal, edge point, standoff
            normal = self._normal(region, free, boundary, (oi, oj), mx, my, myaw, res)
            lg.info(f"Normal (inward): ({normal[0]:.3f},{normal[1]:.3f})")
            ex, ey = self._edge(ox, oy, normal, free, mx, my, myaw, res)
            lg.info(f"Edge point: ({ex:.3f},{ey:.3f})")
            base_standoff = self._standoff()
            lg.info(f"Base standoff: {base_standoff:.3f} m")

            # 5) Clearance config + candidate search with planner validation
            self._config_clearance(free, cost, res, mx, my, myaw)
            chosen = self._choose(ex, ey, normal, base_standoff, res, mx, my, myaw, w, h, cost, cf)
            if not chosen:
                lg.error("No feasible goal found")
                return

            # 6) Publish: transform goal, markers, and pose
            gx, gy, gyaw = self._to_publish_frame(cf, chosen)
            pub_frame = self.frame_publish or cf
            lg.info(f"Chosen goal in publish frame {pub_frame}: ({gx:.3f},{gy:.3f}), yaw={math.degrees(gyaw):.1f}°")
            self._publish_edge_markers(pub_frame, boundary, mx, my, myaw, res)
            self._publish_goal(gx, gy, gyaw)
            self.markers.publish_goal(pub_frame, gx, gy, gyaw)

        except Exception as e:
            lg.error(f"Object callback error: {e}")

    # -------- pipeline steps --------
    def _transform_object(self, msg: PointStamped) -> Tuple[str, float, float]:
        compute_frame = self.frame_compute or (self.costmap_msg.header.frame_id if self.costmap_msg else "")
        if not compute_frame:
            raise NoMapError("No compute frame")
        if msg.header.frame_id == compute_frame:
            return compute_frame, msg.point.x, msg.point.y
        tf = self.tf_buffer.lookup_transform(
            compute_frame, msg.header.frame_id, rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_timeout)
        )
        obj = do_transform_point(msg, tf)
        return compute_frame, obj.point.x, obj.point.y

    def _load_map_arrays(self):
        if self.use_nav2_global and self.costmap_msg is not None:
            c, r, mx, my, myaw = costmap_to_array(self.costmap_msg); self._res_cached = r
            return c, r, mx, my, myaw, self.costmap_msg.header.frame_id
        if self.map_msg is not None:
            c, r, mx, my, myaw = occupancy_to_array(self.map_msg); self._res_cached = r
            return c, r, mx, my, myaw, self.map_msg.header.frame_id
        raise NoMapError("No map/costmap")

    def _align_frames(self, compute_frame: str, map_frame: str, ox: float, oy: float):
        if compute_frame == map_frame:
            return compute_frame, ox, oy
        ps = PointStamped()
        ps.header.frame_id = compute_frame
        ps.point.x, ps.point.y = ox, oy
        tf = self.tf_buffer.lookup_transform(
            map_frame, compute_frame, rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_timeout)
        )
        pt = do_transform_point(ps, tf)
        return map_frame, pt.point.x, pt.point.y

    def _build_free(self, cost):
        return build_free_mask(cost, self.lethal, self.inscribed, self.unknown_cost, self.unknown_as_obs, self.disallow_inscribed)

    def _obj_index(self, ox, oy, mx, my, myaw, res, free):
        oi, oj = world_to_map(ox, oy, mx, my, myaw, res)
        h, w = free.shape
        if oi < 0 or oi >= w or oj < 0 or oj >= h: raise InvalidObjectPointError("Object OOB")
        if free[oj, oi]: raise InvalidObjectPointError("Object not in obstacle")
        return oi, oj, w, h

    def _extract_region(self, free, obj_idx, res):
        occ = ~free
        # occ = close_small_gaps(occ, res, self.gap_close_m)
        r_cells = max(1, int(self.max_search / res))
        region = flood_fill_local(occ, obj_idx, r_cells)
        boundary = boundary_points_toward_free(region, free)
        if not boundary: raise NoFaceDetectedError("No free-facing boundary")
        return region, boundary

    def _classify_cost(self, v: int) -> str:
        if v == self.unknown_cost:
            return "UNKNOWN"
        if v >= self.lethal:
            return "LETHAL"
        if v >= self.inscribed:
            return "INSCRIBED"
        if v == self.free_cost:
            return "FREE"
        return f"COST={v}"

    def _cost_at_world(self, cost, x: float, y: float, mx: float, my: float, myaw: float, res: float):
        i, j = world_to_map(x, y, mx, my, myaw, res)
        h, w = cost.shape
        if 0 <= i < w and 0 <= j < h:
            v = int(cost[j, i])
            return i, j, v, self._classify_cost(v)
        return i, j, None, "OOB"

    def _normal(self, region, free, boundary, obj_idx, mx, my, myaw, res) -> Tuple[float, float]:
        ests = [DistanceGradEstimator(), FrontierEstimator(boundary), PcaBoundaryEstimator(boundary), RaycastEstimator(self.max_search, self.sample_step) ]

        n = CompositeNormal(ests).estimate(region, free, obj_idx, mx, my, myaw, res)
        if n is None:
            raise NoFaceDetectedError("Normal failed")
        nx, ny = n; m = math.hypot(nx, ny)
        if m == 0:
            raise NoFaceDetectedError("Degenerate normal")
        return nx/m, ny/m

    def _edge(self, ox, oy, normal, free, mx, my, myaw, res) -> Tuple[float, float]:
        d = self.edge_finder.distance_to_edge(ox, oy, normal[0], normal[1], free, mx, my, myaw, res, self.max_search)
        if d < 0: raise NoFaceDetectedError("Edge not found")
        return ox - normal[0]*d, oy - normal[1]*d

    def _standoff(self) -> float:
        return float(self.standoff + self.safety)

    def _config_clearance(self, free, cost, res, mx, my, myaw):
        self.clearing.configure_grid(free, cost, res, mx, my, myaw, self.min_free_ring, self.disallow_inscribed, self.inscribed)

    def _choose(self, ex, ey, normal, base_standoff, res, mx, my, myaw, w, h, cost, compute_frame):
        lg = self.get_logger()
        try:
            preview = []; tried = 0
            for (cx, cy, yaw) in self.cand_gen.generate(ex, ey, normal[0], normal[1], base_standoff):
                if tried < 80: preview.append((cx, cy, yaw))
                tried += 1
                if tried > self.max_candidates: break
                if not self.clearing.ok(cx, cy):
                    if tried % 10 == 0: lg.debug(f"Candidate {tried}: failed local clearance")
                    continue

                goal = pose_xyyaw(cx, cy, yaw, compute_frame)
                start = self._start_pose(compute_frame) if (self.nav2_enabled and self.use_current_tf) else None

                ok = True if not self.nav2_enabled else self.planner.feasible(goal, start, self.nav2_mode)
                lg.debug(f"Candidate {tried}: planner_ok={ok}")
                if ok:
                    self.markers.publish_candidates(compute_frame, preview, limit=80)
                    return (cx, cy, yaw)

            self.markers.publish_candidates(compute_frame, preview, limit=80)
            return
        except Exception as e:
            lg.error(f"_choose error: {e}")

    def _to_publish_frame(self, compute_frame: str, chosen: Tuple[float, float, float]):
        gx, gy, gyaw = chosen
        pub = self.frame_publish or compute_frame
        if pub == compute_frame:
            return gx, gy, gyaw
        tmp = pose_xyyaw(gx, gy, gyaw, compute_frame)
        try:
            tf = self.tf_buffer.lookup_transform(
                pub, compute_frame, rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout)
            )
            out = do_transform_pose(tmp, tf)
            gx, gy = out.pose.position.x, out.pose.position.y
            _, _, gyaw = tft.euler_from_quaternion([
                out.pose.orientation.x, out.pose.orientation.y,
                out.pose.orientation.z, out.pose.orientation.w
            ])
        except Exception:
            pass
        return gx, gy, gyaw

    def _publish_goal(self, gx: float, gy: float, gyaw: float) -> None:
        out = pose_xyyaw(gx, gy, gyaw, self.frame_publish or self.frame_compute)
        out.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(out)

    def _start_pose(self, compute_frame: str) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(compute_frame, self.frame_base, rclpy.time.Time())
            _, _, yaw = tft.euler_from_quaternion([
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w
            ])
            return pose_xyyaw(tf.transform.translation.x, tf.transform.translation.y, yaw, compute_frame)
        except Exception:
            return None

def main() -> None:
    rclpy.init()
    node = ApproachingPlannerNode()
    ex = MultiThreadedExecutor(num_threads=2)  # >=2 so action callbacks run while _obj_cb waits
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
