import time
from yacs.config import CfgNode

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import tf2_ros

from utils import Transform, load_cfg, _reliability, _durability
from octomap2d import Octomap2D, Octo2DConfig
from preprocessor import Preprocessor, PreprocConfig
from msg_converter import ros_pointcloud_to_numpy_points


class OctomapServerPy(Node):
    """
    Subscribes to a PointCloud2 in map frame, looks up sensor origin via TF,
    updates a 2D OctoMap with raycasting, and publishes OccupancyGrid.
    """
    def __init__(self, cfg: CfgNode):
        super().__init__("octomap_server_py")
        self.cfg = cfg

        # Modules
        self.map = Octomap2D(Octo2DConfig(
            resolution=cfg.map.resolution,
            prob_hit=cfg.map.prob_hit,
            prob_miss=cfg.map.prob_miss,
            clamp_min=cfg.map.clamp_min,
            clamp_max=cfg.map.clamp_max,
            occ_threshold=cfg.map.occ_threshold,
            init_x_size=cfg.map.initial_size[0],
            init_y_size=cfg.map.initial_size[1],
        ))
        self.pre = Preprocessor(PreprocConfig(
            z_min=cfg.preproc.z_range[0],
            z_max=cfg.preproc.z_range[1],
            range_filter=cfg.preproc.range_filter,
            fov_filter=cfg.preproc.fov_filter,
            outlier_mode=cfg.preproc.outlier_mode,
            grid_r=cfg.preproc.outlier_grid_r,
            min_pts_per_cell=int(cfg.preproc.outlier_min_pts_per_cell),
        ))

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=float(cfg.io.tf_cache_sec)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriber
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(cfg.io.input_qos.depth),
            reliability=_reliability(cfg.io.input_qos.reliability),
        )
        self.sub = self.create_subscription(PointCloud2, cfg.io.input_topic, self._on_cloud, sensor_qos)

        # Publisher
        pub_qos = QoSProfile(depth=int(cfg.io.output_qos.depth))
        pub_qos.durability = _durability(cfg.io.output_qos.durability)
        self.pub_map = self.create_publisher(OccupancyGrid, cfg.io.output_topic, pub_qos)
        self.pub_map_prob = self.create_publisher(OccupancyGrid, cfg.io.output_prob_topic, pub_qos)

        if cfg.logging.debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.get_logger().info("octomap_server_py started.")

        # Last origin
        self._last_origin: Transform = None
        self._last_origin_time = Time(seconds=0)

    def _lookup_sensor_origin(self, cloud_time: Time) -> Transform | None:
        target = self.cfg.io.frame_id
        source = self.cfg.io.sensor_frame_id

        wait_dur = Duration(seconds=float(self.cfg.io.tf_wait_sec))
        max_stale = Duration(seconds=float(self.cfg.io.tf_max_staleness_sec))
        reuse_for = Duration(seconds=float(self.cfg.io.tf_reuse_last_sec))

        # 1) Exact at cloud stamp (short wait)
        try:
            t = self.tf_buffer.lookup_transform(target, source, cloud_time, timeout=wait_dur)
            tf_time = Time.from_msg(t.header.stamp)
            self._last_origin = Transform.from_msg(t)
            self._last_origin_time = cloud_time
            self.get_logger().debug(f"[Latest lookup] success: diff={(cloud_time - tf_time).nanoseconds * 1e-9:.3f} s")
            return self._last_origin
        except Exception as e:
            self.get_logger().debug(f"[Short lookup] error: {e}")

        # 2) Latest TF, accept if not too stale
        try:
            t = self.tf_buffer.lookup_transform(target, source, Time())  # latest
            tf_time = Time.from_msg(t.header.stamp)
            time_diff = cloud_time - tf_time
            if time_diff <= max_stale:
                self._last_origin = Transform.from_msg(t)
                self._last_origin_time = cloud_time
                self.get_logger().debug(f"[Latest lookup] success: diff={time_diff.nanoseconds * 1e-9:.3f} s")
                return self._last_origin
        except Exception as e:
            self.get_logger().debug(f"[Latest lookup] error: {e}")

        # 3) Reuse last good briefly
        if not self._last_origin:
            self.get_logger().warn(f"TF missing; no last origin available")
            return None

        time_diff = cloud_time - self._last_origin_time
        if time_diff <= reuse_for:
            self.get_logger().warn(f"TF missing; reusing last origin, time_diff: {time_diff.nanoseconds * 1e-9:.3f} s")
            return self._last_origin
        self.get_logger().warn(f"TF {target}<-{source} @ {time_diff.nanoseconds * 1e-9:.3f} s unavailable; dropping scan.")
        return None


    def _on_cloud(self, msg: PointCloud2) -> None:
        t0 = time.perf_counter()
        pts3d = ros_pointcloud_to_numpy_points(msg)
        cloud_time = Time().from_msg(msg.header.stamp)
        self.get_logger().debug(f"receive point cloud {cloud_time.nanoseconds*1e-9:.3f}: {pts3d.shape[0]} points")
        t1 = time.perf_counter()

        origin = self._lookup_sensor_origin(cloud_time)

        t2 = time.perf_counter()
        if origin is None:
            return

        xy = self.pre.process(pts3d, origin)
        t3 = time.perf_counter()

        self.map.integrate_scan(
            xy,
            origin_xy=origin.translation[:2],
            raycast_range=self.cfg.map.raycast_range,
            do_raycast=bool(self.cfg.map.raycast_free_space),
            angle_bin_deg=bool(self.cfg.map.angle_bin_deg),
            use_map_occlusion=bool(self.cfg.map.use_map_occlusion),
            clear_occluder=bool(self.cfg.map.clear_occluder),
        )
        t4 = time.perf_counter()

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.cfg.io.frame_id
        grid, grid_prob = self.map.build_grids(header, z_level = self.cfg.map.output_level)
        t5 = time.perf_counter()

        self.pub_map.publish(grid)
        self.pub_map_prob.publish(grid_prob)

        self.get_logger().info(
            f"cloud->np {1000*(t1-t0):.1f} ms | lookup {1000*(t2-t1):.1f} ms | pre {1000*(t3-t2):.1f} ms | "
            f"update {1000*(t4-t3):.1f} ms | to_msg {1000*(t5-t4):.1f} ms | total {1000*(t5-t0):.1f} ms"
        )



def main():
    cfg = load_cfg("config/octomap.yaml")

    rclpy.init()
    node = OctomapServerPy(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
