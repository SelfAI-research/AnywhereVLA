#!/usr/bin/env python3
import os
import struct
from typing import List, Optional
import numpy as np
from collections import deque
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection2DArray, Detection3DArray
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, ConnectivityException

# Project modules
from semantic_map.common import common_utils
from semantic_map.common.voxelization import voxelize_numpy
from semantic_map.common.densifyer import LidarDensifier
from semantic_map.common.profiller import Profiler
from semantic_map.common.projection import ObjectProjection
from semantic_map.common.semantic import SemanticPerception

# ====================== Node ======================
class SemanticMap3D(Node):
    def __init__(self):
        super().__init__("semantic_map_3d")
        # --- Load configuration files
        self.declare_parameter("config_folder", "config")
        cfg_folder = self.get_parameter("config_folder").get_parameter_value().string_value
        self.get_logger().info(f"Loaded config from: {cfg_folder=}")

        cfg = common_utils.extract_configuration(cfg_folder, cfg_file='general_configuration.yaml')
        extrinsic_yaml = os.path.join(cfg_folder, cfg['general']['camera_extrinsic_calibration'])
        intr_yaml      = os.path.join(cfg_folder, cfg['general']['camera_intrinsic_calibration'])
        coco_yaml      = os.path.join(cfg_folder, cfg['general']['coco_names_path'])
        self.semantic_yaml  = os.path.join(cfg_folder, cfg['general']['semantic_perception_file'])
        common_utils.load_labels_from_yaml(coco_yaml)

        self.lidar_frame  = cfg['general']['lidar_frame']    # e.g. "map" or "odom"
        self.world_frame  = cfg['general']['world_frame']    # e.g. "map" or "odom"
        self.image_topic_sub = cfg['general']['image_topic']   # e.g. "/camera/image_raw"
        self.lidar_topic_sub = cfg['general']['lidar_topic']   # e.g. "/lidar/points"
        self.detect_topic_sub = cfg['general']['detection_topic']   # e.g. "/object_detection"

        self.max_time_diff = cfg['general']['max_time_diff']
        self.topic_queue   = cfg['general']['topic_queue']

        self.T_lidar_to_cam = common_utils.load_extrinsic(extrinsic_yaml)
        self.T_cam_to_lidar = common_utils.invert_h(self.T_lidar_to_cam)
        self.camera_matrix, self.dist_coeffs, _ = common_utils.load_intrinsics(intr_yaml)

        # Load submodule-specific configs
        prof_cfg      = cfg['profiler']
        densify_cfg   = cfg['lidar_densifier']
        projection_cfg= cfg['object_projection']
        semantic_cfg  = cfg['semantic_perception']

        self.ampl_bb = (projection_cfg["increase_x_size"], projection_cfg["increase_y_size"])
        self.en_densify = densify_cfg['enable']
        self.voxel_size = densify_cfg['voxel_size_m']
        self.flip_image = -1

        # --- Initialize submodules
        self.prof = Profiler(logger=self.get_logger(), **prof_cfg)
        self.lidar_densifier = LidarDensifier(self.T_lidar_to_cam, self.camera_matrix, self.dist_coeffs, **densify_cfg)
        self.object_projection = ObjectProjection(**projection_cfg)
        self.semantic_perception = SemanticPerception(self.world_frame, **semantic_cfg)

        # Bridge for image conversion
        self.bridge = CvBridge()

        # TF buffer
        tf_buffer_dur = float(cfg['general']['tf_buffer_dur'])
        self.tf_buffer = Buffer(cache_time=Duration(seconds=tf_buffer_dur))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Synchronized subscriptions
        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.topic_queue,
        )
        qos_lid = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.topic_queue,
        )
        qos_det = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.topic_queue,
        )

        self.cb_sub   = MutuallyExclusiveCallbackGroup()
        self.sub_img  = Subscriber(self, Image, self.image_topic_sub, qos_profile=qos_img, callback_group=self.cb_sub)
        self.sub_lid  = Subscriber(self, PointCloud2, self.lidar_topic_sub, qos_profile=qos_lid, callback_group=self.cb_sub)
        self.sub_det  = Subscriber(self, Detection2DArray, self.detect_topic_sub, qos_profile=qos_det, callback_group=self.cb_sub)

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_lid, self.sub_det],
            queue_size=self.topic_queue, slop=self.max_time_diff,
            allow_headerless=False
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info(
            f"Initialized SemanticMap3D. Subscribed: img={self.image_topic_sub}, lidar={self.lidar_topic_sub}, det={self.detect_topic_sub}"
        )

        # --- Publishers
        self.en_image = cfg['debug']['en_image']
        self.en_colored = cfg['debug']['en_colored']
        self.en_objects = cfg['debug']['en_objects']
        if self.en_image:
            self.debug_img_pub      = self.create_publisher(Image,     cfg['debug']['debug_image_topic'],         1)
        if self.en_colored:
            self.colored_points_pub = self.create_publisher(PointCloud2,cfg['debug']['colored_pointcloud_topic'], 1)
        if self.en_objects:
            self.objects_points_pub = self.create_publisher(PointCloud2,cfg['debug']['objects_pointcloud_topic'], 1)
        self.markers_pub        = self.create_publisher(MarkerArray,cfg['debug']['markers_topic'],            1)

        # --- Thread for 3D detections
        detection_queue = cfg['general']['detection_queue']
        update_hz = cfg['general']['update_rate_hz']
        self.det3d_queue = deque(maxlen=detection_queue)
        self.det_lock = threading.Lock()
        self.cb_timer = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(1.0/update_hz, self.process_detection3d, callback_group=self.cb_timer)

    def get_tf(self, target: str, source: str, stamp: Time) -> Optional[TransformStamped]:
        """Lookup TF at 'stamp'. Returns None if unavailable within timeout."""
        try:
            timeout = Duration(seconds=self.max_time_diff)
            stamp_time = stamp if isinstance(stamp, Time) else Time.from_msg(stamp)
            if self.tf_buffer.can_transform(target, source, stamp_time, timeout):
                return self.tf_buffer.lookup_transform(target, source, stamp_time)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning(f"TF error {source}->{target} @ {stamp}: {e}")
        return None

    def time_sync_check(self, t_img, t_lid, t_det) -> bool:
        t_img = t_img if isinstance(t_img, Time) else Time.from_msg(t_img)
        t_lid = t_lid if isinstance(t_lid, Time) else Time.from_msg(t_lid)
        t_det = t_det if isinstance(t_det, Time) else Time.from_msg(t_det)

        skew_img_det = abs((t_det - t_img).nanoseconds) * 1e-6  # ms
        skew_lid_det = abs((t_det - t_lid).nanoseconds) * 1e-6  # ms

        if (skew_img_det * 1e-3) > self.max_time_diff or (skew_lid_det * 1e-3) > self.max_time_diff:
            self.get_logger().warning(f"Large skew: img-det={skew_img_det:.1f} ms, lidar-det={skew_lid_det:.1f} ms")
            return False

        # 2) Ensure TF exists at the data time
        timeout = Duration(seconds=self.max_time_diff)
        if not self.tf_buffer.can_transform(self.world_frame, self.lidar_frame, t_det, timeout=timeout):
            if not self.tf_buffer.can_transform(self.world_frame, self.lidar_frame, t_lid, timeout=timeout):
                self.get_logger().warning("No TF at detection or lidar stamp; skipping frame")
                return False

        return True

    # --- Main synchronized callback ---
    def sync_callback(self, img_msg: Image, lidar_msg: PointCloud2, dets_msg: Detection2DArray):
        stamp_img = img_msg.header.stamp
        stamp_lid = lidar_msg.header.stamp
        stamp_det = dets_msg.header.stamp
        if not self.time_sync_check(stamp_img, stamp_lid, stamp_det):
            return

        t0_cb = t0 = self.prof.t()
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        xyz_world = common_utils.cloud_to_xyz(lidar_msg)
        self.prof.add("convert", self.prof.t() - t0)

        # Prefer detection time if available; else fall back to LiDAR time.
        t_det = Time.from_msg(stamp_det)
        timeout = Duration(seconds=self.max_time_diff)
        use_det_time = self.tf_buffer.can_transform(self.world_frame, self.lidar_frame, t_det, timeout=timeout)
        stamp_tf = stamp_det if use_det_time else stamp_lid
        T_lidar_to_world_tf = self.get_tf(self.world_frame, self.lidar_frame, stamp_tf)
        if not T_lidar_to_world_tf:
            return

        T_lidar_to_world = common_utils.tf_msg_to_matrix(T_lidar_to_world_tf)
        T_world_to_lidar = common_utils.invert_h(T_lidar_to_world)
        xyz_lidar = common_utils.transform_points_matrix(xyz_world, T_world_to_lidar)

        # Lidar Filtering
        t0 = self.prof.t()
        xyz_proj, uvz_proj = self.lidar_densifier.filter_points_on_image(xyz_lidar, cv_image)
        self.prof.add("filter_points_on_image", self.prof.t() - t0)

        # Lidar Densifier
        xyz_work = xyz_proj
        if self.en_densify:
            t0 = self.prof.t()
            xyz_work = self.lidar_densifier.densify_velodyne_rings(xyz_proj)
            self.prof.add("lidar_densifier", self.prof.t() - t0)

        # Lidar Voxelization
        t0 = self.prof.t()
        xyz_voxel = voxelize_numpy(xyz_work, self.voxel_size)
        self.prof.add("voxelize", self.prof.t() - t0)

        xyz_in_fov, uvz_in_fov = self.lidar_densifier.filter_points_on_image(xyz_voxel, cv_image)

        # Object Projection (2D -> 3D)
        t0 = self.prof.t()
        detection3d_array, point_clouds = self.object_projection(
            detections2d=dets_msg,
            xyz_in_fov=xyz_in_fov,
            uvz_in_fov=uvz_in_fov,
            frame_id=lidar_msg.header.frame_id,
            stamp=stamp_lid,
        )
        self.prof.add("object_projection", self.prof.t() - t0)

        with self.det_lock:
            self.det3d_queue.append((detection3d_array, point_clouds, T_lidar_to_world))

        # Publish
        t0 = self.prof.t()
        if self.en_image:
            t1 = self.prof.t()
            header_cam = Header(stamp=stamp_img, frame_id=img_msg.header.frame_id)
            debug_img = common_utils.draw_on_image(cv_image, uvz_proj, dets_msg, self.ampl_bb)
            common_utils.publish_image(self.debug_img_pub, debug_img, header_cam, flip_image=self.flip_image)
            self.prof.add("pub_image", self.prof.t() - t1)
        if self.en_colored:
            t1 = self.prof.t()
            header_lidar = Header(stamp=stamp_lid, frame_id=self.lidar_frame)
            common_utils.publish_colored_cloud(self.colored_points_pub, xyz_in_fov, uvz_in_fov, header_lidar, cv_image)
        if self.en_objects:
            header_lidar = Header(stamp=stamp_lid, frame_id=self.lidar_frame)
            self.publish_objects3d(self.objects_points_pub, detection3d_array, point_clouds, header_lidar)
        self.prof.add("pub_total", self.prof.t() - t0)

        self.prof.add("total", self.prof.t() - t0_cb)
        self.prof.maybe_log()

    @staticmethod
    def rgb_to_float(r, g, b):
        return struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]

    @staticmethod
    def publish_objects3d(pub, detection3d_array: Detection3DArray, point_clouds: List[np.ndarray], header: Header):
        all_points = []
        for det3d, pts in zip(detection3d_array.detections, point_clouds, strict=True):
            class_id = det3d.results[0].hypothesis.class_id if det3d.results else "unknown"
            rgb = np.array(common_utils.color_from_id(class_id), dtype=np.uint8)
            rgb_float = SemanticMap3D.rgb_to_float(*rgb)
            rgb_column = np.full((pts.shape[0], 1), rgb_float, dtype=np.float32)
            pts_rgb = np.hstack([pts, rgb_column])
            all_points.append(pts_rgb)
        if all_points:
            all_points = np.vstack(all_points)
            common_utils.publish_xyzr_cloud(pub, all_points, Header(stamp=header.stamp, frame_id=header.frame_id))

    def process_detection3d(self):
        """Periodic processing of queued 3D detections to update semantic map and publish markers."""
        batch = []
        with self.det_lock:
            while self.det3d_queue:
                batch.append(self.det3d_queue.popleft())

        if not batch:
            return

        t0 = self.prof.t()
        for detection3d_array, point_clouds, T_lidar_to_world in batch:
            self.semantic_perception.update(detection3d_array, point_clouds, T_lidar_to_world)
        self.prof.add("semantic_update", self.prof.t() - t0)

        self.semantic_perception.build_objects()
        marker_array = self.semantic_perception.to_marker_array()
        self.markers_pub.publish(marker_array)

        self.semantic_perception.save_to_file(self.semantic_yaml)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMap3D()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
