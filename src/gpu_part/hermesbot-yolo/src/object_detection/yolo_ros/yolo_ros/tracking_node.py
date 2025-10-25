# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modifications (C) 2025 Artem Voronov
# GPL-3.0-or-later

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge

from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D


class TrackingNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("tracking_node")

        # params
        self.declare_parameter("input_image_topic", "/image_raw")
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        tracker_name = self.get_parameter("tracker").get_parameter_value().string_value

        self.input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
        self.image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

        self.tracker = self.create_tracker(tracker_name)
        # publish tracked detections as Detection2DArray (minimal change; IDs omitted)
        self._pub = self.create_publisher(Detection2DArray, "tracking", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # subs
        self.image_sub = message_filters.Subscriber(
            self, Image, self.input_image_topic, qos_profile=image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, Detection2DArray, "detections", qos_profile=10
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer
        self._synchronizer = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tracker

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap") # for linear_assignment
        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**YAML.load(tracker))
        assert cfg.tracker_type in [
            "bytetrack", "botsort"
        ], f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def detections_cb(self, img_msg: Image, detections_msg: Detection2DArray) -> None:
        tracked_detections_msg = Detection2DArray()
        tracked_detections_msg.header = img_msg.header

        # image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # build detector input: [x1,y1,x2,y2,score,cls] as float32
        det_list = []
        for d in detections_msg.detections:
            if not d.results:
                continue
            r0 = d.results[0]
            clsid = float(int(r0.hypothesis.class_id))  # Humble: class_id is string → cast to int → float
            score = float(r0.hypothesis.score)

            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
            det_list.append([x1, y1, x2, y2, score, clsid])

        if det_list:
            arr = np.asarray(det_list, dtype=np.float32)
            det = Boxes(arr, (img_msg.height, img_msg.width))
            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:
                for t in tracks:
                    # t may contain an index to source detection as last element in this setup
                    det_idx = int(t[-1]) if isinstance(t[-1], (int, np.integer, float, np.floating)) else 0
                    det_idx = max(0, min(det_idx, len(detections_msg.detections) - 1))

                    tracked_box = Boxes(t[:-1], (img_msg.height, img_msg.width))
                    src_det: Detection2D = detections_msg.detections[det_idx]

                    # update bbox with tracked xywh
                    xywh = tracked_box.xywh[0]
                    out_det = Detection2D()
                    out_det.header = img_msg.header
                    out_det.bbox = BoundingBox2D()
                    out_det.bbox.center.position.x = float(xywh[0])
                    out_det.bbox.center.position.y = float(xywh[1])
                    out_det.bbox.center.theta = src_det.bbox.center.theta
                    out_det.bbox.size_x = float(xywh[2])
                    out_det.bbox.size_y = float(xywh[3])

                    # keep classification hypothesis
                    out_det.results = src_det.results
                    tracked_detections_msg.detections.append(out_det)

        # publish detections
        self._pub.publish(tracked_detections_msg)


def main():
    rclpy.init()
    node = TrackingNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
