# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modifications (C) 2025 Artem Voronov
# GPL-3.0-or-later

import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray
from yolo_ros.yolo_node import YoloNode


class DebugNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # params
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("flip_method", -2)
        self.declare_parameter("input_image_topic", "/image_raw")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("detections_topic", "detections")  # "detections" or "tracking"

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.flip_method = self.get_parameter("flip_method").get_parameter_value().integer_value
        self.yolo_encoding = self.get_parameter("yolo_encoding").get_parameter_value().string_value
        self.input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        self.image_sub = message_filters.Subscriber(
            self, Image, self.input_image_topic, qos_profile=self.image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, Detection2DArray, self.detections_topic, qos_profile=10
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

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_publisher(self._dbg_pub)
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    # --- drawing helpers
    def draw_box(self, cv_image, det: Detection2D, color: Tuple[int, int, int]):
        bb = det.bbox
        min_pt = (round(bb.center.position.x - bb.size_x / 2.0), round(bb.center.position.y - bb.size_y / 2.0))
        max_pt = (round(bb.center.position.x + bb.size_x / 2.0), round(bb.center.position.y + bb.size_y / 2.0))

        rect_pts = np.array(
            [
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]],
            ]
        )
        M = cv2.getRotationMatrix2D((bb.center.position.x, bb.center.position.y), -np.rad2deg(bb.center.theta), 1.0)
        rect_pts = np.int0(cv2.transform(np.array([rect_pts]), M)[0])

        for i in range(4):
            pt1 = tuple(rect_pts[i])
            pt2 = tuple(rect_pts[(i + 1) % 4])
            cv2.line(cv_image, pt1, pt2, color, 2)

        if det.results:
            r0 = det.results[0]
            clsid = r0.hypothesis.class_id
            score = r0.hypothesis.score
            label = f"id={clsid} ({score:.3f})"
        else:
            label = "id=? (n/a)"

        pos = (min_pt[0] + 5, min_pt[1] + 25)
        cv2.putText(cv_image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        return cv_image

    def detections_cb(self, img_msg: Image, detection_msg: Detection2DArray) -> None:
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        if self.flip_method in [0, 1, -1]:
            cv_image = cv2.flip(cv_image, self.flip_method)

        img_h, img_w = cv_image.shape[:2]

        for det in detection_msg.detections:
            if det.results:
                r0 = det.results[0]
                clsid = r0.hypothesis.class_id
            else:
                clsid = -1

            if clsid not in self._class_to_color:
                self._class_to_color[clsid] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            color = self._class_to_color[clsid]

            # --- unflip the bbox center before drawing ---
            if self.flip_method in (0, 1, -1):
                cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
                bw, bh = det.bbox.size_x, det.bbox.size_y
                th = det.bbox.center.theta
                cx, cy, bw, bh, th = YoloNode._unflip_center_and_theta(
                    cx, cy, bw, bh, th, img_w, img_h, self.flip_method
                )
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.center.theta = th
                det.bbox.size_x = bw
                det.bbox.size_y = bh

            cv_image = self.draw_box(cv_image, det, color)

        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8", header=img_msg.header))


def main():
    rclpy.init()
    node = DebugNode()
    node.trigger_configure()
    node.trigger_activate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
