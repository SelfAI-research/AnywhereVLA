import rclpy
from pathlib import Path
from sensor_msgs.msg import Image
from yolo_node import YoloNode

rclpy.init()

params = {
    # core model
    "model_type": "YOLO",
    "model": "yolo12x.pt",
    "weights_dir": "/abs/path/to/object_detection/yolo_weights",  # <- put your real path
    "device": "cuda:0",

    # inference
    "enable": True,
    "flip_method": -1,
    "threshold": 0.45,
    "iou": 0.5,
    "imgsz_height": 480,
    "imgsz_width": 640,
    "half": True,
    "max_det": 200,
    "augment": False,
    "agnostic_nms": False,
    "retina_masks": False,
    "min_box_area": 16.0,

    "allow_classes": "",
    "block_classes": "person, refrigerator",

    # image topic / qos (unused when calling inference() directly)
    "input_image_topic": "/camera/camera/color/image_raw",
    "yolo_encoding": "bgr8",
    "image_reliability": 1,

    # profiling
    "profile_log_every": 50,
    "profile_window_sec": 5.0,

    # important: disable internal subscriber/worker
    "use_subscription": False,
}

node = YoloNode(initial_params=params)
node.trigger_configure()
node.trigger_activate()

# 1
import cv2
frame = cv2.imread("/data/yolo_img.png")

detections = node.inference(frame, publish=True)

print(f"1) got {len(detections.detections)} detections")
if len(detections.detections) > 0:
    print(f"{detections=}")
print()
print()

# 2
from cv_bridge import CvBridge
from std_msgs.msg import Header

bridge = CvBridge()
frame = cv2.imread("/data/yolo_img.png")

img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
img_msg.header = Header()
img_msg.header.stamp = node.get_clock().now().to_msg()
img_msg.header.frame_id = "camera"

detections = node.inference(img_msg, publish=True)

print(f"2) got {len(detections.detections)} detections")
if len(detections.detections) > 0:
    print(f"{detections=}")

node.destroy_node()
rclpy.shutdown()
