#!/usr/bin/env python3
import os, argparse, time, yaml, torch, numpy as np
import cv2
import ros2_numpy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from smolVLA import VLA, VLAConfig


class VLA_ROS_ControlNode(Node):
    def __init__(self, yaml_cgf: dict):
        super().__init__("vla_ros_control")
        rw = yaml_cgf["ros_wrapper"]; vc = yaml_cgf["vla_class"]

        self.vla_keys = rw["vla_keys"]
        self.image_sizes = vc["image_sizes"]

        used_sizes = {k: self.image_sizes[k] for k in self.vla_keys.values() if k in self.image_sizes}

        vla_cfg = VLAConfig(
            ckpt=vc["ckpt"], one_step=bool(vc["one_step"]),
            fp16=bool(vc["fp16"]), device=vc["device"],
            image_sizes=used_sizes,
        )
        self.vla = VLA(vla_cfg)

        # State
        self.stateDIM = self.vla.state_dim
        self.last = {img_topic: None for img_topic in self.vla_keys.keys()}
        self.command_text = ""
        self.active = False
        self.current_manipulator_state = [0.0] * self.stateDIM

        # Subscribers
        for img_topic in self.vla_keys.keys():
            self.create_subscription(Image, img_topic, self._make_img_cb(img_topic), 10)
        self.create_subscription(String, rw["start_topic"], self._cmd_cb, 10)
        self.create_subscription(Float32MultiArray, rw["manipulator_state_topic"], self._manipulator_state_cb, 10)

        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, rw["manipulator_control_topic"], 10)

        # Inference timer
        self.create_timer(1.0/rw["command_HZ"], self._on_timer)

    def _make_img_cb(self, key):
        def cb(msg):
            self.get_logger().debug(f'Control {"started" if self.active else "stopped"}')
            arr = ros2_numpy.numpify(msg)
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if msg.encoding.upper() == "BGR8" else arr
            self.last[key] = rgb
        return cb

    def _cmd_cb(self, msg: String):
        self.command_text = msg.data
        self.active = not self.active
        self.get_logger().info(f'Control {"started" if self.active else "stopped"}: "{self.command_text}"')

    def _manipulator_state_cb(self, msg: Float32MultiArray):
        self.current_manipulator_state = list(msg.data)

    def _collect_imgs_for_vla(self):
        imgs = {}
        for img_topic, vla_key in self.vla_keys.items():
            img = self.last[img_topic]
            if img is None:
                return None
            imgs[vla_key] = img
        return imgs

    def _on_timer(self):
        if not self.active:
            return
        imgs = self._collect_imgs_for_vla()
        if imgs is None:
            return
        try:
            t1 = time.perf_counter()
            with torch.inference_mode():
                out = self.vla.infer(self.command_text, imgs, self.current_manipulator_state)
            t2 = time.perf_counter()
            arr = np.array(out, dtype=np.float32).ravel()
            self.get_logger().info(f'[{1000*(t2 - t1):.1f} ms] Next state: {arr.tolist()}')
            self.pub.publish(Float32MultiArray(data=arr.tolist()))
        except Exception as e:
            self.get_logger().warning(f"VLA inference failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("VLA_ROS_CONFIG", "vla_ros.yaml"))
    args = parser.parse_args()

    yaml_cfg = yaml.safe_load(open(args.config, "r"))
    rclpy.init()
    node = VLA_ROS_ControlNode(yaml_cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
