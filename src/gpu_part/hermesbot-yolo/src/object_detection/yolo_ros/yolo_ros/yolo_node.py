# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modifications (C) 2025 Artem Voronov
# GPL-3.0-or-later

import shutil
import cv2, threading
import numpy as np
from typing import Dict, Optional, Any
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from rclpy.parameter import Parameter

import torch
from pathlib import Path
from ultralytics import YOLO, YOLOWorld, YOLOE
from ultralytics.engine.results import Results

from std_msgs.msg import Header
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from vision_msgs.msg import Pose2D as VMPose2D
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose


class YoloNode(LifecycleNode):
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("yolo_node")

        # params
        self.declare_parameter("input_image_topic", "/image_raw")
        self.declare_parameter("detections_topic", "detections")

        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("weights_dir", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("use_subscription", True)  # new

        self.declare_parameter("flip_method", -2)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        self.declare_parameter("min_box_area", 16.0)

        self.declare_parameter("allow_classes", "")
        self.declare_parameter("block_classes", "")

        self.declare_parameter("profile_log_every", 0)
        self.declare_parameter("profile_window_sec", 5.0)

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}

        # state
        self._prof: Dict[str, list] = {}
        self._prof_count: int = 0
        self._last_cb_t = None

        # async worker state
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._frame_lock = threading.Lock()
        self._new_frame_evt = threading.Event()
        self._latest_img = None
        self._latest_header = None

        # ROS entities
        self._enable_srv = None
        self._sub = None

        # profiling timer
        self._prof_timer = None

        # programmatic overrides
        if initial_params:
            self.set_parameters([Parameter(name=k, value=v) for k, v in initial_params.items()])

    # ---------- profiling ----------
    def _t(self): return self.get_clock().now().nanoseconds
    def _prof_add(self, name: str, dt_sec: float): self._prof.setdefault(name, []).append(dt_sec)
    def _log_profile_medians(self):
        if not self._prof: return
        parts = []
        periods = self._prof.get("period", [])
        n = 0; sum_period = 0.0
        if periods:
            for v in reversed(periods):
                sum_period += float(v); n += 1
                if sum_period >= self.profile_window_sec: break
            if sum_period > 0.0: parts.append(f"rate={n/sum_period:.1f}Hz")
        for k, values in self._prof.items():
            if values:
                tail = values[-n:] if n>0 else values
                parts.append(f"{k}={(sum(tail)/len(tail))*1e3:.2f}ms")
        self.get_logger().info(f"Timing (last {self.profile_window_sec:.1f}s): " + ", ".join(parts))
    def _maybe_log_profile(self):
        # self._prof_count += 1
        # if self.profile_log_every > 0 and (self._prof_count % self.profile_log_every) == 0:
        self._log_profile_medians()

    # ---------- lifecycle ----------
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type   = self.get_parameter("model_type").get_parameter_value().string_value
        self.model        = self.get_parameter("model").get_parameter_value().string_value
        self.weights_dir  = self.get_parameter("weights_dir").get_parameter_value().string_value
        self.device       = self.get_parameter("device").get_parameter_value().string_value
        self.yolo_encoding= self.get_parameter("yolo_encoding").get_parameter_value().string_value

        # inference params
        self.flip_method  = self.get_parameter("flip_method").get_parameter_value().integer_value
        self.threshold    = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou          = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width  = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half         = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det      = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment      = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
        self.min_box_area = self.get_parameter("min_box_area").get_parameter_value().double_value

        # class filters
        allow_raw = self.get_parameter("allow_classes").get_parameter_value().string_value
        self.allow_classes = [s.strip().lower() for s in allow_raw.split(",") if s.strip()]
        block_raw = self.get_parameter("block_classes").get_parameter_value().string_value
        self.block_classes = [s.strip().lower() for s in block_raw.split(",") if s.strip()]

        # profiling params
        self.profile_log_every = self.get_parameter("profile_log_every").get_parameter_value().integer_value
        self.profile_window_sec = self.get_parameter("profile_window_sec").get_parameter_value().double_value

        # ros params
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.use_subscription = self.get_parameter("use_subscription").get_parameter_value().bool_value
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
    
        # qos param (int → enum)
        rel = QoSReliabilityPolicy.SYSTEM_DEFAULT
        if   self.reliability == 1: rel = QoSReliabilityPolicy.RELIABLE
        elif self.reliability == 2: rel = QoSReliabilityPolicy.BEST_EFFORT

        # pub
        self.qos = QoSProfile(
            reliability=rel,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self._pub = self.create_lifecycle_publisher(Detection2DArray, self.detections_topic, self.qos)
        self.cv_bridge = CvBridge()

        # backend speedups
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS

    def _resolve_model_spec(self) -> str:
        m = Path(self.model)
        if m.is_file():
            self.get_logger().info(f"Using model from explicit path: {m}")
            self._resolved_local = True
            return str(m)
        wd = Path(self.weights_dir) if self.weights_dir else None
        self.get_logger().info(f"Search model in the folder: {wd}")
        if wd and wd.is_dir():
            candidate = wd / self.model
            if candidate.is_file():
                self.get_logger().info(f"Using model from weights_dir: {candidate}")
                self._resolved_local = True
                return str(candidate)
            else:
                self.get_logger().info(f"Not found in weights_dir: {candidate} — fallback to auto-download if supported.")
        else:
            if self.weights_dir:
                self.get_logger().warning(f"weights_dir not a directory: '{self.weights_dir}' (ignoring)")
        self._resolved_local = False
        return self.model

    def _warmup(self):
        try:
            h, w = int(self.imgsz_height), int(self.imgsz_width)
            dummy = torch.zeros(1, 3, h, w, device=self.device)
            if self.half and "cuda" in self.device: dummy = dummy.half()
            with torch.inference_mode():
                for _ in range(3):
                    _ = self.yolo.predict(
                        source=dummy, imgsz=(h, w), conf=self.threshold, iou=self.iou,
                        half=self.half, max_det=self.max_det, augment=False,
                        agnostic_nms=self.agnostic_nms, retina_masks=self.retina_masks,
                        device=self.device, verbose=False, stream=False
                    )
        except Exception as e:
            self.get_logger().warn(f"Warmup skipped ({e})")

    def _persist_downloaded_weights(self):
        if not self.weights_dir: return
        try:
            wd = Path(self.weights_dir)
            wd.mkdir(parents=True, exist_ok=True)
            src = None
            for cand in [getattr(self.yolo, "ckpt_path", None), getattr(self.yolo, "weights", None)]:
                if isinstance(cand, (str, Path)):
                    p = Path(cand)
                    if p.is_file(): src = p; break
            if src is None:
                cache = Path.home() / ".cache" / "ultralytics"
                p = cache / self.model
                if p.is_file(): src = p
            if src is None:
                self.get_logger().warning("Could not locate downloaded weights to save.")
                return
            dst = wd / self.model
            try:
                if src.resolve() == dst.resolve():
                    self.get_logger().info("Weights already in target folder.")
                    return
            except Exception:
                pass
            if not dst.exists() or src.stat().st_size != dst.stat().st_size:
                shutil.copy2(src, dst)
                self.get_logger().info(f"Saved downloaded weights to: {dst}")
        except Exception as e:
            self.get_logger().warning(f"Failed to save downloaded weights: {e}")

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
        try:
            model_spec = self._resolve_model_spec()
            yolo_cls = self.type_to_model.get(self.model_type)
            if yolo_cls is None:
                self.get_logger().error(f"Unknown model_type='{self.model_type}'. Use one of: {list(self.type_to_model.keys())}")
                return TransitionCallbackReturn.ERROR

            self.get_logger().info(f"Loading {self.model_type} model: '{model_spec}'")
            self.yolo = yolo_cls(model_spec)

            if not getattr(self, "_resolved_local", True):
                self._persist_downloaded_weights()

            if isinstance(self.yolo, YOLO) or isinstance(self.yolo, YOLOWorld):
                try:
                    self.get_logger().info("Trying to fuse model...")
                    self.yolo.fuse()
                except TypeError as e:
                    self.get_logger().warning(f"Fuse skipped: {e}")

            self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

            if self.use_subscription:
                self._sub = self.create_subscription(Image, self.input_image_topic, self.image_cb, self.qos)
                self._stop_evt.clear(); self._new_frame_evt.clear()
                self._worker = threading.Thread(target=self._infer_loop, name="yolo_infer", daemon=True)
                self._worker.start()
            else:
                self._sub = None
                self._worker = None

            # start periodic profiling logger
            if self.profile_window_sec and self.profile_window_sec > 0.0:
                self._prof_timer = self.create_timer(self.profile_window_sec, self._maybe_log_profile)


            self._warmup()

        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist and could not be downloaded.")
            return TransitionCallbackReturn.ERROR
        except Exception as e:
            self.get_logger().error(f"Failed to load model '{self.model}': {e}")
            return TransitionCallbackReturn.ERROR

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        try:
            # stop profiling timer
            if self._prof_timer:
                self._prof_timer.cancel()
                self._prof_timer = None
            self._stop_evt.set(); self._new_frame_evt.set()
            if self._worker: self._worker.join(timeout=2.0)
            self._worker = None
        except Exception:
            pass

        if hasattr(self, "yolo"):
            del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            try: torch.cuda.empty_cache()
            except Exception: pass

        if self._enable_srv:
            self.destroy_service(self._enable_srv); self._enable_srv = None
        if self._sub:
            self.destroy_subscription(self._sub); self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_publisher(self._pub)
        del self.image_qos_profile
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response

    # ---------- helpers ----------
    def _class_allowed(self, class_id_str: str) -> bool:
        try:
            idx = int(class_id_str)
            name = str(self.yolo.names[idx]).lower() if hasattr(self.yolo, "names") else class_id_str
        except Exception:
            name = str(class_id_str).lower()
        if self.allow_classes and name not in self.allow_classes: return False
        if self.block_classes and name in self.block_classes: return False
        return True

    def _make_bbox2d_xywhr(self, xywhr) -> BoundingBox2D:
        bb = BoundingBox2D()
        bb.center = VMPose2D()
        bb.center.position.x = float(xywhr[0])
        bb.center.position.y = float(xywhr[1])
        bb.center.theta = float(xywhr[4]) if len(xywhr) >= 5 else 0.0
        bb.size_x = float(xywhr[2]); bb.size_y = float(xywhr[3])
        return bb

    # --- add inside class YoloNode (near other helpers) ---
    @staticmethod
    def _wrap_pi(theta: float) -> float:
        # wrap to [-pi, pi]
        import math
        t = (theta + math.pi) % (2.0 * math.pi)
        if t < 0: t += 2.0 * math.pi
        return t - math.pi

    @staticmethod
    def _unflip_center_and_theta(cx, cy, bw, bh, theta, img_w, img_h, flip_method):
        fm = flip_method
        # centers back to original orientation
        if fm in (1, -1):  # horizontal
            cx = (img_w - 1) - cx
            if theta is not None:
                theta = np.pi - theta
        if fm in (0, -1):  # vertical
            cy = (img_h - 1) - cy
            if theta is not None:
                theta = -theta
        if theta is not None:
            theta = YoloNode._wrap_pi(theta)
        return cx, cy, bw, bh, theta

    # --- replace _parse_results with this version ---
    def _parse_results(self, results: Results, header) -> Detection2DArray:
        msg = Detection2DArray(); msg.header = header

        # original image geometry
        try:
            img_h, img_w = int(results.orig_shape[0]), int(results.orig_shape[1])
        except Exception:
            img_h = self.imgsz_height
            img_w = self.imgsz_width

        if results.boxes:
            boxes = results.boxes
            cls = boxes.cls.detach().to("cpu", non_blocking=True)
            conf = boxes.conf.detach().to("cpu", non_blocking=True)
            xywh = boxes.xywh.detach().to("cpu", non_blocking=True)  # pixels in image space

            for i in range(xywh.shape[0]):
                bw = float(xywh[i, 2]); bh = float(xywh[i, 3])
                if (bw * bh) < self.min_box_area:
                    continue

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cls[i]))
                hyp.hypothesis.score = float(conf[i])
                if not self._class_allowed(hyp.hypothesis.class_id):
                    continue

                cx = float(xywh[i, 0]); cy = float(xywh[i, 1])
                # unflip back to original if we flipped before inference
                if self.flip_method in (0, 1, -1):
                    cx, cy, bw, bh, _ = YoloNode._unflip_center_and_theta(cx, cy, bw, bh, None, img_w, img_h, self.flip_method)

                det = Detection2D(); det.header = header
                det.bbox = self._make_bbox2d_xywhr([cx, cy, bw, bh, 0.0])
                det.results.append(hyp)
                msg.detections.append(det)

        elif results.obb:
            xywhr = results.obb.xywhr.detach().to("cpu", non_blocking=True)  # [cx, cy, w, h, theta]
            cls = results.obb.cls.detach().to("cpu", non_blocking=True)
            conf = results.obb.conf.detach().to("cpu", non_blocking=True)

            for i in range(xywhr.shape[0]):
                bw = float(xywhr[i, 2]); bh = float(xywhr[i, 3])
                if (bw * bh) < self.min_box_area:
                    continue

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cls[i]))
                hyp.hypothesis.score = float(conf[i])
                if not self._class_allowed(hyp.hypothesis.class_id):
                    continue

                cx = float(xywhr[i, 0]); cy = float(xywhr[i, 1]); th = float(xywhr[i, 4])

                if self.flip_method in (0, 1, -1):
                    cx, cy, bw, bh, th = YoloNode._unflip_center_and_theta(cx, cy, bw, bh, th, img_w, img_h, self.flip_method)

                det = Detection2D(); det.header = header
                det.bbox = self._make_bbox2d_xywhr([cx, cy, bw, bh, th])
                det.results.append(hyp)
                msg.detections.append(det)

        return msg

    # ---------- public one-shot inference ----------
    def inference(self, image_or_msg, *, publish: bool = False) -> Detection2DArray:
        if not hasattr(self, "yolo"):
            raise RuntimeError("Model not loaded. Call trigger_configure() and trigger_activate() first.")
        if isinstance(image_or_msg, Image):
            header = image_or_msg.header
            img = self.cv_bridge.imgmsg_to_cv2(image_or_msg, desired_encoding=self.yolo_encoding)
        else:
            img = image_or_msg
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "yolo_inference"   # or some meaningful name

        if self.flip_method in [0, 1, -1]:
            img = cv2.flip(img, self.flip_method)
        with torch.inference_mode():
            results_list = self.yolo.predict(
                source=img, verbose=False, stream=False,
                conf=self.threshold, iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half, max_det=self.max_det,
                augment=self.augment, agnostic_nms=self.agnostic_nms,
                retina_masks=self.retina_masks, device=self.device,
            )
        res: Results = results_list[0]
        det_msg = self._parse_results(res, header)
        if publish: self._pub.publish(det_msg)
        return det_msg

    # ---------- ROS callbacks ----------
    def image_cb(self, msg: Image) -> None:
        if not self.enable: return
        t_cb0 = self._t()
        if self._last_cb_t is not None: self._prof_add("period", (t_cb0 - self._last_cb_t) * 1e-9)
        self._last_cb_t = t_cb0

        t0 = self._t()
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=self.yolo_encoding)
        if self.flip_method in [0, 1, -1]: img = cv2.flip(img, self.flip_method)
        self._prof_add("convert", (self._t() - t0) * 1e-9)

        with self._frame_lock:
            self._latest_img = img
            self._latest_header = msg.header
        self._new_frame_evt.set()

    # ---------- worker thread ----------
    def _infer_loop(self):
        self.get_logger().info("YOLO worker started")
        while not self._stop_evt.is_set():
            fired = self._new_frame_evt.wait(timeout=0.2)
            if not fired: continue
            with self._frame_lock:
                img = self._latest_img; header = self._latest_header
                self._latest_img = None; self._new_frame_evt.clear()
            if img is None: continue
            t0 = self._t()
            try:
                with torch.inference_mode():
                    results_list = self.yolo.predict(
                        source=img, verbose=False, stream=False,
                        conf=self.threshold, iou=self.iou,
                        imgsz=(self.imgsz_height, self.imgsz_width),
                        half=self.half, max_det=self.max_det,
                        augment=self.augment, agnostic_nms=self.agnostic_nms,
                        retina_masks=self.retina_masks, device=self.device,
                    )
                res: Results = results_list[0]
                self._prof_add("infer", (self._t() - t0) * 1e-9)
                det_msg = self._parse_results(res, header)
                t_pub0 = self._t(); self._pub.publish(det_msg)
                self._prof_add("publish", (self._t() - t_pub0) * 1e-9)
            except Exception as e:
                self.get_logger().error(f"infer loop error: {e}")
            self._prof_add("total_time", (self._t() - t0) * 1e-9)
        self.get_logger().info("YOLO worker stopped")


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
