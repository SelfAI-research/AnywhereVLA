#!/usr/bin/env python3
# GPL-3.0-or-later

import os, cv2, shutil, threading, numpy as np
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from rclpy.parameter import Parameter

import torch
from cv_bridge import CvBridge

from std_msgs.msg import Header
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from vision_msgs.msg import Pose2D as VMPose2D
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose


# =========================
# Lightweight result types
# =========================

class SimpleBoxes:
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh  # [N,4] (pixels)
        self.conf = conf  # [N]
        self.cls  = cls   # [N]

class SimpleResult:
    def __init__(self, orig_shape: Tuple[int,int], boxes: Optional[SimpleBoxes] = None, obb: Any = None):
        self.orig_shape = orig_shape
        self.boxes = boxes
        self.obb = obb


# =========================
# Backends
# =========================

class BaseBackend:
    def __init__(self, logger=None): self._logger = logger
    @property
    def names(self) -> Optional[List[str]]: return None
    def fuse(self): pass
    def warmup(self, h: int, w: int): pass
    def predict(self, img_bgr: np.ndarray, **kw) -> Any: raise NotImplementedError


class PTBackend(BaseBackend):
    def __init__(self, model_type: str, model_spec: str, logger=None):
        super().__init__(logger)
        # lazy import so TRT-only envs are fine
        from ultralytics import YOLO, YOLOWorld, YOLOE
        type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}
        if model_type not in type_to_model:
            raise ValueError(f"Unknown PT model_type='{model_type}'. Use one of: {list(type_to_model.keys())}")
        self._yolo = type_to_model[model_type](model_spec)

    @property
    def names(self):
        try:
            return list(self._yolo.names.values()) if isinstance(self._yolo.names, dict) else list(self._yolo.names)
        except Exception:
            return None

    def fuse(self):
        try:
            self._yolo.fuse()
            if self._logger: self._logger.info("PTBackend: model fused")
        except TypeError as e:
            if self._logger: self._logger.warning(f"PTBackend: fuse skipped: {e}")

    def warmup(self, h: int, w: int):
        try:
            with torch.inference_mode():
                _ = self._yolo.predict(
                    source=torch.zeros(1, 3, h, w),
                    imgsz=(h, w), conf=0.05, iou=0.5, half=False,
                    max_det=1, augment=False, agnostic_nms=False,
                    retina_masks=False, device="cpu", verbose=False, stream=False
                )
        except Exception as e:
            if self._logger: self._logger.warning(f"PTBackend: warmup skipped ({e})")

    def predict(self, img_bgr: np.ndarray, **kw):
        with torch.inference_mode():
            return self._yolo.predict(
                source=img_bgr, verbose=False, stream=False,
                conf=kw.get("conf", 0.5), iou=kw.get("iou", 0.5),
                imgsz=kw.get("imgsz", (640,640)), half=kw.get("half", False),
                max_det=kw.get("max_det", 300), augment=False,
                agnostic_nms=kw.get("agnostic_nms", False),
                retina_masks=kw.get("retina_masks", False),
                device=kw.get("device","cpu"),
            )[0]


class TRTBackend(BaseBackend):
    def __init__(self, engine_path: str, logger=None,
                 score_th: float = 0.6, nms_iou: float = 0.5,
                 pre_nms_topk: int = 600, topk_per_class: int = 50):
        super().__init__(logger)
        import tensorrt as trt
        import pycuda.driver as cuda
        import threading

        self.trt = trt
        self.cuda = cuda
        self.cuda.init()
        self._tls = threading.local()  # per-thread stream

        # primary context
        self.dev = self.cuda.Device(0)
        self.pctx = self.dev.retain_primary_context()
        self.pctx.push()
        try:
            self.logger = self.trt.Logger(self.trt.Logger.ERROR)
            with open(engine_path, "rb") as f:
                self.runtime = self.trt.Runtime(self.logger)
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to load engine '{engine_path}' (TRT={self.trt.__version__})")
            self.ctx = self.engine.create_execution_context()

            self.inputs, self.outputs = [], []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                    self.inputs.append(name)
                else:
                    self.outputs.append(name)
            if len(self.inputs) != 1:
                raise RuntimeError(f"TRTBackend expects exactly one input, got {self.inputs}")

            self.inp_name = self.inputs[0]
            self.in_dtype = self._npdtype(self.engine.get_tensor_dtype(self.inp_name))

            self.cfg_shape = None
            self.d_in_alloc = None
            self.d_in_ptr = None
            self.d_out_alloc, self.d_out_ptr, self.host_out = {}, {}, {}

            # knobs
            self.score_th = float(score_th)
            self.nms_iou = float(nms_iou)
            self.pre_nms_topk = int(pre_nms_topk)
            self.topk_per_class = int(topk_per_class)
        finally:
            self.pctx.pop()

    # context helpers
    def _push(self):
        self.pctx.push()
        if not hasattr(self._tls, "stream"):
            self._tls.stream = self.cuda.Stream()
        return self._tls.stream
    def _pop(self): self.pctx.pop()

    @staticmethod
    def _npdtype(trt_dtype):
        import numpy as _np, tensorrt as _trt
        return {
            _trt.DataType.FLOAT: _np.float32,
            _trt.DataType.HALF:  _np.float16,
            _trt.DataType.INT32: _np.int32,
            _trt.DataType.INT8:  _np.int8,
            _trt.DataType.BOOL:  _np.bool_,
        }.get(trt_dtype, _np.float32)

    def _validate_profile(self, h: int, w: int):
        try:
            mn, opt, mx = self.engine.get_tensor_profile_shape(self.inp_name, 0)
            if len(mx) == 4:
                Hmn, Wmn, Hmx, Wmx = int(mn[2]), int(mn[3]), int(mx[2]), int(mx[3])
                if not (Hmn <= h <= Hmx and Wmn <= w <= Wmx):
                    raise ValueError(f"Input {(1,3,h,w)} outside profile range [{(1,3,Hmn,Wmn)}..{(1,3,Hmx,Wmx)}]")
        except Exception:
            pass

    def _configure_bindings(self, h: int, w: int):
        if self.cfg_shape == (h, w): return
        self._validate_profile(h, w)

        self.ctx.set_input_shape(self.inp_name, (1, 3, h, w))
        nbytes = int(np.prod((1,3,h,w))) * np.dtype(self.in_dtype).itemsize
        self.d_in_alloc = self.cuda.mem_alloc(nbytes)
        self.d_in_ptr = int(self.d_in_alloc)
        self.ctx.set_tensor_address(self.inp_name, self.d_in_ptr)

        self.host_out.clear(); self.d_out_alloc.clear(); self.d_out_ptr.clear()
        for out in self.outputs:
            shp = tuple(self.ctx.get_tensor_shape(out))
            dt  = self._npdtype(self.engine.get_tensor_dtype(out))
            nbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
            d_out = self.cuda.mem_alloc(nbytes)
            self.ctx.set_tensor_address(out, int(d_out))
            self.d_out_alloc[out] = d_out
            self.d_out_ptr[out] = int(d_out)
            self.host_out[out] = np.empty(shp, dtype=dt, order="C")

        self.cfg_shape = (h, w)
        if self._logger: self._logger.info(f"TRTBackend: configured {(h,w)}; outputs: {{ {', '.join([f'{k}: {v.shape}' for k,v in self.host_out.items()])} }}")

    @staticmethod
    def _preprocess_bgr(img_bgr: np.ndarray, h: int, w: int, dtype) -> np.ndarray:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return x.astype(dtype, copy=False).copy()

    @staticmethod
    def _iou_xywh(box, boxes):
        cx, cy, w, h = box
        x1 = cx - 0.5*w; y1 = cy - 0.5*h
        x2 = cx + 0.5*w; y2 = cy + 0.5*h
        bx1 = boxes[:,0] - 0.5*boxes[:,2]
        by1 = boxes[:,1] - 0.5*boxes[:,3]
        bx2 = boxes[:,0] + 0.5*boxes[:,2]
        by2 = boxes[:,1] + 0.5*boxes[:,3]
        xx1 = np.maximum(x1, bx1); yy1 = np.maximum(y1, by1)
        xx2 = np.minimum(x2, bx2); yy2 = np.minimum(y2, by2)
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area_i = (x2-x1) * (y2-y1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_i + area_b - inter + 1e-9)

    def _nms_classwise_fast(self, xywh, scores, classes, iou_th, topk_per_class):
        keep = []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            if idx.size == 0: continue
            order = scores[idx].argsort()[::-1]
            idx = idx[order]
            kept_c = []
            while idx.size > 0 and len(kept_c) < topk_per_class:
                i = idx[0]
                kept_c.append(i)
                if idx.size == 1: break
                iou = self._iou_xywh(xywh[i], xywh[idx[1:]])
                idx = idx[1:][iou <= iou_th]
            keep.extend(kept_c)
        return np.array(keep, dtype=np.int32)

    def warmup(self, h: int, w: int):
        s = self._push()
        try:
            self._configure_bindings(h, w)
            x = self._preprocess_bgr(np.zeros((h, w, 3), np.uint8), h, w, self.in_dtype)
            self.cuda.memcpy_htod_async(self.d_in_alloc, x, s)
            if hasattr(self.ctx, "execute_async_v3"): self.ctx.execute_async_v3(s.handle)
            else: self.ctx.execute_v2(bindings=None)
            for out in self.outputs:
                self.cuda.memcpy_dtoh_async(self.host_out[out], self.d_out_alloc[out], s)
            s.synchronize()
        finally:
            self._pop()

    def _postprocess(self, img_h: int, img_w: int, conf_th: float) -> SimpleResult:
        result = SimpleResult(orig_shape=(img_h, img_w), boxes=None)
        keys = set(self.host_out.keys())

        # EfficientNMS path
        if {"num_dets","boxes","scores","classes"}.issubset(keys):
            n = int(self.host_out["num_dets"][0])
            xyxy   = self.host_out["boxes"][0][:n].astype(np.float32, copy=False)
            scores = self.host_out["scores"][0][:n].astype(np.float32, copy=False)
            cls    = self.host_out["classes"][0][:n].astype(np.int32,  copy=False)
            m = scores >= conf_th
            xyxy, scores, cls = xyxy[m], scores[m], cls[m]
            if xyxy.size == 0:
                result.boxes = SimpleBoxes(np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32))
                return result
            cx = (xyxy[:,0]+xyxy[:,2])*0.5; cy = (xyxy[:,1]+xyxy[:,3])*0.5
            w  = (xyxy[:,2]-xyxy[:,0]);     h  = (xyxy[:,3]-xyxy[:,1])
            xywh = np.stack([cx,cy,w,h], axis=-1)
            result.boxes = SimpleBoxes(xywh=xywh, conf=scores, cls=cls)
            return result

        # Raw YOLO head (e.g., 1x84x6300)
        if len(keys) == 1:
            name = next(iter(keys))
            out = self.host_out[name]
            if out.ndim == 3 and out.shape[1] >= 6:
                arr = out[0].astype(np.float32, copy=False).T  # (N,84)
                xywh = arr[:, :4]
                obj  = 1.0 / (1.0 + np.exp(-arr[:, 4]))
                cls_logits = arr[:, 5:]
                cls_prob   = 1.0 / (1.0 + np.exp(-cls_logits))
                cls_id = np.argmax(cls_prob, axis=1).astype(np.int32)
                cls_p  = cls_prob[np.arange(cls_prob.shape[0]), cls_id]
                scores = obj * cls_p

                xywh[:,0] *= float(img_w); xywh[:,1] *= float(img_h)
                xywh[:,2] *= float(img_w); xywh[:,3] *= float(img_h)

                base_th = max(conf_th, self.score_th)
                m = scores >= base_th
                if not np.any(m):
                    result.boxes = SimpleBoxes(np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32))
                    return result
                xywh = xywh[m]; scores = scores[m]; cls_id = cls_id[m]

                if xywh.shape[0] > self.pre_nms_topk:
                    k = self.pre_nms_topk
                    idx = np.argpartition(scores, -k)[-k:]
                    xywh = xywh[idx]; scores = scores[idx]; cls_id = cls_id[idx]

                keep = self._nms_classwise_fast(xywh, scores, cls_id, self.nms_iou, self.topk_per_class)
                xywh = xywh[keep]; scores = scores[keep]; cls_id = cls_id[keep]

                result.boxes = SimpleBoxes(xywh=xywh.astype(np.float32),
                                           conf=scores.astype(np.float32),
                                           cls=cls_id.astype(np.int32))
                return result

        if self._logger: self._logger.error(f"TRTBackend: unsupported outputs {keys}")
        result.boxes = SimpleBoxes(np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32))
        return result

    @property
    def names(self): return None

    def predict(self, img_bgr: np.ndarray, *, conf: float, iou: float,
                imgsz: Tuple[int,int], half: bool, max_det: int,
                agnostic_nms: bool, retina_masks: bool, device: str) -> SimpleResult:
        h, w = int(imgsz[0]), int(imgsz[1])
        s = self._push()
        try:
            self._configure_bindings(h, w)
            x = self._preprocess_bgr(img_bgr, h, w, self.in_dtype)
            self.cuda.memcpy_htod_async(self.d_in_alloc, x, s)
            if hasattr(self.ctx, "execute_async_v3"): self.ctx.execute_async_v3(s.handle)
            else: self.ctx.execute_v2(bindings=None)
            for out in self.outputs:
                self.cuda.memcpy_dtoh_async(self.host_out[out], self.d_out_alloc[out], s)
            s.synchronize()
            return self._postprocess(img_bgr.shape[0], img_bgr.shape[1], float(conf))
        except Exception as e:
            if self._logger: self._logger.error(f"TRTBackend: inference failed: {e}")
            return SimpleResult(orig_shape=(img_bgr.shape[0], img_bgr.shape[1]),
                                boxes=SimpleBoxes(np.zeros((0,4),np.float32),
                                                  np.zeros((0,),np.float32),
                                                  np.zeros((0,),np.int32)))
        finally:
            self._pop()


# =========================
# ROS2 Node
# =========================

class YoloNode(LifecycleNode):
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("yolo_node")

        # core & model
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("weights_dir", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("prefer_backend", "auto")  # auto|pt|trt
        self.declare_parameter("tracker", "bytetrack.yaml")

        # inference
        self.declare_parameter("enable", True)
        self.declare_parameter("flip_method", -2)
        self.declare_parameter("threshold", 0.45)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 200)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        self.declare_parameter("min_box_area", 16.0)
        self.declare_parameter("allow_classes", "")
        self.declare_parameter("block_classes", "")

        # TRT postproc knobs
        self.declare_parameter("trt_score_th", -1.0)         # -1 -> use threshold
        self.declare_parameter("trt_iou_th", 0.5)
        self.declare_parameter("trt_pre_nms_topk", 600)
        self.declare_parameter("trt_topk_per_class", 50)

        # topics & QoS
        self.declare_parameter("input_image_topic", "/image_raw")
        self.declare_parameter("detections_topic", "detections")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("use_subscription", True)

        # launch flags
        self.declare_parameter("use_detect", True)
        self.declare_parameter("use_tracking", False)
        self.declare_parameter("use_debug", True)
        self.declare_parameter("profile_log_every", 50)
        self.declare_parameter("profile_window_sec", 5.0)
        self.declare_parameter("namespace", "")

        # state
        self._prof: Dict[str, list] = {}
        self._prof_count: int = 0
        self._last_cb_t = None

        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._frame_lock = threading.Lock()
        self._new_frame_evt = threading.Event()
        self._latest_img = None
        self._latest_header = None

        self._enable_srv = None
        self._sub = None

        self._backend: Optional[BaseBackend] = None
        self._class_names: Optional[List[str]] = None
        self._resolved_local = True

        if initial_params:
            self.set_parameters([Parameter(name=k, value=v) for k, v in initial_params.items()])

    # profiling
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
        self._prof_count += 1
        if self.profile_log_every > 0 and (self._prof_count % self.profile_log_every) == 0:
            self._log_profile_medians()

    # lifecycle
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.model_type    = self.get_parameter("model_type").value
        self.model         = self.get_parameter("model").value
        self.weights_dir   = self.get_parameter("weights_dir").value
        self.device        = self.get_parameter("device").value
        self.prefer_backend= str(self.get_parameter("prefer_backend").value).lower()
        self.tracker       = self.get_parameter("tracker").value

        self.enable        = self.get_parameter("enable").value
        self.flip_method   = int(self.get_parameter("flip_method").value)
        self.threshold     = float(self.get_parameter("threshold").value)
        self.iou           = float(self.get_parameter("iou").value)
        self.imgsz_height  = int(self.get_parameter("imgsz_height").value)
        self.imgsz_width   = int(self.get_parameter("imgsz_width").value)
        self.half          = bool(self.get_parameter("half").value)
        self.max_det       = int(self.get_parameter("max_det").value)
        self.augment       = bool(self.get_parameter("augment").value)
        self.agnostic_nms  = bool(self.get_parameter("agnostic_nms").value)
        self.retina_masks  = bool(self.get_parameter("retina_masks").value)
        self.min_box_area  = float(self.get_parameter("min_box_area").value)

        self.trt_score_th        = float(self.get_parameter("trt_score_th").value)
        self.trt_iou_th          = float(self.get_parameter("trt_iou_th").value)
        self.trt_pre_nms_topk    = int(self.get_parameter("trt_pre_nms_topk").value)
        self.trt_topk_per_class  = int(self.get_parameter("trt_topk_per_class").value)

        allow_raw = self.get_parameter("allow_classes").value
        self.allow_classes = [s.strip().lower() for s in str(allow_raw).split(",") if str(s).strip()]
        block_raw = self.get_parameter("block_classes").value
        self.block_classes = [s.strip().lower() for s in str(block_raw).split(",") if str(s).strip()]

        self.detections_topic  = self.get_parameter("detections_topic").value
        self.input_image_topic = self.get_parameter("input_image_topic").value
        self.yolo_encoding     = self.get_parameter("yolo_encoding").value
        self.use_subscription  = bool(self.get_parameter("use_subscription").value)
        self.reliability       = int(self.get_parameter("image_reliability").value)

        self.use_detect   = bool(self.get_parameter("use_detect").value)
        self.use_tracking = bool(self.get_parameter("use_tracking").value)
        self.use_debug    = bool(self.get_parameter("use_debug").value)
        self.profile_log_every = int(self.get_parameter("profile_log_every").value)
        self.profile_window_sec = float(self.get_parameter("profile_window_sec").value)
        self.namespace    = self.get_parameter("namespace").value

        rel = QoSReliabilityPolicy.SYSTEM_DEFAULT
        if   self.reliability == 1: rel = QoSReliabilityPolicy.RELIABLE
        elif self.reliability == 2: rel = QoSReliabilityPolicy.BEST_EFFORT

        self.qos = QoSProfile(reliability=rel, history=QoSHistoryPolicy.KEEP_LAST,
                              durability=QoSDurabilityPolicy.VOLATILE, depth=10)

        self._pub = self.create_lifecycle_publisher(Detection2DArray, self.detections_topic, self.qos)
        self.cv_bridge = CvBridge()

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

    def _persist_downloaded_weights(self, backend: BaseBackend):
        if not self.weights_dir: return
        if not isinstance(backend, PTBackend): return
        try:
            wd = Path(self.weights_dir); wd.mkdir(parents=True, exist_ok=True)
            src = None
            for cand in [getattr(backend._yolo, "ckpt_path", None), getattr(backend._yolo, "weights", None)]:
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
            dst = Path(self.weights_dir) / self.model
            try:
                if Path(src).resolve() == Path(dst).resolve():
                    self.get_logger().info("Weights already in target folder.")
                    return
            except Exception:
                pass
            if not Path(dst).exists() or Path(src).stat().st_size != Path(dst).stat().st_size:
                shutil.copy2(src, dst)
                self.get_logger().info(f"Saved downloaded weights to: {dst}")
        except Exception as e:
            self.get_logger().warning(f"Failed to save downloaded weights: {e}")

    def _choose_backend(self, model_spec: str) -> BaseBackend:
        ext = Path(model_spec).suffix.lower()
        prefer = self.prefer_backend
        if prefer == "trt" or ext == ".engine":
            self.get_logger().info("Selecting TensorRT backend")
            return TRTBackend(
                model_spec,
                logger=self.get_logger(),
                score_th=(self.trt_score_th if self.trt_score_th >= 0 else self.threshold),
                nms_iou=self.trt_iou_th,
                pre_nms_topk=self.trt_pre_nms_topk,
                topk_per_class=self.trt_topk_per_class,
            )
        self.get_logger().info("Selecting PyTorch/Ultralytics backend")
        return PTBackend(self.model_type, model_spec, logger=self.get_logger())

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
        try:
            model_spec = self._resolve_model_spec()
            self._backend = self._choose_backend(model_spec)
            self._class_names = self._backend.names

            # fuse if PT
            try: self._backend.fuse()
            except Exception as e: self.get_logger().warning(f"Fuse skipped: {e}")

            # warmup BEFORE spinning worker
            self._backend.warmup(self.imgsz_height, self.imgsz_width)

            self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)
            if self.use_subscription and self.use_detect:
                self._sub = self.create_subscription(Image, self.input_image_topic, self.image_cb, self.qos)
                self._stop_evt.clear(); self._new_frame_evt.clear()
                self._worker = threading.Thread(target=self._infer_loop, name="yolo_infer", daemon=True)
                self._worker.start()
            else:
                self._sub = None; self._worker = None

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
            self._stop_evt.set(); self._new_frame_evt.set()
            if self._worker: self._worker.join(timeout=2.0)
            self._worker = None
        except Exception:
            pass
        self._backend = None
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            try: torch.cuda.empty_cache()
            except Exception: pass
        if self._enable_srv: self.destroy_service(self._enable_srv); self._enable_srv = None
        if self._sub: self.destroy_subscription(self._sub); self._sub = None
        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        try:
            if self._pub: self.destroy_publisher(self._pub)
        except Exception:
            pass
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

    # helpers
    def _class_allowed(self, class_id_str: str) -> bool:
        try:
            idx = int(class_id_str)
            name = (self._class_names[idx].lower() if self._class_names and 0 <= idx < len(self._class_names)
                    else class_id_str)
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

    @staticmethod
    def _wrap_pi(theta: float) -> float:
        import math
        t = (theta + math.pi) % (2.0 * math.pi)
        if t < 0: t += 2.0 * math.pi
        return t - math.pi

    @staticmethod
    def _unflip_center_and_theta(cx, cy, bw, bh, theta, img_w, img_h, fm):
        if fm in (1, -1):  # horizontal
            cx = (img_w - 1) - cx
            if theta is not None: theta = np.pi - theta
        if fm in (0, -1):  # vertical
            cy = (img_h - 1) - cy
            if theta is not None: theta = -theta
        if theta is not None:
            theta = YoloNode._wrap_pi(theta)
        return cx, cy, bw, bh, theta

    @staticmethod
    def _to_numpy(x):
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                return x.detach().to("cpu", non_blocking=True).numpy()
        except Exception:
            pass
        return np.asarray(x) if x is not None else None

    def _parse_results(self, results: Any, header) -> Detection2DArray:
        msg = Detection2DArray(); msg.header = header
        try:
            img_h, img_w = int(results.orig_shape[0]), int(results.orig_shape[1])
        except Exception:
            img_h = self.imgsz_height; img_w = self.imgsz_width

        # boxes path (PT or TRT)
        boxes_obj = getattr(results, "boxes", None)
        if boxes_obj is not None:
            xywh = self._to_numpy(getattr(boxes_obj, "xywh", None))
            cls  = self._to_numpy(getattr(boxes_obj, "cls", None))
            conf = self._to_numpy(getattr(boxes_obj, "conf", None))
            if xywh is None or cls is None or conf is None: return msg
            for i in range(xywh.shape[0]):
                bw = float(xywh[i, 2]); bh = float(xywh[i, 3])
                if (bw * bh) < self.min_box_area: continue
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cls[i])); hyp.hypothesis.score = float(conf[i])
                if not self._class_allowed(hyp.hypothesis.class_id): continue
                cx = float(xywh[i, 0]); cy = float(xywh[i, 1])
                if self.flip_method in (0,1,-1):
                    cx, cy, bw, bh, _ = YoloNode._unflip_center_and_theta(cx, cy, bw, bh, None, img_w, img_h, self.flip_method)
                det = Detection2D(); det.header = header
                det.bbox = self._make_bbox2d_xywhr([cx, cy, bw, bh, 0.0])
                det.results.append(hyp); msg.detections.append(det)
            return msg

        # OBB path (PT)
        obb = getattr(results, "obb", None)
        if obb is not None:
            xywhr = self._to_numpy(getattr(obb, "xywhr", None))
            cls   = self._to_numpy(getattr(obb, "cls", None))
            conf  = self._to_numpy(getattr(obb, "conf", None))
            if xywhr is None or cls is None or conf is None: return msg
            for i in range(xywhr.shape[0]):
                bw = float(xywhr[i, 2]); bh = float(xywhr[i, 3])
                if (bw * bh) < self.min_box_area: continue
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cls[i])); hyp.hypothesis.score = float(conf[i])
                if not self._class_allowed(hyp.hypothesis.class_id): continue
                cx = float(xywhr[i, 0]); cy = float(xywhr[i, 1]); th = float(xywhr[i, 4])
                if self.flip_method in (0,1,-1):
                    cx, cy, bw, bh, th = YoloNode._unflip_center_and_theta(cx, cy, bw, bh, th, img_w, img_h, self.flip_method)
                det = Detection2D(); det.header = header
                det.bbox = self._make_bbox2d_xywhr([cx, cy, bw, bh, th])
                msg.detections.append(det)
            return msg

        return msg

    # one-shot inference
    def inference(self, image_or_msg, *, publish: bool = False) -> Detection2DArray:
        if self._backend is None:
            raise RuntimeError("Model not loaded. Call trigger_configure() and trigger_activate() first.")
        if isinstance(image_or_msg, Image):
            header = image_or_msg.header
            img = self.cv_bridge.imgmsg_to_cv2(image_or_msg, desired_encoding=self.yolo_encoding)
        else:
            img = image_or_msg
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "yolo_inference"

        if self.flip_method in [0, 1, -1]:
            img = cv2.flip(img, self.flip_method)
        res = self._backend.predict(
            img_bgr=img, conf=self.threshold, iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half, max_det=self.max_det,
            agnostic_nms=self.agnostic_nms, retina_masks=self.retina_masks,
            device=self.device
        )
        det_msg = self._parse_results(res, header)
        if publish: self._pub.publish(det_msg)
        return det_msg

    # ROS callbacks
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

    # worker
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
                res = self._backend.predict(
                    img_bgr=img, conf=self.threshold, iou=self.iou,
                    imgsz=(self.imgsz_height, self.imgsz_width),
                    half=self.half, max_det=self.max_det,
                    agnostic_nms=self.agnostic_nms, retina_masks=self.retina_masks,
                    device=self.device
                )
                self._prof_add("infer", (self._t() - t0) * 1e-9)
                det_msg = self._parse_results(res, header)
                t_pub0 = self._t(); self._pub.publish(det_msg)
                self._prof_add("publish", (self._t() - t_pub0) * 1e-9)
            except Exception as e:
                self.get_logger().error(f"infer loop error: {e}")
            self._prof_add("total_time", (self._t() - t0) * 1e-9)
            self._maybe_log_profile()
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
