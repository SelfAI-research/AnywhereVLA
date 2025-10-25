#!/usr/bin/env python3
import argparse, numpy as np, cv2, tensorrt as trt
import pycuda.driver as cuda
from typing import Dict

def _npdtype(t): return trt.nptype(t)

class TRTOnce:
    def __init__(self, engine_path: str, H: int, W: int, device_id: int = 0):
        self.H, self.W = int(H), int(W)
        cuda.init()
        self.dev  = cuda.Device(device_id)
        self.pctx = self.dev.retain_primary_context()
        self.pctx.push()
        try:
            self.log = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f:
                self.rt  = trt.Runtime(self.log)
                self.eng = self.rt.deserialize_cuda_engine(f.read())
            if self.eng is None:
                raise RuntimeError(f"Failed to load engine '{engine_path}' (TRT {trt.__version__})")
            self.ctx = self.eng.create_execution_context()
            self.stream = cuda.Stream()

            # If engine has multiple profiles, pick 0 and then set shape.
            if hasattr(self.ctx, "set_optimization_profile_async"):
                self.ctx.set_optimization_profile_async(0, self.stream.handle)

            # Input setup
            self.inputs, self.outputs = [], []
            for i in range(self.eng.num_io_tensors):
                name = self.eng.get_tensor_name(i)
                (self.inputs if self.eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)
            if len(self.inputs) != 1:
                raise RuntimeError(f"Expected 1 input, got {self.inputs}")
            self.inp = self.inputs[0]
            self.ctx.set_input_shape(self.inp, (1, 3, self.H, self.W))
            self.in_dtype = _npdtype(self.eng.get_tensor_dtype(self.inp))
            self.d_in = cuda.mem_alloc(int(np.prod((1,3,self.H,self.W))) * np.dtype(self.in_dtype).itemsize)
            self.ctx.set_tensor_address(self.inp, int(self.d_in))

            # Outputs (sizes are known after input shape is set)
            self.host_out: Dict[str, np.ndarray] = {}
            self.d_out: Dict[str, int] = {}
            for name in self.outputs:
                shp = tuple(self.ctx.get_tensor_shape(name))
                dt  = _npdtype(self.eng.get_tensor_dtype(name))
                dbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
                dptr = cuda.mem_alloc(dbytes)
                self.ctx.set_tensor_address(name, int(dptr))
                self.d_out[name] = int(dptr)
                self.host_out[name] = np.empty(shp, dtype=dt)

            # Try to find EfficientNMS
            low = {k.lower(): k for k in self.outputs}
            def find(*keys):
                for kk in keys:
                    for lk, orig in low.items():
                        if kk in lk: return orig
                return None
            self.k_num    = find("num_det", "num_detections")
            self.k_boxes  = find("boxes", "nms_boxes")
            self.k_scores = find("scores", "nms_scores")
            self.k_cls    = find("classes", "nms_classes")
            self.has_nms  = all([self.k_num, self.k_boxes, self.k_scores, self.k_cls])

        finally:
            self.pctx.pop()  # pop until we actually run

    @staticmethod
    def _prep_bgr(img: np.ndarray, H: int, W: int, out_dtype):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0).transpose(2,0,1)[None]
        if out_dtype == np.float16: x = x.astype(np.float16, copy=False)
        return x.copy()

    def infer(self, image_path: str, conf_th: float = 0.25):
        self.pctx.push()
        try:
            img = cv2.imread(image_path)
            if img is None: raise FileNotFoundError(image_path)
            x = self._prep_bgr(img, self.H, self.W, self.in_dtype)

            cuda.memcpy_htod_async(self.d_in, x, self.stream)
            if hasattr(self.ctx, "execute_async_v3"):
                self.ctx.execute_async_v3(self.stream.handle)
            elif hasattr(self.ctx, "enqueue_v3"):
                self.ctx.enqueue_v3(self.stream.handle)
            else:
                # Not recommended with tensor API, but kept as last resort.
                self.ctx.execute_v2()

            for name in self.outputs:
                cuda.memcpy_dtoh_async(self.host_out[name], self.d_out[name], self.stream)
            self.stream.synchronize()

            if self.has_nms:
                n = int(self.host_out[self.k_num][0])
                boxes  = self.host_out[self.k_boxes][0][:n].astype(np.float32)   # xyxy
                scores = self.host_out[self.k_scores][0][:n].astype(np.float32)
                clss   = self.host_out[self.k_cls][0][:n].astype(np.int32)
                m = scores >= conf_th
                return boxes[m], scores[m], clss[m]

            # Fallback: no NMS head -> return empty (or add your own decoder)
            return np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32)
        finally:
            self.pctx.pop()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, default="/data/yolo_weights/yolo12s_480x640_fp16.engine")
    ap.add_argument("--image",  type=str, default="/workspace/ros2_ws/src/scripts/bus.jpg")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--conf",   type=float, default=0.25)
    args = ap.parse_args()

    det = TRTOnce(args.engine, args.height, args.width)
    xyxy, scores, clss = det.infer(args.image, conf_th=args.conf)

    print(f"TRT: {trt.__version__}")
    print(f"Input: (1,3,{args.height},{args.width}) dtype={det.in_dtype.__name__}")
    print("Outputs:", {k: v.shape for k, v in det.host_out.items()})
    print(f"Detections: {len(scores)}")
    for i in range(len(scores)):
        x1,y1,x2,y2 = xyxy[i].tolist()
        print(f"[{i:02d}] class={int(clss[i])} score={scores[i]:.3f} box=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")

if __name__ == "__main__":
    main()
