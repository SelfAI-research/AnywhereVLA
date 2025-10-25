#!/usr/bin/env python3
import argparse, time, statistics, numpy as np, cv2, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

def npdtype(t): return trt.nptype(t)

def load_engine(path: str) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.ERROR)
    with open(path, "rb") as f:
        rt = trt.Runtime(logger)
        eng = rt.deserialize_cuda_engine(f.read())
    if eng is None:
        raise RuntimeError(f"Failed to load engine '{path}'. TRT={trt.__version__}")
    return eng

def preprocess(img_path: str, height: int, width: int, dtype=np.float16) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return x.astype(dtype, copy=False).copy()

def enqueue(ctx, stream_handle, bindings=None):
    return ctx.execute_v2(bindings=bindings)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, default="/data/yolo_weights/yolo12s_480x640_fp16.engine")
    ap.add_argument("--image", type=str, default="/workspace/ros2_ws/src/scripts/bus.jpg")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters",  type=int, default=100)
    args = ap.parse_args()

    H, W = args.height, args.width
    eng = load_engine(args.engine)
    ctx = eng.create_execution_context()
    stream = cuda.Stream()

    # I/O tensor discovery (TensorRT 10 tensor API)
    inputs, outputs = [], []
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    assert len(inputs) == 1, f"Expected 1 input, got {inputs}"
    inp = inputs[0]

    # Fix input shape and dtypes
    ctx.set_input_shape(inp, (1, 3, H, W))
    in_dtype = npdtype(eng.get_tensor_dtype(inp))
    x = preprocess(args.image, H, W, np.float16 if in_dtype == np.float16 else np.float32)

    # Device allocations and binding of tensor addresses
    d_in = cuda.mem_alloc(x.nbytes)
    ctx.set_tensor_address(inp, int(d_in))

    host_out, dev_ptr = {}, {inp: int(d_in)}
    for out in outputs:
        shp = tuple(ctx.get_tensor_shape(out))  # resolved after set_input_shape
        dt  = npdtype(eng.get_tensor_dtype(out))
        nbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
        d_out = cuda.mem_alloc(nbytes)
        ctx.set_tensor_address(out, int(d_out))
        dev_ptr[out] = int(d_out)
        host_out[out] = np.empty(shp, dtype=dt)

    # Warmup
    for _ in range(args.warmup):
        cuda.memcpy_htod_async(d_in, x, stream)
        enqueue(ctx, stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(host_out[out], dev_ptr[out], stream)
        stream.synchronize()

    # Benchmark
    t_comp, t_e2e = [], []
    start_evt, stop_evt = cuda.Event(), cuda.Event()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_in, x, stream)
        start_evt.record(stream)
        enqueue(ctx, stream.handle)
        stop_evt.record(stream)
        for out in outputs:
            cuda.memcpy_dtoh_async(host_out[out], dev_ptr[out], stream)
        stream.synchronize()
        t1 = time.perf_counter()
        t_comp.append(start_evt.time_till(stop_evt))
        t_e2e.append((t1 - t0) * 1000.0)

    def stats(a):
        return dict(
            mean=round(statistics.mean(a),3),
            median=round(statistics.median(a),3),
            p90=round(float(np.percentile(a,90)),3),
            p95=round(float(np.percentile(a,95)),3),
            min=round(min(a),3),
            max=round(max(a),3),
        )

    print("TRT:", trt.__version__)
    print("Input:", (1,3,H,W), "dtype:", in_dtype.__name__)
    print("Outputs:", {k: v.shape for k, v in host_out.items()})
    print("GPU compute (ms):", stats(t_comp))
    print("End-to-end  (ms):", stats(t_e2e))

if __name__ == "__main__":
    main()
