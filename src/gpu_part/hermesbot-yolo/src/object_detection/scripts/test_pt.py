#!/usr/bin/env python3
import argparse, time, statistics, numpy as np, cv2, torch
from ultralytics import YOLO

def preprocess(img_path, height, width, dtype=torch.float16):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)  # (W,H)
    x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    t = torch.from_numpy(x).to(dtype)
    return t

def stats(a):
    return dict(
        mean=round(statistics.mean(a), 3),
        median=round(statistics.median(a), 3),
        p90=round(float(np.percentile(a, 90)), 3),
        p95=round(float(np.percentile(a, 95)), 3),
        min=round(min(a), 3),
        max=round(max(a), 3),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data/yolo_weights/yolo12s.pt")
    ap.add_argument("--image", type=str, default="/workspace/ros2_ws/src/scripts/bus.jpg")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters",  type=int, default=100)
    ap.add_argument("--half", action="store_true", default=True)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    # Load Ultralytics YOLO; access raw torch model
    yolom = YOLO(args.model)
    net = yolom.model.eval().cuda()
    if args.half:
        net = net.half()
    try:
        yolom.fuse()
    except Exception:
        pass

    x = preprocess(args.image, args.height, args.width,
                   torch.float16 if args.half else torch.float32).cuda(non_blocking=True)

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = net(x)
        torch.cuda.synchronize()

    # Benchmark
    start_evt, stop_evt = torch.cuda.Event(True), torch.cuda.Event(True)
    t_comp, t_e2e = [], []
    with torch.inference_mode():
        for _ in range(args.iters):
            t0 = time.perf_counter()
            start_evt.record()
            y = net(x)
            stop_evt.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_comp.append(start_evt.elapsed_time(stop_evt))       # ms, GPU compute
            t_e2e.append((t1 - t0) * 1000.0)                     # ms, host wall

    print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Dev: {torch.cuda.get_device_name(0)}")
    print("Input shape:", (1, 3, args.height, args.width), "| half:", args.half)
    print("GPU compute (ms):", stats(t_comp))
    print("End-to-end  (ms):", stats(t_e2e))

if __name__ == "__main__":
    main()
