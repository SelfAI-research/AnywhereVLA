#!/usr/bin/env python3
import time, cv2, torch
from smolVLA import VLA, VLAConfig
 
BENCH_REPEAT      = 10
# ====== configs ======
ONE_STEP    = True
CKPT        = "/workspace/model/smolvla_hermesbot/checkpoints/001000/pretrained_model"
TASK        = "Pick up the bottle and place it into the blue box"
VIEWS = [
    ("observation.images.wrist", "/workspace/lerobot/bottle.jpg", (640, 480)),
    ("observation.images.base",  "/workspace/lerobot/bottle.jpg", (640, 480)),
    ("observation.images.side",  "/workspace/lerobot/bottle.jpg", (640, 480)),
]
# =======================

def dev(): return "cuda" if torch.cuda.is_available() else "cpu"

def load_img(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def main():
    device = dev()
    image_sizes = {key: hw for key, _, hw in VIEWS} 
    imgs = [load_img(path) for _, path, _ in VIEWS]

    cfg = VLAConfig(
        ckpt=CKPT, one_step=ONE_STEP, fp16=True, 
        device=device, image_sizes=image_sizes,
    )
    vla = VLA(cfg)

    test_state = [0.0] * vla.state_dim
    test_task  = "pick up the bottle"

    # Measure
    times = []
    with torch.inference_mode():
        for _ in range(BENCH_REPEAT):
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            out = vla.infer(test_task, imgs, test_state)
            if device == "cuda": torch.cuda.synchronize()
            times.append((time.time() - t0) * 1e3)

    print(f"[time ms] mean={sum(times)/len(times):.1f}  best={min(times):.1f}  worst={max(times):.1f}")
    print(f"[info] out example: {out}")
    # print(f"[info] times: {times}")

if __name__ == "__main__":
    main()
