#!/usr/bin/env python3
import os, json, cv2, torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Trim host/GPU overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# cuDNN / TF32 / SDPA
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

@dataclass
class VLAConfig:
    ckpt: str
    one_step: bool = True
    fp16: bool = True
    device: Optional[str] = None                    # "cuda" | "cpu" | None(auto)
    image_sizes: Dict[str, Tuple[int, int]] = None  # {key: (H,W)} in RGB

class VLA:
    def __init__(self, cfg: VLAConfig):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (self.device == "cuda" and cfg.fp16) else torch.float32
        self.image_sizes = cfg.image_sizes or {}
        print(f"[info] device={self.device}")

        self.check_view_keys()
        self.initialize()
        self.warmup()

    def check_view_keys(self):
        # Ensure the visual keys match the checkpoint config
        model_json = json.load(open(os.path.join(self.cfg.ckpt, "config.json"), "r"))
        model_keys = [k for k,v in model_json["input_features"].items() if v["type"]=="VISUAL"]
        user_keys = list(self.image_sizes.keys())
        if set(user_keys) != set(model_keys):
            raise ValueError(f"image_sizes keys {user_keys} != model visual keys {model_keys}")

    def initialize(self):
        print(f"[info] loading VLA model...")
        self.policy = SmolVLAPolicy.from_pretrained(self.cfg.ckpt).eval().to(self.device)
        self.policy.requires_grad_(False)  # disable gradients
        self._tok_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.state_dim = int(self.policy.normalize_inputs.buffer_observation_state.mean.shape[-1])
        print(f"[info] VLA loaded successfully, state_dim={self.state_dim}")

    def warmup(self):
        state = torch.zeros((1, self.state_dim), dtype=self.dtype)
        state = state.pin_memory() if self.device == "cuda" else state
        state = state.to(self.device, non_blocking=True)
        batch = {"task": ["warmup"], "observation.state": state}
        for k, (H, W) in self.image_sizes.items():
            img = torch.zeros((1, 3, H, W), dtype=self.dtype).contiguous(memory_format=torch.channels_last)
            img = img.pin_memory() if self.device == "cuda" else img
            batch[k] = img.to(self.device, non_blocking=True)
        self._set_task_tokens("warmup")
        with torch.inference_mode():
            self._predict(batch)

    def infer(self, TEST: str, image_list: List[np.ndarray], state: List[float]) -> List[float]:
        # Map list -> keys in a deterministic order
        keys = list(self.image_sizes.keys())
        if len(image_list) != len(keys):
            raise ValueError(f"Expected {len(keys)} images for keys {keys}, got {len(image_list)}")

        state = torch.as_tensor(state, dtype=self.dtype).unsqueeze(0)
        state = state.pin_memory() if self.device == "cuda" else state
        state = state.to(self.device, non_blocking=True)
        batch = {"task": [TEST], "observation.state": state}
        for rgb, (key, hw) in zip(image_list, self.image_sizes.items()):
            batch[key] = self._prep_img(rgb, hw)
        self._set_task_tokens(TEST)
        with torch.inference_mode():
            out = self._predict(batch)  # (1,A) or (1,T,A)
        return out[0].float().tolist()

    # ---- helpers ----
    def _predict(self, batch):
        if self.cfg.one_step:
            return self.policy.select_action(batch)
        return self.policy.predict_action_chunk(batch)

    def _set_task_tokens(self, text: str):
        # Optional acceleration: only if tokenizer exists
        tok_fn = getattr(self.policy, "language_tokenizer", None)
        if tok_fn is None:
            print("[WARN] No tokenizer")
            return
        tt = self._tok_cache.get(text)
        if tt is None:
            tok = tok_fn(text, return_tensors="pt")
            tt = {k: v.to(self.device) for k, v in tok.items()}
            self._tok_cache[text] = tt
        setattr(self.policy, "_task_tokens", tt)

    def _prep_img(self, rgb: np.ndarray, hw: Tuple[int, int]):
        H, W = hw
        if (rgb.shape[0], rgb.shape[1]) != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().mul_(1/255.0)  # (3,H,W)
        t = t.unsqueeze(0).contiguous(memory_format=torch.channels_last)
        t = t.pin_memory() if self.device == "cuda" else t
        return t.to(self.device, dtype=self.dtype, non_blocking=True)
