# hermesbot-vla — SmolVLA (Docker, GPU)

Minimal, JetPack-matched container for **Jetson Orin NX 16GB (JP 6.0, L4T r36.3)** that runs:
- **SmolVLA** via **LeRobot** (`.[smolvla]`)

## How To Run
```bash
# Optional perf while testing
sudo nvpmodel -m 2 && sudo jetson_clocks
docker compose up smolvla 
```
