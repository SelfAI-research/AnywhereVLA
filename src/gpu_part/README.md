# GPU Components

AI models and GPU-accelerated processing for AnywhereVLA.

## Components

- **SmolVLA** (`hermesbot-smolvla/`): Vision-Language-Action model
- **YOLO** (`hermesbot-yolo/`): Object detection and recognition

## Quick Start

```bash
# Build GPU components
./scripts/build.sh --gpu-only

# Run GPU components
./scripts/run.sh --gpu-only

# Platform-specific builds
./scripts/run.sh --jetson  # For Jetson platforms
./scripts/run.sh --x86     # For x86 platforms
```

## SmolVLA

Vision-Language-Action model for robot control.

```bash
cd hermesbot-smolvla
docker compose up smolvla_ros
```

**Config**: `config/vla_ros.yaml`

## YOLO

Object detection with ROS2 integration.

```bash
cd hermesbot-yolo
# Jetson
docker compose -f docker.jetson/compose.jetson.yml up yolo_ros
# x86
docker compose -f docker.nuc/compose.nuc.yml up yolo_ros
```

**Config**: `src/object_detection/yolo_bringup/config/yolo_params.yaml`

## Requirements

- NVIDIA GPU with proper drivers
- nvidia-docker2 installed and configured
- Sufficient GPU memory for models
