# YOLO Object Detection

ROS2 wrapper for YOLO object detection with multi-platform support.

## Quick Start

```bash
# Build
make build

# Run (auto-detects platform)
make run

# Manual platform selection
# Jetson
docker compose -f docker.jetson/compose.jetson.yml up yolo_ros
# x86
docker compose -f docker.nuc/compose.nuc.yml up yolo_ros
```

## Configuration

- **YOLO Params**: `src/object_detection/yolo_bringup/config/yolo_params.yaml`
- **RViz Config**: `src/object_detection/yolo_bringup/config/rviz_cfg.rviz`

## Features

- Real-time object detection
- ROS2 topic integration
- Multi-platform support (Jetson/x86)
- Configurable model parameters