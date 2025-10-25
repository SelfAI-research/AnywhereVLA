# AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2509.21006)
[![Email](https://img.shields.io/badge/✉️-Email-informational?style=for-the-badge)](mailto:artem.voronov@skoltech.ru?subject=AnywhereVLA)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

<hr style="border: 2px solid gray;"></hr>

![Teaser](https://raw.githubusercontent.com/SelfAI-research/AnywhereVLA/main/.github/workflows/teaser.png)

## 1. Introduction

AnywhereVLA is a comprehensive robotics system that integrates Vision-Language-Action (VLA) models with advanced SLAM, navigation, and manipulation capabilities. The system enables robots to understand natural language commands, navigate complex environments, and perform manipulation tasks through a unified pipeline combining CPU processing, GPU-accelerated AI, and hardware control.

The system features:
- **Semantic SLAM**: Advanced mapping with semantic understanding
- **Active Exploration**: Intelligent exploration strategies for unknown environments  
- **Vision-Language-Action**: Natural language command processing and execution
- **Multi-modal Integration**: Seamless coordination between sensors, AI models, and actuators
- **Real-time Performance**: Optimized for both Jetson and x86 platforms

### 1.1 Related Paper

Paper submitted to ICRA 2026.

### 1.2 Project Website

Visit our project website: [AnywhereVLA Landing Page](https://selfai-research.github.io/AnywhereVLA/)

## 2. System Architecture

![Architecture](https://raw.githubusercontent.com/SelfAI-research/AnywhereVLA/main/.github/workflows/architecture.png)

The AnywhereVLA code is organized into three main components:

- **Hardware Part** (`src/hardwere_part/`): Sensor drivers, motor control, and gamepad interface
- **GPU Part** (`src/gpu_part/`): AI models including SmolVLA and YOLO object detection
- **CPU Part** (`src/cpu_part/`): SLAM, navigation, exploration, and semantic mapping

## 3. Dependencies

AnywhereVLA is tested on Ubuntu 20.04/22.04. Please install the following before compilation:

1. **Docker & Docker Compose**: Latest version with `docker compose` support
   - [Docker Installation Guide](https://docs.docker.com/engine/install/ubuntu/)
   - [Docker Compose Installation](https://docs.docker.com/compose/install/)
2. **NVIDIA Docker** (for GPU components): `nvidia-docker2`
   - [NVIDIA Docker Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 4. Quick Start

### Build the System

```bash
# Build all components
./scripts/build.sh

# Build specific parts
./scripts/build.sh --cpu-only
./scripts/build.sh --gpu-only  
./scripts/build.sh --hardware-only
```

### Run the System

```bash
# Launch complete pipeline
./scripts/run.sh

# Run specific components
./scripts/run.sh --cpu-only
./scripts/run.sh --gpu-only
./scripts/run.sh --hardware-only

# Run in background
./scripts/run.sh --detach
```

## 5. Dataset

Download the AnywhereVLA dataset to reproduce semantic SLAM and exploration results:

```bash
# Download dataset (replace with actual release URL)
wget https://github.com/your-org/AnywhereVLA/releases/download/v1-data/anywherevla_office_data_v1.tar.gz
tar -xzf anywherevla_office_data_v1.tar.gz
```

### Dataset Description

The dataset contains a ROS2 bag file (`anywherevla_office_data_v1`) recorded in an office corridor environment:

**Recording Details:**
- **Duration**: 96.2 seconds
- **Size**: 1.0 GiB


**Recorded Topics:**
- `/velodyne_points` (659 messages): Velodyne LiDAR point clouds (`sensor_msgs/msg/PointCloud2`)
- `/camera/camera/imu` (19,258 messages): IMU data (`sensor_msgs/msg/Imu`)
- `/camera/camera/color/image_raw` (726 messages): RGB camera images (`sensor_msgs/msg/Image`)
- `/camera/camera/color/camera_info` (726 messages): Camera calibration info (`sensor_msgs/msg/CameraInfo`)

**Environment & Objects:**

- Office corridor setting
- Objects of interest: bottles and bananas on tables
- Robot navigate through the environment

**Usage:**
When running the semantic SLAM component, the system will detect and label objects with IDs in RViz, enabling visualization of semantic understanding capabilities.

![Semantic SLAM Visualization](rviz_semantic.png)

## 6. Platform Support

- **Jetson Platforms**: Optimized for NVIDIA Jetson with ARM64 architecture
- **x86 Platforms**: Compatible with Intel/AMD systems
- **Hardware**: Intel RealSense cameras, Velodyne LiDAR, custom motor controllers

## 7. Configuration

Key configuration files:
- `src/cpu_part/hermesbot-semantic-map/config/`: Semantic mapping parameters
- `src/gpu_part/hermesbot-smolvla/config/vla_ros.yaml`: VLA model settings
- `src/hardwere_part/motors/src/configs/`: Motor control parameters

## 8. Troubleshooting

### Common Issues

1. **Docker Compose Error**: Ensure you're using `docker compose` (not `docker-compose`)
2. **GPU Not Detected**: Install `nvidia-docker2` and restart Docker
3. **Hardware Not Found**: Check USB connections and device permissions

### Getting Help

- Check individual component README files in each subdirectory
- Review Docker logs: `docker logs <container_name>`
- Ensure all prerequisites are installed

## 9. Citation

If you use AnywhereVLA in your research, please cite our work:

```bibtex
@article{gubernatorov2025anywherevla,
  title={AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation},
  author={Gubernatorov, Konstantin and Voronov, Artem and Voronov, Roman and Pasynkov, Sergei and Perminov, Stepan and Guo, Ziang and Tsetserukou, Dzmitry},
  journal={arXiv preprint arXiv:2509.21006},
  year={2025}
}
```

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 11. Acknowledgments

- We thank the authors of FastLIO2, SmolVLA, and other open-source projects for their contributions
- Special thanks to the robotics community for continuous feedback and improvements

## Contact

For questions and collaboration, please contact: [artem.voronov@skoltech.ru](mailto:artem.voronov@skoltech.ru?subject=AnywhereVLA)


---

**Note**: This repository contains research code. Please refer to the [paper](https://selfai-research.github.io/AnywhereVLA/) for detailed technical descriptions and evaluation results.
