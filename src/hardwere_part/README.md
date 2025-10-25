# Hardware Components

Hardware drivers and control interfaces for AnywhereVLA robot.

## Components

- **Motors** (`motors/`): Motor control and robot movement
- **RealSense** (`realsense-ros/`): Intel RealSense camera integration  
- **Velodyne** (`velodyne_driver/`): LiDAR sensor integration
- **Gamepad** (`gamepad/`): Manual control interface

## Quick Start

```bash
# Build all hardware components
docker compose build

# Run all hardware services
docker compose up realsense_ros2 velodyne_ros2 motor_control gamepad_control

# Run specific components
docker compose up realsense_ros2  # Camera only
docker compose up velodyne_ros2   # LiDAR only
docker compose up motor_control   # Motors only
docker compose up gamepad_control # Gamepad only
```

## Requirements

- Hardware devices properly connected
- USB device permissions configured
- Appropriate device drivers installed

## Configuration

- Motor configs: `motors/src/configs/`
- Camera settings: `realsense-ros/config/realsense_minimal.yaml`
- LiDAR params: `velodyne_driver/mapper_params_online_async.yaml`