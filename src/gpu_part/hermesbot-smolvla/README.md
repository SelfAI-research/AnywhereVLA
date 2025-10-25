# SmolVLA

Vision-Language-Action model for robotic control.

## Quick Start

```bash
# Build and run SmolVLA ROS node
docker compose up smolvla_ros

# Interactive development
docker compose up smolvla
```

## Configuration

- **VLA Config**: `config/vla_ros.yaml`
- **Model**: SmolVLA via LeRobot framework

## Platform Support

- **Jetson**: Optimized for ARM64 architecture
- **x86**: Compatible with Intel/AMD systems

## Usage

Send VLA commands via ROS2:
```bash
ros2 topic pub /VLA_start_control std_msgs/msg/String "data: 'Pick up the bottle'" --once
```