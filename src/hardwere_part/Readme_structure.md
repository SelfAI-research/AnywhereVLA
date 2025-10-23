# Module Overview

A brief description of each module in this project.

## Hardware

### - motors (ROS Noetic)

Implements motor control logic. Interfaces with motor hardware via ROS topics and services, handling velocity and position commands.
#### Description: TODO

## Drivers

### - realsense-ros

ROS 2 wrapper for Intel RealSense cameras driver.
Builds and launches the `realsense2_camera_node` to stream depth, color, and IMU data.

### - velodyne_driver

ROS 2 driver for Velodyne LiDAR sensors. Builds and launches the Velodyne driver node to publish point clouds and scans.

## Software

### - gamepad

Contains the code for handling gamepad input for robot motion.

Control:
 - **X**: `enable/disable` gamepad controlling (default is `disabled`)
 - **Left Stick**: `forward/backward` linear movement
 - **Right Stick**: `Left/Right` rotation
