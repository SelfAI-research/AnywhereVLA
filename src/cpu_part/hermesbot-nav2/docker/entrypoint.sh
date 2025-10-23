#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash

exec ros2 launch $(pwd)/launch/navigation_launch.py params_file:=$(pwd)/config/nav2_params.yaml