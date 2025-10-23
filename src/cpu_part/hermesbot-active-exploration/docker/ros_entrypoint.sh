#!/bin/bash
set -e

# ROS 2 environment
source "/opt/ros/humble/setup.bash"

exec "$@"
