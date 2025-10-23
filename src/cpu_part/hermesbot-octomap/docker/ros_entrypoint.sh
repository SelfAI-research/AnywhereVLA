#!/bin/bash
set -e

# ROS 2 environment
source "/opt/ros/humble/setup.bash"

# If you later add an overlay workspace:
if [ -f "/app/install/setup.bash" ]; then
  source "/app/install/setup.bash"
fi

exec "$@"
