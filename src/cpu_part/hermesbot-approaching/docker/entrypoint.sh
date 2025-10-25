#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH=/app/src:$PYTHONPATH
source /opt/ros/humble/setup.bash

python3 src/approach_planner_node.py --ros-args --params-file config/params.yaml