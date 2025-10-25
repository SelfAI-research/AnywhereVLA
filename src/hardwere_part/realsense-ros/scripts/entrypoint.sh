#!/usr/bin/env bash
set -e

[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f /realsense_ros/install/setup.bash ] && source /realsense_ros/install/setup.bash

GRIPPER_SERIAL="049322070172"
UPPER_SERIAL="032622071350"
SIDE_SERIAL="135122071874"

gripper_camera=true
upper_camera=true
side_camera=true

for kv in "$@"; do
  case "$kv" in
    gripper_camera=*) gripper_camera="${kv#*=}";;
    upper_camera=*)   upper_camera="${kv#*=}";;
    side_camera=*)    side_camera="${kv#*=}";;
    *) echo "Unknown arg: $kv";;
  esac
done

to_bool(){ case "${1,,}" in true|1|yes|on) echo true;; *) echo false;; esac; }

run_cam() {
  local role="$1" serial="$2"

  ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -r __ns:=/${role} \
    -r __node:=realsense_${role} \
    -p camera_name:=camera \
    -p "serial_no:='${serial}'" \
    -p enable_color:=true -p enable_depth:=false -p enable_infra1:=false -p enable_infra2:=false \
    -p enable_gyro:=true -p enable_accel:=true -p unite_imu_method:=2 \
    -p rgb_camera.color_profile:=640x480x15 -p rgb_camera.color_format:=RGB8 \
    -p align_depth.enable:=false -p pointcloud.enable:=false -p enable_rgbd:=false -p enable_sync:=false \
    -p hold_back_imu_for_frames:=false -p publish_tf:=true -p tf_publish_rate:=0.0 &
  PIDS+=($!)
}


PIDS=()
[ "$(to_bool "$gripper_camera")" = true ] && run_cam gripper "$GRIPPER_SERIAL"
[ "$(to_bool "$upper_camera")"   = true ] && run_cam upper   "$UPPER_SERIAL"
[ "$(to_bool "$side_camera")"    = true ] && run_cam side    "$SIDE_SERIAL"

if [ ${#PIDS[@]} -eq 0 ]; then
  echo "No cameras enabled. Use e.g.: r_camera=true upper_camera=false side_camera=true"
  exit 1
fi

trap 'kill "${PIDS[@]}" 2>/dev/null || true' INT TERM
wait
