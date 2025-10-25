#!/bin/bash

set -e

# AnywhereVLA unified run script

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info(){ echo -e "${BLUE}[RUN]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CPU_DIR="$ROOT_DIR/src/cpu_part"
GPU_DIR="$ROOT_DIR/src/gpu_part"
HW_DIR="$ROOT_DIR/src/hardwere_part"

RUN_HW=true
RUN_GPU=true
RUN_CPU=true
DETACH=false
PLATFORM="auto" # jetson|x86|auto

usage(){
  echo "Usage: $0 [--cpu-only|--gpu-only|--hw-only] [--detach] [--jetson|--x86]"; exit 0;
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only) RUN_GPU=false; RUN_HW=false; shift ;;
    --gpu-only) RUN_CPU=false; RUN_HW=false; shift ;;
    --hw-only|--hardware-only) RUN_CPU=false; RUN_GPU=false; shift ;;
    -d|--detach) DETACH=true; shift ;;
    --jetson) PLATFORM="jetson"; shift ;;
    --x86) PLATFORM="x86"; shift ;;
    -h|--help) usage ;;
    *) err "Unknown arg: $1"; usage ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then err "docker not found"; exit 1; fi
if ! docker compose version >/dev/null 2>&1; then err "docker compose not found"; exit 1; fi

UP_CMD="up"; $DETACH && UP_CMD="up -d"

# 1) Hardware (sensors, motors, gamepad)
if [ "$RUN_HW" = true ]; then
  info "Hardware stack"
  if [ -f "$HW_DIR/docker compose.yaml" ]; then
    (cd "$HW_DIR" && docker compose -f docker compose.yaml $UP_CMD realsense_ros2 velodyne_ros2 motor_control gamepad_control)
  else
    warn "Hardware compose not found"
  fi
fi

# 2) GPU (YOLO, SmolVLA)
if [ "$RUN_GPU" = true ]; then
  info "GPU stack"
  # YOLO
  if [ -d "$GPU_DIR/hermesbot-yolo" ]; then
    YOLO_DIR="$GPU_DIR/hermesbot-yolo"
    if [ "$PLATFORM" = "jetson" ] && [ -f "$YOLO_DIR/docker.jetson/compose.jetson.yml" ]; then
      (cd "$YOLO_DIR" && docker compose -f docker.jetson/compose.jetson.yml $UP_CMD yolo_ros)
    elif [ "$PLATFORM" = "x86" ] && [ -f "$YOLO_DIR/docker.nuc/compose.nuc.yml" ]; then
      (cd "$YOLO_DIR" && docker compose -f docker.nuc/compose.nuc.yml $UP_CMD yolo_ros)
    else
      if [ -f "$YOLO_DIR/docker.jetson/compose.jetson.yml" ]; then
        (cd "$YOLO_DIR" && docker compose -f docker.jetson/compose.jetson.yml $UP_CMD yolo_ros)
      elif [ -f "$YOLO_DIR/docker.nuc/compose.nuc.yml" ]; then
        (cd "$YOLO_DIR" && docker compose -f docker.nuc/compose.nuc.yml $UP_CMD yolo_ros)
      elif [ -f "$YOLO_DIR/docker compose.yml" ]; then
        (cd "$YOLO_DIR" && docker compose -f docker compose.yml $UP_CMD yolo_ros)
      else
        warn "YOLO compose not found"
      fi
    fi
  fi
  # SmolVLA (ros)
  if [ -f "$GPU_DIR/hermesbot-smolvla/docker compose.yml" ]; then
    (cd "$GPU_DIR/hermesbot-smolvla" && docker compose -f docker compose.yml $UP_CMD smolvla_ros)
  else
    warn "SmolVLA compose not found"
  fi
fi

# 3) CPU (SLAM, Maps, Nav, Exploration, Bridges, Behavior)
if [ "$RUN_CPU" = true ]; then
  info "CPU stack"
  if [ -f "$CPU_DIR/docker compose.yaml" ]; then
    # SLAM
    (cd "$CPU_DIR" && docker compose -f docker compose.yaml $UP_CMD fastlivo2)
    # Maps
    (cd "$CPU_DIR" && docker compose -f docker compose.yaml $UP_CMD octomap_ros2 visible_octomap semantic_map)
    # Navigation
    (cd "$CPU_DIR" && docker compose -f docker compose.yaml $UP_CMD hermesbot_nav2 approach_planner)
  else
    warn "CPU compose not found"
  fi
  # Exploration
  if [ -f "$CPU_DIR/hermesbot-active-exploration/docker compose.yml" ]; then
    (cd "$CPU_DIR/hermesbot-active-exploration" && docker compose -f docker compose.yml $UP_CMD)
  fi
  # Bridges + behavior (python)
  EP_DIR="$CPU_DIR/hermesbot-entrypoint/src"
  if [ -d "$EP_DIR" ] && [ "$DETACH" = false ]; then
    info "Starting interface bridges (two terminals recommended)"
    python3 "$EP_DIR/approach_detection_node.py" &
    python3 "$EP_DIR/yolo_interface_bridge.py" &
    python3 "$EP_DIR/entry_point.py"
  else
    warn "Entrypoint scripts not started (detached or not found)"
  fi
fi

ok "Run sequence complete"

echo ""
echo "Examples to send commands:" 
echo "CPU: ros2 topic pub --once /interpreter/commands std_msgs/msg/String \"data: '[{\\\"bottle\\\":\\\"Pick up the bottle and place it into the blue box\\\"}]'\""
echo "GPU: ros2 topic pub /VLA_start_control std_msgs/msg/String \"data: 'Pick up the bottle and place it into the blue box'\" --once"
