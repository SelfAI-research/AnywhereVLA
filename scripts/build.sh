#!/bin/bash

set -e

# AnywhereVLA unified build script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info(){ echo -e "${BLUE}[BUILD]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CPU_DIR="$ROOT_DIR/src/cpu_part"
GPU_DIR="$ROOT_DIR/src/gpu_part"
HW_DIR="$ROOT_DIR/src/hardwere_part"

BUILD_CPU=true
BUILD_GPU=true
BUILD_HW=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only) BUILD_GPU=false; BUILD_HW=false; shift ;;
    --gpu-only) BUILD_CPU=false; BUILD_HW=false; shift ;;
    --hw-only|--hardware-only) BUILD_CPU=false; BUILD_GPU=false; shift ;;
    -h|--help)
      echo "Usage: $0 [--cpu-only|--gpu-only|--hw-only]"; exit 0 ;;
    *) err "Unknown arg: $1"; exit 1 ;;
  esac
done

# Pre-checks
if ! command -v docker >/dev/null 2>&1; then err "docker not found"; exit 1; fi
if ! docker compose version >/dev/null 2>&1; then err "docker compose not found"; exit 1; fi

# CPU
if [ "$BUILD_CPU" = true ]; then
  info "Building CPU stack"
  # Build ros2-base first to avoid circular dependencies
  if [ -f "$CPU_DIR/hermesbot-semantic-map/docker-compose.yml" ]; then
    info "Building ros2-base image first"
    (cd "$CPU_DIR/hermesbot-semantic-map" && docker compose build build_ros2_base)
  fi
  if [ -f "$CPU_DIR/docker-compose.yaml" ]; then
    (cd "$CPU_DIR" && docker compose -f docker-compose.yaml build)
    ok "CPU stack built"
  else
    warn "CPU compose not found: $CPU_DIR/docker-compose.yaml"
  fi
  # Active exploration
  if [ -f "$CPU_DIR/hermesbot-active-exploration/docker-compose.yml" ]; then
    (cd "$CPU_DIR/hermesbot-active-exploration" && docker compose -f docker-compose.yml build)
  fi
  # Entrypoint has only python; nothing to build
fi

# GPU
if [ "$BUILD_GPU" = true ]; then
  info "Building GPU stack"
  # smolVLA
  if [ -f "$GPU_DIR/hermesbot-smolvla/docker-compose.yml" ]; then
    (cd "$GPU_DIR/hermesbot-smolvla" && docker compose -f docker-compose.yml build smolvla smolvla_ros smolvlax86 || docker compose -f docker-compose.yml build)
  else
    warn "SmolVLA compose not found"
  fi
  # YOLO (both platforms if available)
  if [ -f "$GPU_DIR/hermesbot-yolo/docker-compose.yml" ]; then
    (cd "$GPU_DIR/hermesbot-yolo" && docker compose -f docker-compose.yml build || true)
  fi
  if [ -f "$GPU_DIR/hermesbot-yolo/docker.jetson/compose.jetson.yml" ]; then
    (cd "$GPU_DIR/hermesbot-yolo" && docker compose -f docker.jetson/compose.jetson.yml build)
  fi
  if [ -f "$GPU_DIR/hermesbot-yolo/docker.nuc/compose.nuc.yml" ]; then
    (cd "$GPU_DIR/hermesbot-yolo" && docker compose -f docker.nuc/compose.nuc.yml build)
  fi
  ok "GPU stack built (where available)"
fi

# Hardware
if [ "$BUILD_HW" = true ]; then
  info "Building hardware stack"
  if [ -f "$HW_DIR/docker-compose.yaml" ]; then
    (cd "$HW_DIR" && docker compose -f docker-compose.yaml build)
    ok "Hardware stack built"
  else
    warn "Hardware compose not found: $HW_DIR/docker-compose.yaml"
  fi
fi

ok "Build complete"
