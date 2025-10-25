# AnywhereVLA Scripts

Helpers to build images and run the stack.

## Build

```bash
# Build all
./scripts/build.sh

# Build specific parts
./scripts/build.sh --cpu-only
./scripts/build.sh --gpu-only
./scripts/build.sh --hardware-only
```

## Run

```bash
# Launch full pipeline (hardware -> GPU -> CPU)
./scripts/run.sh

# Detach (run containers in background)
./scripts/run.sh --detach

# Launch specific parts
./scripts/run.sh --cpu-only
./scripts/run.sh --gpu-only
./scripts/run.sh --hardware-only

# Select platform for GPU (auto by default)
./scripts/run.sh --jetson
./scripts/run.sh --x86
```

## Commands

```bash
# CPU behavior
ros2 topic pub --once /interpreter/commands std_msgs/msg/String "data: '[{\"bottle\":\"Pick up the bottle and place it into the blue box\"}]'"

# VLA control
ros2 topic pub /VLA_start_control std_msgs/msg/String "data: 'Pick up the bottle and place it into the blue box'" --once
```

## Notes
- Requires Docker and docker-compose.
- Hardware compose is in `src/hardwere_part`.
- CPU compose is in `src/cpu_part`.
- GPU stacks are in `src/gpu_part`.
