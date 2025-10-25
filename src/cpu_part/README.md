# CPU Components

Processing and control components for AnywhereVLA robot.

## Components

- **SLAM** (`hermesbot-slam/`): FAST-LIVO2 simultaneous localization and mapping
- **Semantic Map** (`hermesbot-semantic-map/`): Semantic understanding and mapping
- **Octomap** (`hermesbot-octomap/`): 3D mapping and obstacle detection
- **Navigation** (`hermesbot-nav2/`): Path planning and navigation
- **Approaching** (`hermesbot-approaching/`): Object approaching and manipulation planning
- **Exploration** (`hermesbot-active-exploration/`): Autonomous exploration strategies
- **Entry Point** (`hermesbot-entrypoint/`): Interface bridges and behavior control

## Quick Start

```bash
# Build CPU components
./scripts/build.sh --cpu-only

# Run CPU components
./scripts/run.sh --cpu-only
```

## Individual Components

### SLAM (FAST-LIVO2)
```bash
cd hermesbot-slam
docker compose up fastlivo_core
```

### Semantic Mapping
```bash
cd hermesbot-semantic-map
docker compose up semantic_map
```

### Navigation
```bash
cd hermesbot-nav2
docker compose up hermesbot_nav2
```

### Exploration
```bash
cd hermesbot-active-exploration
docker compose up
```

### Interface Bridges
```bash
cd hermesbot-entrypoint/src
python3 approach_detection_node.py &
python3 yolo_interface_bridge.py &
python3 entry_point.py
```

## Configuration

- SLAM config: `hermesbot-slam/src/FAST-LIVO2/config/`
- Semantic map: `hermesbot-semantic-map/src/semantic_map/config/`
- Navigation: `hermesbot-nav2/config/nav2_params.yaml`
- Exploration: `hermesbot-active-exploration/configs/config.yaml`
