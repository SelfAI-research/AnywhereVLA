# Robust Approach Planner

## Command in development:

```
ros2 param get /planner_server GridBased.allow_unknown 

ros2 action send_goal /compute_path_to_pose nav2_msgs/action/ComputePathToPose "{goal: {header: {frame_id: 'camera_init'}, pose: {position: {x: -3, y: 0}, orientation: {w: 1.0}}}}"

ros2 topic pub -1 /approaching_object_point geometry_msgs/PointStamped \
"{header: {frame_id: 'camera_init'}, point: {x: 2, y: -3, z: 0.0}}"



check this:


ros2 param set /planner_server GridBased.allow_unknown false
ros2 param set /planner_server Smac2D.allow_unknown false
ros2 param set /planner_server SmacHybrid.allow_unknown false

 ./check_goal.py 0 -4 0

 
```

The Robust Approach Planner is a ROS 2 (Humble) node that computes a path for a robot to approach a target pose from an optimal direction. It uses costmap information to determine the best approach orientation (surface normal) and calls the Nav2 planner to get a safe path to an offset "approach" pose in front of the target.

## Features

- **Costmap-based orientation** – Calculates the approach orientation by estimating the surface normal from the occupancy grid (obstacles) near the target position.
- **Robust planning** – Tries multiple approach angles if the first approach path is not feasible, adjusting orientation in configurable increments.
- **Modular design** – Components for normal estimation, planner interfacing, TF transforms, and visualization are separated for easy maintenance or replacement.
- **Debug visualization** – Publishes an RViz visualization marker (arrow) indicating the final approach pose and orientation relative to the target.

## File Structure

```text
approach_planner/
├── config/
│   └── params.yaml              # ROS 2 parameters for the approach planner node
├── launch/
│   └── approaching_planner.launch.py  # Launch file to start the approach planner node with params
├── docker/
│   ├── Dockerfile              # Docker image definition for the approach planner
│   └── docker-compose.yaml     # Docker Compose configuration for containerized deployment
├── scripts/
│   ├── approach_planner_node.py   # Main ROS 2 node script
│   ├── coord_utils.py            # Coordinate frame and quaternion utilities (TF helper)
│   ├── costmap_utils.py          # Costmap subscriber and query utilities
│   ├── normal_estimators.py      # Normal estimation strategies for approach orientation
│   ├── planner_client.py         # Action client to Nav2 planner (ComputePathToPose)
│   └── markers.py                # Marker publisher for visualization in RViz
└── tests/
    ├── test_core.py              # Core logic unit tests (normal estimation, planning attempts)
    └── test_smoke_node.py        # Smoke test for the ROS node integration with mocks
```

## Usage

Run with ROS 2: You can launch the approach planner node using the provided ROS 2 launch file:

```sh
ros2 launch approach_planner approaching_planner.launch.py
```
This will start the node with parameters loaded from config/params.yaml. The node listens for incoming target poses on the approach_goal topic (geometry_msgs/PoseStamped). When a target pose is received, it computes an approach path and publishes the resulting path to approach_path (nav_msgs/Path).

- Parameters: Key configurable parameters (see params.yaml):

    - costmap_topic: Topic name of the occupancy grid to use for obstacle data (e.g., /global_costmap/costmap).

    - approach_distance: Distance (meters) to stand off from the target when approaching.

    - angle_increment_deg: Incremental angle (degrees) to adjust orientation if the initial approach path is not feasible.

    - num_attempts: Maximum number of orientation attempts (including the initial) for planning.

    - planner_action: Name of the Nav2 planner action (ComputePathToPose) to use.

    - planner_id: Planner plugin ID to request (leave blank for default global planner).

- Visualization: The node publishes a visualization marker on the approach_markers topic. In RViz, add a Marker display subscribing to approach_markers to see an arrow indicating the final approach pose and orientation relative to the target.

## Running Tests

This package includes both unit tests and integration (smoke) tests. The tests use dummy components to simulate planner responses and tf transforms:

- Core unit tests (tests/test_core.py): Validate normal vector computation from a costmap and the approach planning logic (multiple orientation attempts) using mock planner and normal estimator classes.

- Smoke test (tests/test_smoke_node.py): Uses a dummy ROS node context with mocked planner, costmap, and marker publisher to ensure the approach planner node processes a goal message end-to-end without errors.

You can run the tests with:
```
colcon test --packages-select approach_planner
```

This will execute the tests and report results. Ensure you have the ROS 2 testing tools (pytest) installed.

## Docker Deployment

A Dockerfile and Compose configuration are provided to containerize the approach planner:

- The Dockerfile builds an image based on ROS 2 Humble, installing necessary dependencies and copying the package code.

- Use docker-compose.yaml to run the container. By default it uses host networking so the container can communicate with a Nav2 stack running on the host.
