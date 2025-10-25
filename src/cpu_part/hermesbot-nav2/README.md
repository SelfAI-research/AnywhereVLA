# Nav2 (Collapsed-Frame) Bringup

This brings up Nav2 with a single world frame: `camera_init`. Your robot frame is `base_link` and the LiDAR is `camera_link`. A static TF (`base_link`→`camera_link`) is published from env vars.

## Prereqs
- Your modules publish:
  - `/map` (nav_msgs/OccupancyGrid, `frame_id: camera_init` or `map`; we use `camera_init` globally)
  - `/cloud_registered` (sensor_msgs/PointCloud2, `frame_id: camera_link`)
  - TF: `camera_init -> base_link` must be valid (via odom or your SLAM). We also publish `base_link -> camera_link`.

## Run
```bash
docker compose up --build -d



# Navigation step:
```
ros2 run tf2_ros tf2_echo camera_init base_link   # leave this running in another tab to see TF is OK
ros2 run tf2_ros tf2_echo camera_init base_link   &


# In a new tab, replace X,Y with the first line's translation values (robot pose)
X=0; Y=0  # <-- put real numbers here
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
"{pose: { header: {frame_id: camera_init}, pose: { position: {x: $((X+3)), y: $Y, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
```


tasks for the next day:

1. check the nav2 modules and understand the responsibilities of each
2. check the topic to send the goal pose
3. check the control law