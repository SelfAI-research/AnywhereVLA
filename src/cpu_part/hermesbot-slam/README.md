# 🧠 hermesbot-SLAM

There is wrapper ower fast-livo2 that build metric map and provide accurate odometry.

## Configs:

Fastlivo configs: [hermes.yaml](./src/FAST-LIVO2/config/hermes.yaml)
Tf static publisher configs: [frames.yaml](./src/FAST-LIVO2/config/frames.yaml)

> For now we disable camera modality

🔌 **Subscribed Topics**


| Topic Name       | Topic Value                     | Type                          | Description                          |
|------------------|----------------------------------|-------------------------------|--------------------------------------|
| LiDAR Scan       | `/velodyne_points`                 | `sensor_msgs/msg/PointCloud2` | Raw point cloud from VLP-16 LiDAR   |
| IMU   | `/camera/camera/imu`                         | `sensor_msgs/msg/Imu`       | IMU measurments to undistort lidar |

---
📤 **Published Topics**

| Topic Name           | Topic Value                   | Type                             | Description                               |
|----------------------|-------------------------------|----------------------------------|-------------------------------------------|
| Robot Pose           | `/aft_mapped_to_init`                       | `nav_msgs/msg/Odometry`  | Pose of robot in map frame      |
| Registered Cloud     | `/cloud_registered`       | `sensor_msgs/msg/PointCloud2`    | Global-aligned undistored scan in map frame|
| Effected Cloud    | `/cloud_effected`    | `sensor_msgs/msg/PointCloud2`    | Motion-compensated LiDAR point cloud      |
| some ros2_tf     | `camera_init -> camera_link`            | `geometry_msgs.msg.TransformStamped`     | Real-time tf from map to robot       |



## Run:

1. Build
```bash
docker compose build
```

2. Run SLAM
```bash
docker compose up fastlivo_core rviz
```

3. Play rosbag
```bash
docker compose run play_bag
```