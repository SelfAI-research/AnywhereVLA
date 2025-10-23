FROM ros:humble-perception

ENV WS_DIR="/realsense_ros"
WORKDIR ${WS_DIR}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        libpcap-dev \
        libboost-all-dev \
        build-essential \
        cmake \
        git-all \
        software-properties-common \
        ros-humble-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get -y install \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-librealsense2* \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    ros-${ROS_DISTRO}-rmw-fastrtps-dynamic-cpp \
 && rm -rf /var/lib/apt/lists/*docke

ENV DEBIAN_FRONTEND=dialog
COPY . ${WS_DIR}

RUN echo "source ${WS_DIR}/install/setup.bash" >> ~/.bashrc

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# CMD ["/bin/bash","-lc","source install/setup.bash && \
#   ros2 run realsense2_camera realsense2_camera_node --ros-args \
#   -p camera_name:=camera_front \
#   -p serial_no:=0123456789 \
#   -p camera_namespace:=camera \
#   -p enable_color:=true -p enable_depth:=false -p enable_infra1:=false -p enable_infra2:=false \
#   -p enable_gyro:=true -p enable_accel:=true -p unite_imu_method:=2 \
#   -p rgb_camera.color_profile:=640x480x15 -p rgb_camera.color_format:=RGB8 \
#   -p align_depth.enable:=false -p pointcloud.enable:=false -p enable_rgbd:=false -p enable_sync:=false \
#   -p hold_back_imu_for_frames:=false -p publish_tf:=true -p tf_publish_rate:=0.0"]