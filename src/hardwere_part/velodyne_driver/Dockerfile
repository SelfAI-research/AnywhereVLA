FROM ros:humble-perception

ENV WS_DIR="/velodyne_driver"
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
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-slam-toolbox \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    ros-${ROS_DISTRO}-rmw-fastrtps-dynamic-cpp \
    # ros-${ROS_DISTRO}-velodyne \
    # ros-${ROS_DISTRO}-velodyne-driver \
 && rm -rf /var/lib/apt/lists/*docke

ENV DEBIAN_FRONTEND=dialog
COPY . ${WS_DIR}

RUN echo "source ${WS_DIR}/install/setup.bash" >> ~/.bashrc

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash"

CMD ["/bin/bash", "-c", "source install/setup.bash && ros2 launch velodyne velodyne_run.launch.py"]