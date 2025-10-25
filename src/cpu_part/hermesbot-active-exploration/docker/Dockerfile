# docker/Dockerfile
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Minimal build + runtime deps
# - nav2 + bringup: Nav2 servers and example launch files
# - nav2-simple-commander: Python API used by many explorer wrappers
# - colcon + build-essential: build workspace
# - git, rosdep: fetch/resolve deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-numpy python3-yaml \
    ros-humble-rclpy \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-simple-commander \
    ros-humble-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# yacs from PyPI
RUN pip3 install --no-cache-dir yacs scipy

# Workspace
WORKDIR /app
RUN mkdir -p src

# Copy package sources + optional local scripts/configs
COPY scripts/ scripts/
COPY configs/ configs/

# Entrypoint script you provided
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Sensible defaults; override in compose via environment if needed
ENV ROS_LOG_DIR=/app/logs

ENTRYPOINT ["/ros_entrypoint.sh"]

# Default command runs the explorer; you can override in compose
CMD ["bash", "-lc", "python3 scripts/explorer_node.py"]