# Use the official ROS 2 Humble core
FROM ros:humble-ros-core

# Install system dependencies first (apt packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
      python3-evdev \
      python3-pymodbus \
      ros-humble-tf2 \
      ros-humble-tf2-ros \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-diagnostic-updater && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies in separate layer
RUN pip3 install --no-cache-dir \
      hjson \
      numpy \
      scipy \
      coloredlogs \
      requests \
      pynput \
      transforms3d

# Set working dir for your code
WORKDIR /app
COPY src/ src/

# On container start, source ROS 2 and launch the node
ENTRYPOINT ["/bin/bash", "-lc", "source /opt/ros/humble/setup.bash && exec python3 /app/src/main_motors_node.py"]