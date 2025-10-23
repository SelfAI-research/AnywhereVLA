# Use the official ROS 2 Humble core
FROM ros:humble-ros-core

# Install gamepad and ros dependencies
RUN apt-get update && \
    apt-get install -y \
      python3-evdev \
      ros-humble-geometry-msgs \
      ros-humble-std-msgs \
      ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Set working dir for your code
WORKDIR /app
COPY src/ src/

# On container start, source ROS 2 and launch the node
ENTRYPOINT ["/bin/bash", "-lc", "source /opt/ros/humble/setup.bash && exec python3 /app/src/gamepad_node.py"]