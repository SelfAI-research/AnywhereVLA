FROM ros:humble
SHELL ["/bin/bash","-c"]

RUN apt-get update && apt-get install -y \
    ros-humble-nav2-msgs ros-humble-geometry-msgs ros-humble-nav-msgs \
    ros-humble-tf2-ros ros-humble-visualization-msgs ros-humble-nav2-msgs ros-humble-tf-transformations  ros-humble-tf2-geometry-msgs \
    python3-pip \
    ros-humble-rmw-cyclonedds-cpp tree
    # && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
   && python3 -m pip install --no-cache-dir \
   "numpy>=2,<3" "scipy>=1.15" \
   &&  python3 -m pip install --no-cache-dir --upgrade --ignore-installed "transforms3d>=0.4.2"

WORKDIR /app
COPY . /app
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc 

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["./entrypoint.sh"]
