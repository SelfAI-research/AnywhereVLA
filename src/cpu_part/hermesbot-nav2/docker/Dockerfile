FROM ros:humble
SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-regulated-pure-pursuit-controller \
    ros-humble-nav2-theta-star-planner \
    ros-humble-tf2-tools \
    ros-humble-twist-mux \
    ros-humble-rmw-cyclonedds-cpp tree \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY config/ config/
COPY launch/ launch/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/entrypoint.sh"]
