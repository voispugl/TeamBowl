FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Basic tools
RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    nano \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete \
    python3-serial \
    python3-can \
    python3-yaml \
    usbutils \
    iputils-ping \
    net-tools \
    can-utils \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Helpful ROS packages commonly useful in your stack
RUN apt-get update && apt-get install -y \
    ros-humble-twist-mux \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-camera-info-manager \
    ros-humble-xacro \
    ros-humble-rviz2 \
    ros-humble-foxglove-bridge \
    ros-humble-vision-msgs \
    ros-humble-nmea-msgs \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Python deps that are commonly useful for your project
RUN pip3 install --no-cache-dir \
    numpy \
    pyvesc \
    pyCRC

# rosdep init may already be done in some bases; don't fail build if so
RUN rosdep init || true
RUN rosdep update || true

# Create workspace mount point
RUN mkdir -p /workspaces/teambowl_ws/src

# Auto-source ROS in interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "if [ -f /workspaces/teambowl_ws/install/setup.bash ]; then source /workspaces/teambowl_ws/install/setup.bash; fi" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspaces/teambowl_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
