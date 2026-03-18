FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Locale setup — required before adding the ROS2 apt repo
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

# Add ROS2 Humble apt repository (not present in l4t-jetpack base)
RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
         http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
         | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    rm -rf /var/lib/apt/lists/*

# Basic tools + USB/CAN/serial utilities + ROS2 base
# NOTE: libopencv-dev and python3-opencv are intentionally omitted — l4t-jetpack ships
# OpenCV built with CUDA 12; installing the apt version would overwrite it with CPU-only.
RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    cmake \
    git \
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
    libusb-1.0-0-dev \
    libopenblas-dev \
    awscli \
    ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/*

# Core ROS2 Humble packages
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
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-image-pipeline \
    ros-humble-diagnostic-updater \
    ros-humble-diagnostic-msgs \
    ros-humble-stereo-msgs \
    ros-humble-rclcpp-components \
    ros-humble-composition-interfaces \
    ros-humble-message-filters \
    ros-humble-ffmpeg-image-transport-msgs \
    ros-humble-mavros-msgs \
    ros-humble-visualization-msgs \
    ros-humble-foxglove-msgs \
    ros-humble-imu-tools \
    ros-humble-depth-image-proc \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# DepthAI C++ SDK — provides depthaiConfig.cmake so that find_package(depthai CONFIG REQUIRED)
# succeeds when building depthai_bridge, depthai_ros_driver, and depthai_examples from source.
# All other depthai-ros packages are built from source by colcon.
RUN apt-get update && apt-get install -y \
    ros-humble-depthai \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# pyvesc==0.2.2 pinned: known-good on aarch64 Python 3.10; PyCRC is auto-installed as
# pyvesc's own dependency at the correct version — do NOT add PyCRC separately.
# depthai (Python SDK) is intentionally omitted: no Python code imports it, and the pip
# wheel bundles libdepthai-core.so which conflicts with ros-humble-depthai at runtime.
RUN pip3 install --no-cache-dir \
    numpy \
    pyvesc==0.2.2

# rosdep init may already be done in the base image
RUN rosdep init || true
RUN rosdep update || true

# Create workspace mount point
RUN mkdir -p /workspaces/teambowl_ws/src

# Auto-source ROS in interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /workspaces/teambowl_ws/install/setup.bash ]; then source /workspaces/teambowl_ws/install/setup.bash; fi" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspaces/teambowl_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
