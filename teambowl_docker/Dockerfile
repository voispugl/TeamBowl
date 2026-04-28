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

# ── OpenCV 4.12.0 with CUDA: build dependencies ──────────────────────────────
# Placed before the OpenCV build layer so it caches independently.
RUN apt-get update && apt-get install -y \
    cmake \
    unzip \
    pkg-config \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── OpenCV 4.12.0 with CUDA ───────────────────────────────────────────────────
# Builds from source to get CUDA 12.6 + CUDNN + contrib modules on the Orin.
# CUDA_ARCH_BIN=8.7 targets Jetson AGX Orin (Ampere sm_87).
# Replaces the JetPack-shipped OpenCV with a newer version built against our
# exact CUDA/cuDNN. Installed to /usr/local so cmake find_package picks it up.
# WARNING: ~30-45 min build time. Placed early so this layer is cached
# independently of robot-specific package changes below.
RUN mkdir -p /tmp/opencv_build && cd /tmp/opencv_build && \
    curl -L https://github.com/opencv/opencv/archive/4.12.0.zip -o opencv-4.12.0.zip && \
    curl -L https://github.com/opencv/opencv_contrib/archive/4.12.0.zip -o opencv_contrib-4.12.0.zip && \
    unzip opencv-4.12.0.zip && \
    unzip opencv_contrib-4.12.0.zip && \
    rm opencv-4.12.0.zip opencv_contrib-4.12.0.zip && \
    cd opencv-4.12.0 && \
    mkdir release && \
    cd release && \
    cmake -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D CUDA_ARCH_BIN="8.7" \
          -D CUDA_ARCH_PTX="" \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules \
          -D WITH_GSTREAMER=ON \
          -D WITH_LIBV4L=ON \
          -D BUILD_opencv_python3=ON \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/opencv_build

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
# libcusparseLt: installed by JetPack SDK Manager on host, volume-mounted into
# the container via docker-compose.yml. Not available in l4t-jetpack base image.
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcusparseLt/12:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages/:${PYTHONPATH}"

# ── Isaac ROS: build dependencies ────────────────────────────────────────────
# Installed as an early layer so the ~75 min Isaac ROS build below is cached
# independently of robot-specific packages further down.
# libgoogle-glog-dev, libgflags-dev, libsqlite3-dev: required by nvblox C++ library.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libboost-dev \
    libboost-thread-dev \
    libbenchmark-dev \
    ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init || true \
    && rosdep update || true

# ── Isaac ROS: git clone (separate layer for caching) ────────────────────────
# Split from the build step so git clones are cached once and re-used on build
# failures without re-downloading ~500 MB of repos.
RUN mkdir -p /opt/isaac_ros_ws/src && \
    cd /opt/isaac_ros_ws/src && \
    git clone --recurse-submodules -b release-3.2 \
        https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git && \
    git clone --recurse-submodules -b release-3.2 \
        https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git && \
    git clone --recurse-submodules -b release-3.2 \
        https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox.git

# ── Isaac ROS: build ──────────────────────────────────────────────────────────
# Packages: isaac_ros_common, isaac_ros_visual_slam (VIO), isaac_ros_nvblox (TSDF costmap)
# Branch: release-3.2 — validated for JetPack 6.1 / L4T R36.4.0 / CUDA 12.6.
# WARNING: First docker build takes ~75 min (CUDA kernel compilation).
# Skip keys: libopencv-dev and variants to protect JetPack's CUDA-accelerated OpenCV.
# set -o pipefail: ensures colcon failure propagates through the tee pipe.
RUN set -o pipefail && \
    apt-get update && \
    cd /opt/isaac_ros_ws && \
    source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys "libopencv-dev python3-opencv libopencv-contrib-dev \
                     isaac_ros_peoplenet_models_install isaac_ros_peoplesemseg_models_install \
                     isaac_ros_nitros_image_type isaac_ros_nitros_camera_info_type \
                     isaac_ros_nitros isaac_ros_gxf ament_python" || true && \
    colcon build \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
        --packages-ignore nvblox_test nvblox_test_data nvblox_examples_bringup \
            isaac_ros_visual_slam nvblox_rviz_plugin nvblox_ros \
        --event-handlers console_direct+ && \
    rm -rf /opt/isaac_ros_ws/build /opt/isaac_ros_ws/log

# ── Robot packages ────────────────────────────────────────────────────────────
# Basic tools + USB/CAN/serial utilities + ROS2 base
# NOTE: libopencv-dev and python3-opencv are intentionally omitted — OpenCV 4.12.0 is
# built from source above with CUDA 12.6 support. Installing the apt version would
# overwrite it with a CPU-only build.
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
    ros-humble-diagnostic-aggregator \
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

# Nav2 planning stack + EKF + pointcloud converter
# Added 2026-04-16: required by bringup (state_estimation EKF, nav2 planner/controller,
# /oak/points → /oak/nav_scan conversion for nav2 local costmap obstacle layer).
RUN apt-get update && apt-get install -y \
    ros-humble-robot-localization \
    ros-humble-navigation2 \
    ros-humble-pointcloud-to-laserscan \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# numpy==1.26.1 pinned: required by the NVIDIA PyTorch wheel below.
# pyvesc==1.0.5 pinned: 0.2.2 was removed from PyPI; 1.0.5 is latest available.
# PyCRC listed explicitly: pyvesc 1.0.5 does not auto-install it unlike 0.2.2.
# depthai (Python SDK) is intentionally omitted: no Python code imports it, and the pip
# wheel bundles libdepthai-core.so which conflicts with ros-humble-depthai at runtime.
# python-can: required by robstride_can_driver (CAN bus communication).
RUN pip3 install --no-cache-dir \
    numpy==1.26.1 \
    pyvesc==1.0.5 \
    python-can \
    websockets

# PyCRC compatibility shim for pyvesc 1.0.5.
# pyvesc imports `from PyCRC.CRCCCITT import CRCCCITT` but the original PyCRC
# PyPI package has been removed. pyvesc's declared dep `pycrc` resolves to a
# completely unrelated C/C++ CRC code-generator tool. Shim implements CRC-CCITT
# (XModem / FFFF / 1D0F variants) directly — no external dependency needed.
RUN python3 - << 'EOF'
import os
d = '/usr/local/lib/python3.10/dist-packages/PyCRC'
os.makedirs(d, exist_ok=True)
open(f'{d}/__init__.py', 'w').close()
with open(f'{d}/CRCCCITT.py', 'w') as f:
    f.write("""class CRCCCITT:
    _INIT = {'FFFF': 0xFFFF, '1D0F': 0x1D0F}
    def __init__(self, version='XModem'):
        self._crc = self._INIT.get(version, 0x0000)
    def calculate(self, data=''):
        crc = self._crc
        for b in (data.encode() if isinstance(data, str) else data):
            crc ^= b << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1) & 0xFFFF
        return crc
""")
EOF

# PyTorch for JetPack 6.1 (aarch64, Python 3.10, CUDA 12.6)
# Wheel from NVIDIA redist — matches the cuDNN/CUDA versions in l4t-jetpack:r36.4.0.
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache \
        https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# rosdep init may already be done in the base image
RUN rosdep init || true
RUN rosdep update || true

# Create workspace mount point
RUN mkdir -p /workspaces/teambowl_ws/src

# Auto-source ROS + Isaac ROS overlay + robot workspace in interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /opt/isaac_ros_ws/install/setup.bash ]; then source /opt/isaac_ros_ws/install/setup.bash; fi" >> /root/.bashrc && \
    echo "if [ -f /workspaces/teambowl_ws/install/setup.bash ]; then source /workspaces/teambowl_ws/install/setup.bash; fi" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspaces/teambowl_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
