# teambowl_docker

## 2026-03-16 — Added colcon_build.sh and teleop.sh

- **`colcon_build.sh`**: Incrementally rebuilds `robstride_can_interfaces`,
  `robstride_can_driver`, and `locomotion` inside the running `teambowl_dev`
  container. Faster than a full rebuild; run after editing any of those packages.
- **`teleop.sh`**: Brings up CAN interfaces, launches the robstride motor driver
  in the background, sets robot mode to "teleop", and opens `teleop_twist_keyboard`
  publishing to `/cmd_vel_teleop`. Resets mode to "off" on Ctrl+C.
  Assumes `teambowl_dev` is already running (`./build.sh`).

---

## 2026-03-17 — Fixed PyCRC pip install casing

### What changed
- **`Dockerfile`**: Changed `pyCRC` → `PyCRC` in the pip install. `pyvesc` imports `from PyCRC.CRCCCITT import CRCCCITT` at runtime. The lowercase `pyCRC` either resolves to a different PyPI package or installs to a different module path on Linux (case-sensitive filesystem). Requires `./build.sh --clean` to take effect.

---

## 2026-03-16 — Fixed depthai_filters build failure + updated build.sh

### What changed
- **`docker/entrypoint.sh`**: Added `--packages-ignore depthai_filters depthai_examples` to the colcon build.
  - `depthai_filters` requires `opencv2/ximgproc/disparity_filter.hpp` (opencv_contrib) — NOT present in Jetson's CUDA OpenCV. Installing `libopencv-contrib-dev` from apt would overwrite the CUDA build with a CPU-only version.
  - `depthai_examples` is only demo code, not used by bringup.
  - Neither package is referenced by `bringup.launch.py`.
- **`src/drivers/depthai-ros/depthai_ros_driver/package.xml`**: Removed `<depend>depthai_examples</depend>`. The driver has no actual dependency on the examples package — this upstream mistake caused `depthai_ros_driver` to be blocked whenever `depthai_examples` failed.
- **`teambowl_docker/build.sh`**: Updated with cleaner output and usage text.
  - `./build.sh` — rebuild Docker image (cached) then `docker compose up`
  - `./build.sh --clean` / `-c` — stop containers, wipe colcon `build/install/log`, rebuild Docker `--no-cache`, then start
  - `./build.sh --help` / `-h` — print usage

---

## 2026-03-16 — Full colcon build of all workspace packages

### What changed
- **Dockerfile**: Added all remaining compile-time deps found by auditing every CMakeLists.txt:
  - `libeigen3-dev` (xsens_mti_ros2_driver needs Eigen3 3.3)
  - `ros-humble-mavros-msgs` (xsens_mti_ros2_driver)
  - `ros-humble-visualization-msgs` (depthai_filters)
  - `ros-humble-foxglove-msgs` (depthai_examples)
  - `ros-humble-imu-tools` (provides `rviz_imu_plugin` for depthai_examples)
  - `ros-humble-depth-image-proc` (depthai_examples exec dep)
  - Removed all `ros-humble-depthai-*` prebuilt packages except `ros-humble-depthai` (the C++ SDK cmake config needed for `find_package(depthai CONFIG REQUIRED)`) — all other depthai-ros packages now built from source.
- **docker/entrypoint.sh**: Removed `--packages-ignore` — colcon now builds ALL workspace packages.

### Important context
- Colcon resolves build order automatically from package.xml dependencies.
- The xsens driver's proprietary `xspublic` C++ libs are compiled automatically by its own CMakeLists during the colcon build.
- If `ros-humble-depthai` doesn't provide the cmake config on aarch64, the fallback is to add the Luxonis apt repo (`packages.luxonis.com`).
- First `docker compose up` will take ~15–20 min due to full C++ compilation.

---

## 2026-03-16 — Rebased on Jetson AGX Orin l4t-jetpack

### What changed
- **Dockerfile**: Changed base image from `ros:humble-ros-base-jammy` to `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (JetPack 6.1 — CUDA 12.6, cuDNN 9.3, TensorRT 10.3).
- Added locale setup layer (required on l4t before adding ROS2 apt repo).
- Added explicit ROS2 Humble apt repo setup (not pre-installed in l4t-jetpack).
- Added `ros-humble-ros-base`, `libopenblas-dev`, `awscli` to tools layer.
- Removed `libopencv-dev` and `python3-opencv` — l4t-jetpack ships CUDA-accelerated OpenCV; apt version would overwrite it with a CPU-only build.

### Important context
- Jetson's cv2 Python bindings are already in the base image — do NOT reinstall via apt or pip.
- `entrypoint.sh` and `docker-compose.yml` were not changed.

---

## 2026-03-16 — Fixed Dockerfile bugs and added auto-launch

### What changed
- **Dockerfile**: Added all missing apt dependencies required by the workspace packages:
  - `libusb-1.0-0-dev`, `python3-opencv` (system)
  - `ros-humble-tf2*`, `ros-humble-image-transport*`, `ros-humble-image-pipeline`, `ros-humble-diagnostic-updater`, `ros-humble-stereo-msgs`, `ros-humble-rclcpp-components`, `ros-humble-composition-interfaces`, `ros-humble-message-filters`, `ros-humble-ffmpeg-image-transport-msgs` (ROS)
  - Full `ros-humble-depthai*` stack (pre-built OAK camera driver + C++ SDK — avoids needing to build the Luxonis SDK from source)
  - Added `depthai` to pip installs
- **docker/entrypoint.sh**: Added auto-build logic. On first container run, `colcon build` is executed with `--packages-ignore` for all depthai-ros source packages (already installed as apt). A marker file (`install/.colcon_build_complete`) on the volume prevents rebuilding on subsequent runs. Falls back to `bash` on build failure.
- **docker-compose.yml**: Changed `command` from `bash` to `ros2 launch bringup bringup.launch.py` so the full robot stack starts automatically.

### Important context
- The workspace source is volume-mounted, so builds CANNOT happen during `docker build` — they must happen at container startup (entrypoint).
- To force a rebuild: `docker exec teambowl_dev rm /workspaces/teambowl_ws/install/.colcon_build_complete` then restart.
- To get a debug shell instead of auto-launch: `docker compose run --rm teambowl bash`
