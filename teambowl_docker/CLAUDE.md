# teambowl_docker

## File inventory

| File | Purpose |
|------|---------|
| `Dockerfile` | **Robot image** — AGX Orin (aarch64, JetPack 6.1, CUDA 12.6) |
| `Dockerfile.laptop` | **Laptop image** — x86_64 (Ubuntu 22.04, no GPU, skips depthai) |
| `docker-compose.yml` | Robot compose — privileged, `/dev` mount, CAN/USB |
| `docker-compose.laptop.yml` | Laptop compose — X11 forwarding, no hardware access |
| `build.sh` | Build + start robot image (`--clean` for full rebuild) |
| `build.laptop.sh` | Build + start laptop image (`--clean` for full rebuild) |
| `docker/entrypoint.sh` | Shared entrypoint — auto-builds workspace on first run |

## Quick start

**Robot (AGX Orin):**
```bash
cd ~/TeamBowl/teambowl_docker
./build.sh                 # build image + start (first run ~20 min)
docker compose up          # subsequent starts
```

**Laptop (x86_64):**
```bash
cd ~/TeamBowl/teambowl_docker
xhost +local:docker        # allow X11 forwarding (macOS: XQuartz must be running)
./build.laptop.sh          # build image + drop to bash (first run ~10 min)
# Inside container:
ros2 launch bringup bringup.launch.py foxglove:=true
```

Override workspace path: `TEAMBOWL_WS=/my/path ./build.laptop.sh`

## Key design decisions

- **Robot base**: `nvcr.io/nvidia/l4t-jetpack:r36.4.0` — JetPack ships CUDA-accelerated
  OpenCV; apt libopencv-dev would overwrite it. `ros-humble-depthai` (C++ SDK) is from
  the standard ROS apt mirror (ARM64 packages available there).
- **Laptop base**: `ros:humble-ros-base-jammy` — ROS2 already set up; uses standard
  apt `libopencv-dev`. Skips all depthai-ros packages (`SKIP_DEPTHAI=1` env var in
  entrypoint) because `ros-humble-depthai` is not in the x86_64 ROS apt mirror and
  the OAK-D is never attached to a laptop.
- **`SKIP_DEPTHAI=1`**: Set in `Dockerfile.laptop`, read by `entrypoint.sh` to add all
  5 depthai source packages to `--packages-ignore` at build time.
- **Workspace mount**: Source code is never copied into the image — always volume-mounted
  at `/workspaces/teambowl_ws`. Colcon build runs inside the container on first start.
- **Marker file**: `install/.colcon_build_complete` prevents rebuilding on every start.
  Delete it to force a rebuild: `rm teambowl_ws/install/.colcon_build_complete`

## 2026-04-16 — Added sim image for MuJoCo simulation on Ubuntu 24.04 VM

- **`Dockerfile.sim`**: New x86_64 sim image. `FROM teambowl:laptop`; adds `mujoco` pip
  package (CPU-only forward simulation, no GPU). Creates `/workspaces/teambowl_mjlab/`
  mount point for the MJCF model files.
- **`docker-compose.sim.yml`**: Sim compose. Extra volume mounts:
  - `${TEAMBOWL_WS}` → `/workspaces/teambowl_ws` (workspace source)
  - `${TEAMBOWL_ROOT}/mjlab_robot` → `/workspaces/teambowl_mjlab` (MJCF + meshes)
  - The meshes symlink (`mjlab_robot/meshes/ → ../teambowl_ws/sim/mujoco/meshes`) resolves
    correctly because both directories are mounted at sibling paths.
  - Auto-launches: `ros2 launch bringup sim.launch.py`
- **`build.sim.sh`**: New build script. Builds `teambowl:laptop` first (dep), then
  `teambowl:sim`. Accepts `--clean`. Respects `TEAMBOWL_ROOT` and `TEAMBOWL_WS` env vars.

## 2026-04-17 — Added websockets pip dependency

- **`Dockerfile`** and **`Dockerfile.laptop`**: Added `websockets` to pip install.
  Required by `steamdeck_teleop` package (`steamdeck_ws_teleop` node runs a WebSocket
  server on port 8888 for Steam Deck browser-based gamepad teleop).

## 2026-04-16 — Added laptop image + missing nav2 packages

- **`Dockerfile.laptop`**: New x86_64 laptop image. Base `ros:humble-ros-base-jammy`;
  adds `libopencv-dev`; sets `SKIP_DEPTHAI=1` to skip OAK-D packages at build time.
- **`Dockerfile`** (robot): Added `ros-humble-robot-localization`,
  `ros-humble-navigation2`, `ros-humble-pointcloud-to-laserscan`, and `python-can`
  pip package. These were missing and caused runtime failures in the nav2/EKF stack.
- **`docker/entrypoint.sh`**: Added `SKIP_DEPTHAI=1` env var check. When set, all 5
  depthai source packages are added to `--packages-ignore` in colcon build.
- **`docker-compose.yml`**: Parameterized source path with
  `${TEAMBOWL_WS:-/home/box/TeamBowl/teambowl_ws}` so any user can override it.
- **`docker-compose.laptop.yml`**: New laptop compose. No privileged/`/dev`; adds X11
  socket mount + `DISPLAY` env var for rviz2/GUI; defaults command to `bash`.
- **`build.laptop.sh`**: New build script matching `build.sh` pattern. Accepts
  `--clean` and respects `TEAMBOWL_WS` env var for workspace path.

## 2026-03-16 — Added colcon_build.sh and teleop.sh

- **`colcon_build.sh`**: Incrementally rebuilds `robstride_can_interfaces`,
  `robstride_can_driver`, and `locomotion` inside the running `teambowl_dev`
  container. Faster than a full rebuild; run after editing any of those packages.
- **`teleop.sh`**: Brings up CAN interfaces, launches the robstride motor driver
  in the background, sets robot mode to "teleop", and opens `teleop_twist_keyboard`
  publishing to `/cmd_vel_teleop`. Resets mode to "off" on Ctrl+C.
  Assumes `teambowl_dev` is already running (`./build.sh`).

---

## 2026-03-18 — Fixed pyvesc/PyCRC version conflict and removed depthai pip install

### What changed
- **`Dockerfile`**: Pinned `pyvesc==0.2.2` and removed the explicit `PyCRC` line.
  - `pyvesc` declares `PyCRC` in its own `install_requires` — having `PyCRC` listed
    separately let pip resolve an incompatible version pair, crashing `cmd_vel_to_vesc`
    at startup with an ImportError. Letting pyvesc install `PyCRC` itself at the
    version it requires fixes this.
  - Pinning `pyvesc==0.2.2` ensures a known-good version on aarch64 / Python 3.10.
- **`Dockerfile`**: Removed `depthai` from the pip install entirely.
  - `depthai` (pip) is the Luxonis **Python** SDK and bundles its own `libdepthai-core.so`.
    No Python file in the workspace imports `depthai`; all camera I/O goes through the
    C++ `depthai_ros_driver`. The pip wheel can fail on JetPack 6.1 aarch64, and even
    when it succeeds the bundled library conflicts with `ros-humble-depthai` at runtime.
  - Requires `./build.sh --clean` to take effect.

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
