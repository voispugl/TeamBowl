# teambowl_docker

## 2026-04-28 — Added launch_cam_debug.sh

**`launch_cam_debug.sh`** (new): Launches OAK-D camera + Isaac VSLAM + nvblox + Foxglove only inside Docker via `docker compose run --rm`. No motors, CAN, Nav2, or robot hardware. Passes through arbitrary launch args (`vslam_debug:=true`, `use_nvblox:=false`). Wraps `bringup/launch/isaac_ros_test.launch.py`.

## 2026-04-28 — Docker-first workflow; use_vslam:=true default; README overhaul

**`docker-compose.yml`**: Added `use_vslam:=true` to launch command. OAK-D W PoE camera at `192.168.11.2` uses `oak_cam_vslam.yaml` (explicit IP, 90 Hz stereo, IMU). Host `eno1` must be on `192.168.11.1/24` before container starts.

**`README.md`**: Full rewrite. Added Prerequisites section (eno1 network config + CAN systemd service). Fixed stale log path (`/tmp/colcon_build.log` → `~/TeamBowl/teambowl_ws/colcon_build.log`). Added launch arg override example. Incremental rebuild section moved here from top-level README.

**`../README.md`**: Restructured to Docker-first. "Quick Start" now uses `./launch.sh`. Native colcon/launch commands removed from main flow. CAN setup updated to use `teambowl-can.service` (not systemd-networkd). Trajectory testing and Steam Deck sections updated to use `docker exec`.

## 2026-04-23 — Added Isaac Sim desktop container (replaces Dockerfile.sim)

**New files:**
- **`Dockerfile.isaac_sim`**: x86_64 Isaac Sim 4.2 container (RTX 5080). Base `nvcr.io/nvidia/isaac-sim:4.2.0` (~22 GB). Adds ROS2 Humble, nvblox (CUDA x86_64), robot workspace software packages. `isaac_ros_visual_slam` intentionally omitted (Jetson VPI only).
- **`docker/entrypoint_isaac_sim.sh`**: Separate entrypoint. Sources Isaac ROS overlay, builds workspace skipping hardware drivers (depthai, robstride, vesc, xsens), then launches Isaac Sim with `setup_scene.py` via WebRTC.
- **`docker-compose.isaac_sim.yml`**: NVIDIA runtime, `network_mode: host`, workspace volume mount. WebRTC on port 8211, Foxglove on 8765.
- **`build.isaac_sim.sh`**: Build + start script. `--clean` wipes workspace build + Docker cache.

**Replaces:**
- `Dockerfile.sim` → `Dockerfile.isaac_sim`
- `docker-compose.sim.yml` → `docker-compose.isaac_sim.yml`
- `build.sim.sh` → `build.isaac_sim.sh`

**Key design decisions:**
- Isaac Sim 4.2 with PhysX 5, all 23 robot joints articulated (full legs)
- Pre-built test course: 7 static box/wall obstacles + static human mesh for YOLO26 testing
- Marker file: `.colcon_isaac_sim_build_complete` (separate from `.colcon_build_complete`) — both containers can share same teambowl_ws volume
- WebRTC browser UI at http://localhost:8211 (no X11 needed on host)
- Isaac Sim publishes: `/imu/data`, `/wheel/odometry`, `/visual_slam/tracking/odometry` (ground-truth VSLAM substitute), `/oak/rgb/image_raw`, `/oak/stereo/image_raw`

## 2026-04-26 — CAN on boot via systemd; build/launch split

**`teambowl-can.service`** (new): Systemd service that brings up `can0`/`can1` at 1 Mbit/s on Jetson boot. Install once:
```bash
sudo cp teambowl-can.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now teambowl-can.service
```

**`build.sh`**: No longer calls `docker compose up`. Builds the image only, then prints "Run ./launch.sh".

**`launch.sh`** (new): Calls `docker compose up`. Use this to start the container after building.

**`docker/entrypoint.sh`**: CAN setup removed — handled by the systemd service on the host at boot instead.

**`docker/entrypoint.sh`**: Added CAN network bring-up at container start (matches what `launch.sh` does manually on the host). Runs as root (no `sudo` needed), `modprobe mttcan || true` to handle already-loaded module, `ip link set canX down || true` to handle missing interfaces gracefully. CAN hardware errors on `set type` / `set up` are intentionally not suppressed.

## 2026-04-26 — Workspace colcon build: fixed stale CMake cache; entrypoint log path

**`docker/entrypoint.sh`**:
1. Colcon log now written to `${WS}/colcon_build.log` (workspace volume, host-visible) instead of `/tmp/colcon_build.log` (lost when container exits).
2. On colcon failure: `exit 1` instead of `exec bash` — `exec bash` hangs non-TTY runs indefinitely.

**Stale CMake cache**: If `build/` was previously generated on the host path (`/home/box/TeamBowl/teambowl_ws/`), CMakeCache.txt conflicts with the container path (`/workspaces/teambowl_ws/`). Fix: wipe `build/install/log` before first container run. Root-owned files need `docker run --rm -v ... bash -c "rm -rf ..."`.

## 2026-04-26 — Fixed multiple Isaac ROS build failures; updated pyvesc pin

### Isaac ROS build layer fixes (`Dockerfile` lines 85–135)

1. **`apt-get update` before rosdep install**: The deps layer ends with `rm -rf /var/lib/apt/lists/*`. Added `apt-get update &&` at the top of the build RUN so rosdep's `apt-get install` calls can find packages.

2. **rosdep non-fatal (`|| true`)**: Isaac ROS has many proprietary NITROS rosdep keys not in the standard database (`isaac_ros_nitros`, `isaac_ros_gxf`, `isaac_ros_managed_nitros`, etc.). Using `-r` alone still exits non-zero. Added `|| true` so the build continues to colcon.

3. **`libbenchmark-dev` and `libboost-thread-dev`** added to Isaac ROS build deps layer. `nvblox_ros` requires Google Benchmark; `isaac_ros_visual_slam` requires Boost thread.

4. **Packages skipped in colcon** (`--packages-ignore`): The following require NVIDIA proprietary binary packages (GXF, cuVSLAM, NITROS) that cannot be built from source without NVIDIA's private apt repo:
   - `isaac_ros_visual_slam` — needs cuVSLAM binary + `isaac_ros_nitros`
   - `nvblox_ros` — needs `isaac_ros_managed_nitros` (proprietary GXF)
   - `nvblox_rviz_plugin` — needs `rviz_default_plugins` (installed in a later layer)
   - `nvblox_test`, `nvblox_test_data`, `nvblox_examples_bringup` — test/example packages
   
   **To get VSLAM + nvblox_ros at runtime**, install from NVIDIA's Isaac ROS apt repo:
   `ros-humble-isaac-ros-visual-slam`, `ros-humble-nvblox` as pre-built debs.

### pyvesc version bump
- **`Dockerfile`**: Updated `pyvesc==0.2.2` → `pyvesc==1.0.5`. Version 0.2.2 was removed from PyPI; latest available is 1.0.5.

## 2026-04-25 — Split Isaac ROS clone/build into separate layers; fixed tee masking

**`Dockerfile`**: Split the Isaac ROS single RUN block into two:
1. **git clone layer** — clones all three repos. Cached independently so re-builds after build failures don't re-download ~500 MB of repos.
2. **build layer** — rosdep install + colcon build. Added `set -o pipefail` so colcon's exit code propagates through `| tee` instead of being masked by tee's exit 0. Without this, colcon failures were silently ignored and the layer appeared to succeed.

## 2026-04-25 — Fixed ros-humble-ros-base missing from Isaac ROS build deps layer

**`Dockerfile`**: Added `ros-humble-ros-base` to the Isaac ROS build deps layer. The Isaac ROS colcon build does `source /opt/ros/humble/setup.bash`, which doesn't exist until ROS base is installed. It was only installed in the later "Robot packages" layer — causing exit code 1 at the `source` step.

## 2026-04-25 — Fixed git/colcon/rosdep missing from Isaac ROS build deps layer

**`Dockerfile`**: Added `build-essential`, `git`, `python3-pip`, `python3-colcon-common-extensions`, `python3-rosdep` to the Isaac ROS build deps layer. These were only installed in the later "Robot packages" layer, but the Isaac ROS `git clone` + `colcon build` step runs before that — causing `command not found` (exit code 127).

## 2026-04-25 — Fixed cmake missing from OpenCV build deps layer

**`Dockerfile`**: Added `cmake` to the OpenCV build deps layer (lines 25-39). It was only installed in the later "Robot packages" layer, but the OpenCV `RUN cmake ...` step runs before that layer — causing `cmake: command not found` (exit code 127) on every build.

## 2026-04-24 — Added OpenCV 4.12.0 with CUDA + PyTorch for JetPack 6.1

**`Dockerfile`**: Added two new early layers (before Isaac ROS build, cached independently):
1. **OpenCV 4.12.0 build deps**: `cmake`, `unzip`, `pkg-config`, `libgtk-3-dev`, `libjpeg-dev`, `libpng-dev`, `libtiff-dev`, `libavcodec-dev`, `libavformat-dev`, `libswscale-dev`, `libv4l-dev`, GStreamer dev headers, `python3-dev`.
2. **OpenCV 4.12.0 from source** with `CUDA_ARCH_BIN=8.7` (Jetson AGX Orin, Ampere sm_87), `WITH_CUDNN=ON`, `WITH_GSTREAMER=ON`, `WITH_LIBV4L=ON`, `BUILD_opencv_python3=ON`, contrib modules. Installed to `/usr/local`. **~30–45 min build time.** Replaces JetPack-shipped OpenCV with a newer CUDA-accelerated 4.12.0 build.
3. `LD_LIBRARY_PATH` and `PYTHONPATH` updated for `/usr/local/lib`.

**Python deps**:
- `numpy==1.26.1` (pinned, required by PyTorch wheel)
- **PyTorch 2.5.0** from NVIDIA JetPack 6.1 redist wheel (`torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl`). Matches cuDNN/CUDA in l4t-jetpack:r36.4.0.

**Updated comment**: Robot tools layer comment updated to reflect that libopencv-dev is omitted because OpenCV 4.12.0 is now built from source (not because JetPack ships it).

## 2026-04-23 — Added Isaac ROS Visual SLAM + nvblox (Option A: baked into Docker image)

**`Dockerfile`**: Added two new layers before the robot-specific apt packages:
1. **nvblox build dependencies**: `libgoogle-glog-dev`, `libgflags-dev`, `libsqlite3-dev` + rosdep init.
2. **Isaac ROS build layer** (`/opt/isaac_ros_ws`): Clones `isaac_ros_common`, `isaac_ros_visual_slam`, `isaac_ros_nvblox` from branch `release-3.2` (JetPack 6.1 / L4T R36.4.0 / CUDA 12.6). Builds all with colcon into `/opt/isaac_ros_ws/install/`. Removes build/log dirs to save image space. **First `docker build` takes ~75 min** due to CUDA kernel compilation.
   - Layer is placed EARLY (before robot apt packages) so it is cached independently. Editing robot-specific apt deps below does NOT invalidate the Isaac ROS layer.
   - Skip keys: `libopencv-dev python3-opencv libopencv-contrib-dev` to protect JetPack's CUDA-accelerated OpenCV.
   - **ARM64/Jetson only** — do NOT add to `Dockerfile.laptop` or `Dockerfile.sim`.

**`docker/entrypoint.sh`**: Sources `/opt/isaac_ros_ws/install/setup.bash` before the colcon robot workspace build. Required so robot packages can find `isaac_ros_visual_slam` and `nvblox_ros` as ament dependencies.

**`Dockerfile`** (bashrc): Added `source /opt/isaac_ros_ws/install/setup.bash` to interactive shell sourcing chain.

**Rebuild command**: `./build.sh --clean` (full rebuild, ~75 min docker + ~8 min workspace). Subsequent `docker compose up` uses cached image.

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
