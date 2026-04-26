# TeamBowl Docker Build Issues

Discovered during iterative dry-run testing on 2026-04-26.
Each entry explains the symptom, root cause, and fix applied.

---

## 1. apt lists wiped before rosdep install (Isaac ROS build layer)

**Symptom:** `E: Unable to locate package ros-humble-nav2-costmap-2d` (and several other ROS packages) during `rosdep install` inside the Isaac ROS build layer.

**Root cause:** The Isaac ROS build deps layer ends with `rm -rf /var/lib/apt/lists/*` to keep the layer small. Docker layer isolation means the next `RUN` block starts with an empty apt cache. When `rosdep install` calls `apt-get install` for its dependency packages, apt can't find anything.

**Fix:** Added `apt-get update &&` at the top of the Isaac ROS build `RUN` block (Dockerfile line 117).

---

## 2. Isaac ROS rosdep keys not in standard database

**Symptom:** `ERROR: the following packages/stacks could not have their rosdep keys resolved: isaac_ros_nitros, isaac_ros_managed_nitros, isaac_ros_gxf, isaac_ros_nitros_image_type, ...`

**Root cause:** Isaac ROS uses many proprietary NITROS/GXF rosdep keys that do not exist in the standard `rosdistro` rosdep database. With `set -o pipefail`, rosdep's non-zero exit code aborts the entire build layer even when `-r` (continue on error) is passed.

**Fix:** Added `|| true` after the `rosdep install` command so the layer continues to `colcon build` regardless. System deps that rosdep _could_ install (rviz, nav2, etc.) are still installed because `apt-get update` now runs first (see issue 1).

---

## 3. Isaac ROS packages require proprietary NVIDIA binary libraries

**Symptom:** `isaac_ros_visual_slam` fails: `ament_index_get_resource() called with not existing resource ('cuvslam' 'isaac_ros_nitros')`. `nvblox_ros` fails: `Could NOT find ... isaac_ros_managed_nitros`.

**Root cause:** Both packages depend on NVIDIA's proprietary NITROS (Native ROS Interface) framework, which in turn requires:
- **cuVSLAM** — a proprietary CUDA Visual SLAM binary, not open-source
- **GXF** (Graph Execution Framework) — a proprietary NVIDIA runtime, not open-source

These cannot be built from source without NVIDIA's private binary packages.

**Fix:** Added to `--packages-ignore` in the colcon build:
- `isaac_ros_visual_slam` — needs cuVSLAM binary
- `nvblox_ros` — needs GXF/NITROS managed framework
- `nvblox_rviz_plugin` — needs `rviz_default_plugins` (from a later layer)
- `nvblox_test`, `nvblox_test_data`, `nvblox_examples_bringup` — need `ISAAC_ROS_WS` env var set at build time

**To get these at runtime:** install pre-built debs from NVIDIA's Isaac ROS apt repo (`ros-humble-isaac-ros-visual-slam`, `ros-humble-nvblox`).

---

## 4. pyvesc 0.2.2 removed from PyPI

**Symptom:** `ERROR: Could not find a version that satisfies the requirement pyvesc==0.2.2 (from versions: 1.0.1, 1.0.2, 1.0.3, 1.0.4, 1.0.5)`

**Root cause:** `pyvesc==0.2.2` was the pinned version (known-good on aarch64 Python 3.10) but has since been removed from PyPI. Available versions are now 1.0.1–1.0.5.

**Fix:** Updated pin to `pyvesc==1.0.5`.

---

## 5. pyvesc 1.0.5 does not auto-install PyCRC

**Symptom:** `import pyvesc` fails with `ModuleNotFoundError: No module named 'PyCRC'`

**Root cause:** `pyvesc 0.2.2` declared `PyCRC` in its `install_requires`, so pip installed it automatically. `pyvesc 1.0.5` dropped this automatic dependency — `PyCRC` must now be installed explicitly.

**Fix:** Added `PyCRC` to the `pip3 install` line in the Dockerfile.

---

## 6. libcusparseLt.so.0 missing — PyTorch fails to import

**Symptom:** `import torch` fails with `ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory`

**Root cause:** The PyTorch 2.5.0 JetPack 6.1 wheel requires `libcusparseLt` (CUDA Sparse Linear Algebra library) at import time. This library ships with a full JetPack SDK Manager install on the host (`/usr/lib/aarch64-linux-gnu/libcusparseLt/12/`) but is **not** present in the `nvcr.io/nvidia/l4t-jetpack:r36.4.0` base Docker image.

**Why apt install won't work:** The package (`libcusparselt0-cuda-12`) is not in any publicly-available apt repo for L4T — it is installed by the JetPack SDK Manager as a local `.deb`, not from a network repo.

**Fix:** Volume-mount the library from the Jetson host into the container (the host always has it after JetPack SDK Manager setup). `LD_LIBRARY_PATH` in the image is extended to include the mount path so the dynamic linker finds it.
- `docker-compose.yml`: volume mount `/usr/lib/aarch64-linux-gnu/libcusparseLt`
- `Dockerfile`: `ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcusparseLt/12:..."`

---

## 6b. pyvesc 1.0.5 declares wrong PyPI dependency for PyCRC

**Symptom:** `import pyvesc` fails with `ModuleNotFoundError: No module named 'PyCRC'` even after adding `PyCRC` to the Dockerfile pip install.

**Root cause:** pyvesc 1.0.5 lists `pycrc` as its PyPI dependency. `pycrc` on PyPI resolves to Thomas Pircher's CRC *code generator tool* for C/C++ — a completely different package that provides a CLI tool, not a Python `PyCRC` module. The original `PyCRC` Python library (which provided `from PyCRC.CRCCCITT import CRCCCITT`) has been removed from PyPI. pyvesc's dependency declaration is broken upstream.

**Fix:** Create a minimal `PyCRC` compatibility shim directly in the Dockerfile. Implements CRC-CCITT (XModem/FFFF/1D0F variants) inline — no external dependency needed, algorithm is standard.

---

## 7. Stale CMake cache when rebuilding workspace after host-path build

**Symptom:** `CMake Error: The current CMakeCache.txt directory /workspaces/teambowl_ws/build/... is different than the directory /home/box/TeamBowl/teambowl_ws/build/...`

**Root cause:** If colcon was previously run on the host (outside Docker), CMakeCache.txt stores the host absolute path (`/home/box/TeamBowl/teambowl_ws`). Inside the container the workspace is mounted at `/workspaces/teambowl_ws`. CMake rejects the mismatch.

**Fix:** Delete `build/`, `install/`, `log/` before the first container run. Files may be root-owned (written by a previous container run), requiring deletion via a privileged container:
```bash
docker run --rm -v /home/box/TeamBowl/teambowl_ws:/ws nvcr.io/nvidia/l4t-jetpack:r36.4.0 bash -c "rm -rf /ws/build /ws/install /ws/log"
```
