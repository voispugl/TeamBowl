# nvblox Setup Guide

GPU-accelerated 3D terrain mapping for TeamBowl. nvblox runs entirely on the Jetson AGX Orin's CUDA cores — it builds a TSDF (Truncated Signed Distance Field) voxel map from the OAK-D depth image and exports a 2D costmap layer that Nav2 consumes directly.

**Why nvblox over Nav2's built-in VoxelLayer:**
- VoxelLayer is CPU-only; nvblox is GPU-native (order of magnitude faster on the Jetson)
- nvblox detects **upward steps AND downward drops** because it tracks freespace, not just occupied voxels
- 5 cm voxel resolution catches elevator thresholds and curbs

---

## Step 0: Check prerequisites

```bash
# JetPack version — must be 5.0 or higher
cat /etc/nv_tegra_release

# ROS distro — must be Humble
printenv ROS_DISTRO
```

---

## Step 1: Add the Isaac ROS apt repository

```bash
sudo apt install software-properties-common curl -y

# Add NVIDIA Isaac ROS repo key
curl -sSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo apt-key add -

# Add the repo
sudo sh -c 'echo "deb https://isaac.download.nvidia.com/isaac-ros/ubuntu/jammy $(lsb_release -cs) release" > /etc/apt/sources.list.d/isaac-ros.list'

sudo apt update
```

---

## Step 2: Install nvblox

```bash
sudo apt install ros-humble-nvblox ros-humble-isaac-ros-common -y
```

---

## Step 3: Verify the install

```bash
source /opt/ros/humble/setup.bash

# Should print nvblox_ros and related packages
ros2 pkg list | grep nvblox

# Quick smoke test — should start and then idle waiting for topics
ros2 run nvblox_ros nvblox_node --ros-args -p voxel_size:=0.05
# Ctrl-C to exit
```

---

## Step 4: Find the OAK-D depth topic

Launch the camera and list depth topics:

```bash
ros2 launch bringup bringup.launch.py foxglove:=false &
ros2 topic list | grep -i depth
```

You're looking for a `sensor_msgs/Image` depth topic from the OAK-D. It will be something like:
- `/oak/stereo/image_raw`
- `/oak/depth/image_raw`

Also grab the matching camera_info topic. Both are needed by nvblox.

---

## Step 5: Add nvblox to the launch file

In `teambowl_ws/src/bringup/launch/bringup.launch.py`, add this node alongside the other safety/planning nodes:

```python
Node(
    package='nvblox_ros',
    executable='nvblox_node',
    name='nvblox',
    output='screen',
    parameters=[{
        'voxel_size': 0.05,          # 5 cm — catches curbs and small steps
        'esdf_slice_height': 0.15,   # height at which the 2D costmap slice is taken
        'max_depth_m': 3.0,
        'min_depth_m': 0.15,
    }],
    remappings=[
        ('depth/image',       '/oak/stereo/image_raw'),   # UPDATE with actual topic
        ('depth/camera_info', '/oak/stereo/camera_info'), # UPDATE with actual topic
    ],
),
```

**Important:** Replace the topic names with the actual ones found in Step 4.

---

## Step 6: Add the nvblox costmap layer to Nav2

In `teambowl_ws/src/planning/config/planning.yaml`, update the local costmap section:

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      # Replace the existing plugins list with:
      plugins: ["nvblox_layer", "obstacle_layer", "inflation_layer"]

      nvblox_layer:
        plugin: "nvblox::NvbloxCostmapLayer"
        enabled: true
        nvblox_map_slice_height: 0.15
        inflation_distance_m: 0.3
```

The existing `obstacle_layer` (2D LaserScan) stays as a fallback — nvblox adds 3D on top.

---

## Step 7: Test

1. Launch the full stack
2. Open Foxglove Studio → connect to `ws://robot-ip:8765`
3. Add a **Costmap** panel, select `/local_costmap/costmap`
4. Place a 10 cm tall box on the floor in front of the robot
5. The box should appear as a red (obstacle) cell in the costmap
6. Place the object right at floor level (like a door threshold) — should also show up at 5 cm resolution

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `ros2 pkg list \| grep nvblox` returns nothing | Install failed — re-run Step 2 |
| nvblox node crashes at startup | JetPack version too old (< 5.0) |
| No costmap layer appears | Wrong depth topic name — re-check Step 4 |
| GPU OOM errors in logs | Reduce `voxel_size` to 0.1 or reduce `max_depth_m` to 2.0 |
| Obstacles flicker | Reduce robot speed or decrease `esdf_slice_height` |

---

## GPU memory estimate

nvblox on a 3m × 3m × 1m map at 5 cm voxel resolution uses approximately **400–700 MB** of GPU memory. The Jetson AGX Orin has 32 GB unified CPU/GPU memory, so headroom is not a concern.
