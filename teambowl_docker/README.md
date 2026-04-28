# TeamBowl Docker

ROS2 Humble container for the TeamBowl robot. On first run it automatically builds the workspace, then launches the full system via `bringup.launch.py use_vslam:=true`.

## Prerequisites (host setup — do once)

These must be in place on the Jetson **before** `docker compose up`.

### 1. OAK-D W PoE camera — host ethernet

Set `eno1` to `192.168.11.1/24` so the container can reach the camera at `192.168.11.2`:

```bash
sudo nmcli connection modify "Wired connection 1" \
    ipv4.method manual \
    ipv4.addresses "192.168.11.1/24" \
    ipv4.gateway "" ipv4.dns ""
sudo nmcli connection down "Wired connection 1" && sudo nmcli connection up "Wired connection 1"
ping -c 1 192.168.11.2   # verify camera reachable
```

### 2. CAN interfaces — persistent via systemd

```bash
sudo cp teambowl-can.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now teambowl-can.service
# Verify:
ip link show can0   # should show state UP
ip link show can1   # should show state UP
```

---

## Build the image

Only needed once (or after changing the Dockerfile):

```bash
cd ~/TeamBowl/teambowl_docker
./build.sh
```

Use `./build.sh --clean` for a full rebuild (clears colcon cache + Docker cache, ~75 min).

---

## Run the robot

```bash
cd ~/TeamBowl/teambowl_docker
./launch.sh
```

- **First run:** builds the ROS2 workspace (~10–15 min), then launches all nodes automatically.
- **Subsequent runs:** skips the build and launches immediately.

Default launch: `bringup.launch.py use_vslam:=true` (OAK-D PoE W, Isaac VSLAM, nvblox).

To launch with different args (e.g. no VSLAM):

```bash
docker compose run --rm teambowl ros2 launch bringup bringup.launch.py use_vslam:=false
```

---

## Stop the robot

```bash
docker compose down
```

Or press `Ctrl+C` if running in the foreground.

---

## Get a debug shell (instead of auto-launching)

```bash
docker compose run --rm teambowl bash
```

## Attach a shell to a running container

```bash
docker exec -it teambowl_dev bash
```

---

## Force a workspace rebuild

Delete the build marker, then restart:

```bash
rm ~/TeamBowl/teambowl_ws/install/.colcon_build_complete
./launch.sh
```

---

## View build logs

Build logs are written to the workspace volume (visible from the host):

```bash
cat ~/TeamBowl/teambowl_ws/colcon_build.log
```

---

## Camera-only debug launch

Launches just the OAK-D camera, Isaac VSLAM, nvblox, and Foxglove — no motors, CAN, Nav2, or robot hardware:

```bash
./launch_cam_debug.sh
# with optional args:
./launch_cam_debug.sh vslam_debug:=true
./launch_cam_debug.sh use_nvblox:=false
```

Connect Foxglove to `ws://ROBOT_IP:8765`. Check `/oak/rgb/image_raw` (~15 Hz), `/oak/left/image_rect` (~90 Hz), `/oak/imu/data` (~200 Hz), `/visual_slam/tracking/odometry`.

---

## Incremental rebuild (after editing robot packages)

Rebuilds only `robstride_can_interfaces`, `robstride_can_driver`, and `locomotion` inside the running container:

```bash
./colcon_build.sh
docker compose restart
```
