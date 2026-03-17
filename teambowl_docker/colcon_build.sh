#!/bin/bash
# Rebuild the locomotion package and its robstride dependencies inside the
# running teambowl_dev container.
#
# Run from the host (assumes the container is already up via ./build.sh):
#   ./colcon_build.sh
#
# Packages built (in dependency order):
#   robstride_can_interfaces  — custom ROS2 service/msg definitions
#   robstride_can_driver      — motor CAN driver node
#   locomotion                — driving_leg_controller + vel_cmd_mux + collision_guard

set -e

CONTAINER="teambowl_dev"
WS="/workspaces/teambowl_ws"
PACKAGES="robstride_can_interfaces robstride_can_driver locomotion"

# Check the container is running.
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[build] ERROR: Container '${CONTAINER}' is not running."
    echo "[build] Start it first with:  cd teambowl_docker && ./build.sh"
    exit 1
fi

echo ""
echo "=============================="
echo "   COLCON BUILD (incremental)"
echo "=============================="
echo "[build] Packages: ${PACKAGES}"
echo ""

docker exec "${CONTAINER}" bash -c "
    set -e
    source /opt/ros/humble/setup.bash
    cd ${WS}
    colcon build \
        --symlink-install \
        --packages-select ${PACKAGES} \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
"

echo ""
echo "[build] Done. Restart the container (or the affected nodes) to pick up changes."
echo "[build] Quick restart:  docker restart ${CONTAINER}"
