#!/bin/bash
# Reset CAN buses (can0 + can1) to 1 Mbit/s. Run from inside the Docker container.
set -e

for BUS in can0 can1; do
    echo "[can] Resetting $BUS..."
    ip link set $BUS down        2>/dev/null || true
    ip link set $BUS type can bitrate 1000000
    ip link set $BUS up
    echo "[can] $BUS up @ 1 Mbit/s"
done
