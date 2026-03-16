#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /workspaces/teambowl_ws/install/setup.bash ]; then
    source /workspaces/teambowl_ws/install/setup.bash
fi

exec "$@"
