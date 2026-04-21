#!/usr/bin/env bash
# build_neopixel.sh — Compile the WS2812B GPIO cdev shared library.
# Usage: sudo bash build_neopixel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/neopixel_cdev_write.c"
OUT="${SCRIPT_DIR}/libneopixel_cdev.so"

echo "Building ${OUT} ..."

gcc \
    -O2 \
    -march=armv8.2-a \
    -std=c11 \
    -Wall -Wextra \
    -fPIC \
    -shared \
    -o "${OUT}" \
    "${SRC}" \
    -lrt \
    -Wl,-soname,libneopixel_cdev.so

echo "Done: ${OUT}"
echo ""
echo "Run with:  sudo chrt -f 99 python3 ${SCRIPT_DIR}/debug_led.py"
