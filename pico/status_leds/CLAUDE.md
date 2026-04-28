# status_leds

## 2026-04-21 — First build and flash on this machine

- Installed ARM GCC toolchain: `sudo apt install gcc-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib`
- Cloned Pico SDK 2.2.0 to `~/pico-sdk` (with submodules)
- Built firmware: `mkdir build && cd build && cmake .. -DPICO_BOARD=pico2 -DPICO_SDK_PATH=$HOME/pico-sdk && make -j$(nproc)`
- Flashed via BOOTSEL mode: mounted at `/dev/sda1`, copied `build/status_leds.uf2`
- Updated `src/safety/config/safety.yaml` serial_port to `usb-Raspberry_Pi_Pico_1B8494CFA7EDCDA1-if00`
- All LED states verified working over USB-serial: red, green, yellow, purple, orange wave, purple blink

## 2026-04-17 — USB CDC serial + kill switch + blink mode

- **`status_leds.c`**:
  - Added `MODE_BLINK` enum value — all pixels toggle on/off at ~3 Hz (17 × 20 ms ticks per half-cycle)
  - Added `CMD_BLINK 0x40` — 4-byte command `0x40 R G B`, starts blink mode with given color
  - Added kill switch on **GP15** (active-low, internal pull-up); sends `K1\n` on press, `K0\n` on release over USB CDC
  - Added USB-serial command reception via `getchar_timeout_us(0)` — mirrors I2C protocol
  - Refactored command parsing into shared `apply_command()` helper used by both I2C and USB-serial paths
  - `stdio_init_all()` already activated USB CDC (CMakeLists had `stdio_usb 1` set previously)
- **`README.md`**: Added kill switch wiring section, serial port discovery guide, ROS2 LED state table, and `CMD_BLINK` documentation



C firmware for Pico 2 (RP2350). Drives 30 WS2812 LEDs on GPIO28 via PIO. Built with Pico SDK 2.2.0.

## Recent Changes

### I2C slave + wave/sequential animations (2026-04-14)
- Added I2C slave mode on GP4 (SDA) / GP5 (SCL) at address 0x2A, 100 kHz
- Added `led_mode_t` enum: STATIC, WAVE_RIGHT, WAVE_LEFT, SEQ_RIGHT, SEQ_LEFT
- **Wave mode**: moving dot with 6-pixel fade tail (head → 100%/70%/50%/30%/20%/10%)
- **Sequential fill (Corvette)**: LEDs fill one-by-one in a direction, hold, reset, loop
- Added `pixel_color(r, g, b, scale)` helper for per-pixel brightness in animations
- Colors stored as raw RGB internally so scale can be applied at render time
- I2C commands: 0x00–0x03 (predefined color), 0x10+RGB (custom), 0x20/0x21 (wave), 0x22 (stop), 0x30/0x31 (sequential fill)
- BOOTSEL cycles through all 8 steps: Red→Green→Yellow→Purple→Wave R→Wave L→Seq R→Seq L (orange for animations)
- Added `led_step_t` struct and `steps[]` array; button increments `step_idx` and applies mode+color atomically
- Built-in green LED blinks at 2 Hz always via `add_repeating_timer_ms` hardware timer (independent of animation loop)
- Added README.md with wiring diagram, build instructions, and I2C protocol reference
