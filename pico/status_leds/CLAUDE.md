# status_leds

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
