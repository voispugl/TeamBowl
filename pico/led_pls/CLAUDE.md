# led_pls

MicroPython NeoPixel driver for Pico 2 (RP2350). Controls 30 WS2812 LEDs on GPIO28 via PIO state machine.

## Recent Changes

### Pin output debug script (2026-04-14)
- Blinks built-in LED 5× at boot to confirm script is running
- Pulses GPIO28 HIGH as plain output (check ~3.3V with multimeter)
- Moves PIO sideset to GPIO2 so DIN wire can be tested on a known good pin

### Stripped to minimal blue test (2026-04-14)
- Removed all button/loop logic to isolate hardware — just lights 6 LEDs blue at boot
- `NUM_PIXELS = 6` to match current test strip size

### Bug fixes + built-in LED pulse (2026-04-14)
- **Added `time.sleep_us(50)` latch pulse** after pixel loop in `set_pixels` — WS2812 requires >50µs LOW to commit data.
- **Added `set_pixels(*colors[0])` at startup** — strip was dark until first button press.
- **Changed `break` → `continue`** in exception handler — loop no longer exits if `rp2.bootsel_button()` is unavailable.
- **Built-in LED pulse** — `machine.Pin("LED")` turns on during NeoPixel transmission and off after latch, giving a visual heartbeat.
- **`sm.put(color, 8)`** instead of `sm.put(color << 8)` — uses MicroPython's native shift argument, same behavior but cleaner.
