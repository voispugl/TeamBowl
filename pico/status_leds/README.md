# status_leds

WS2812 NeoPixel driver for the **Pico 2 (RP2350)**. Drives 30 RGB LEDs on GPIO 28 via the PIO state machine. Supports on-board BOOTSEL button cycling, and full remote control over I2C.

---

## Hardware wiring

### NeoPixel strip

```
Pico 2                  74AHCT125             LED strip
─────────               ─────────             ─────────
GPIO 28 ──────────────► A (input)
                        Y (output) ──────────► DIN
VBUS (pin 40, 5V) ────► VCC
                        GND ──────────────────► GND
                        /OE ──► GND  (always enabled)

VBUS (pin 40, 5V) ──────────────────────────► VCC
GND ────────────────────────────────────────► GND
```

> **Why a level shifter?** The Pico's GPIO outputs 3.3 V; WS2812B strips running at 5 V require a data HIGH of ≥ 3.5 V. The 74AHCT125 translates 3.3 V → 5 V logic.

### I2C (slave)

| Signal | Pico 2 pin | GPIO |
|--------|-----------|------|
| SDA    | Pin 6     | GP4  |
| SCL    | Pin 7     | GP5  |

Add **4.7 kΩ pull-up resistors** from SDA and SCL to 3.3 V (pin 36). Most Raspberry Pi I2C masters already have internal pull-ups, but external ones are recommended for reliability.

---

## Building

Requires the [Raspberry Pi Pico SDK](https://github.com/raspberrypi/pico-sdk) (v2.2.0+) and the ARM embedded GCC toolchain.

```bash
mkdir build && cd build
cmake .. -DPICO_BOARD=pico2
make -j$(nproc)
```

The build produces `build/status_leds.uf2`.

---

## Flashing

1. Hold the **BOOTSEL** button on the Pico 2 and plug in USB — it mounts as a USB drive.
2. Drag `build/status_leds.uf2` onto the drive.
3. The Pico reboots automatically and the strip lights red.

---

## BOOTSEL button

Each press cycles through the four predefined static colors:

| Index | Color  |
|-------|--------|
| 0     | Red    |
| 1     | Green  |
| 2     | Yellow |
| 3     | Purple |

Pressing the button while an animation is running switches back to static mode with the next color.

---

## I2C command reference

**Slave address:** `0x2A`  
**Bus speed:** 100 kHz

### Predefined color — 1 byte

Send a single byte `0x00`–`0x03` to select a color and enter static mode.

| Byte | Color  |
|------|--------|
| `0x00` | Red    |
| `0x01` | Green  |
| `0x02` | Yellow |
| `0x03` | Purple |

### Custom RGB color — 4 bytes

```
0x10  R  G  B
```

Sets a custom static color. R, G, B are 0–255.

### Animation commands — 1 byte

| Byte   | Effect |
|--------|--------|
| `0x20` | **Wave right** — single lit pixel with 6-pixel fade tail moving left→right, wraps |
| `0x21` | **Wave left** — same, moving right→left |
| `0x22` | **Stop** — return to static with current color |
| `0x30` | **Sequential fill right** — LEDs fill one-by-one left→right (Corvette indicator), loops |
| `0x31` | **Sequential fill left** — fills right→left, loops |

Animations use the **last set color** (from button press or I2C color command). To change the animation color, send a color command first, then an animation command.

---

## Python examples (Raspberry Pi / smbus2)

```python
import smbus2

bus  = smbus2.SMBus(1)   # /dev/i2c-1
ADDR = 0x2A

def send(data: list[int]):
    msg = smbus2.i2c_msg.write(ADDR, data)
    bus.i2c_rdwr(msg)

# Predefined colors
send([0x00])   # Red
send([0x01])   # Green
send([0x02])   # Yellow
send([0x03])   # Purple

# Custom color (orange)
send([0x10, 255, 80, 0])

# Animations
send([0x20])   # wave right
send([0x21])   # wave left
send([0x30])   # sequential fill right (Corvette)
send([0x31])   # sequential fill left
send([0x22])   # stop → static

# Example: cyan Corvette sweep
send([0x10, 0, 220, 220])   # set color to cyan
send([0x30])                 # start fill
```

---

## File overview

| File | Purpose |
|------|---------|
| `status_leds.c` | Main firmware — PIO init, I2C slave, animation loop |
| `ws2812.pio` | PIO assembly for WS2812 bit timing |
| `CMakeLists.txt` | Build configuration (Pico SDK, PIO header generation) |
