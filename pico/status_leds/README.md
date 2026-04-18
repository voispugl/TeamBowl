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

### Kill switch / lid button (GP15)

Wire a normally-open momentary switch between **GP15 (pin 20)** and **GND (any GND pin)**. The firmware enables the internal pull-up — no external resistor needed.

| Terminal | Connect to |
|----------|-----------|
| A        | GP15 (pin 20) |
| B        | GND (e.g. pin 18) |

**Behavior (decided by the `pico_bridge` ROS2 node):**
- **While moving** — press asserts `/kill_switch true` → `system_health` triggers e-stop, LEDs go red.
- **While stopped** — press publishes `"toggle"` to `/lid_command` → lid opens or closes.

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

## Finding the Pico serial port

The Pico appears as a USB CDC device. Use the stable `by-id` symlink so the path doesn't change if USB devices are re-ordered.

```bash
ls /dev/serial/by-id/
```

Look for an entry containing `Raspberry_Pi_Pico` or `RP2350`, e.g.:

```
usb-Raspberry_Pi_Pico_2_E6614103E3536E28-if00
```

Copy the **full path** into `safety.yaml`:

```yaml
pico_bridge:
  ros__parameters:
    serial_port: "/dev/serial/by-id/usb-Raspberry_Pi_Pico_2_E6614103E3536E28-if00"
```

Alternatively, enumerate ports with Python:

```bash
python3 -c "import serial.tools.list_ports; [print(p.device, p.description) for p in serial.tools.list_ports.comports()]"
```

---

## ROS2 LED states

The `pico_bridge` node maps robot state to LED colors automatically. Manual overrides are possible by sending raw bytes to the serial port.

| State | Color | Pattern |
|-------|-------|---------|
| E-stopped | Red | Solid |
| Turning right | Orange | Wave right |
| Turning left | Orange | Wave left |
| Moving forward/back | Yellow | Solid |
| Stuck (`/robot_stuck true`) | Purple | Blink ~3 Hz |
| Lid open / Alive | Green | Solid |

---

## I2C / USB-serial command reference

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

### Blink — 4 bytes

```
0x40  R  G  B
```

Starts a ~3 Hz blink using the given color (all pixels toggle on/off). Used for "stuck" indication (purple: `0x40 0x80 0x00 0x80`).

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
| `status_leds.c` | Main firmware — PIO init, I2C slave, USB-serial, kill switch, animation loop |
| `ws2812.pio` | PIO assembly for WS2812 bit timing |
| `CMakeLists.txt` | Build configuration (Pico SDK, PIO header gen, USB CDC enabled) |
