import machine
import time
import rp2

# NeoPixel configuration
NUM_PIXELS = 10
PIN_NUM = 28
BRIGHTNESS = 0.3

@rp2.asm_pio(sideset_init=rp2.PIO.OUT_LOW, out_shiftdir=rp2.PIO.SHIFT_LEFT, autopull=True, pull_thresh=24)
def ws2812():
    T1 = 2
    T2 = 5
    T3 = 3
    wrap_target()
    label("bitloop")
    out(x, 1)               .side(0) [T3 - 1]
    jmp(not_x, "do_zero")   .side(1) [T1 - 1]
    jmp("bitloop")          .side(1) [T2 - 1]
    label("do_zero")
    nop()                   .side(0) [T2 - 1]
    wrap()

# Initialize PIO State Machine for NeoPixels
sm = rp2.StateMachine(0, ws2812, freq=8_000_000, sideset_base=machine.Pin(PIN_NUM))
sm.active(1)

def set_pixels(r, g, b):
    # Apply brightness and convert to GRB 24-bit
    # MicroPython PIO expects data in the upper bits if shifting left
    color = (int(g * BRIGHTNESS) << 16) | (int(r * BRIGHTNESS) << 8) | int(b * BRIGHTNESS)
    for _ in range(NUM_PIXELS):
        sm.put(color << 8)

# Define our color sequence: Red, Green, Yellow, Purple
colors = [
   
    (0, 255, 0),    # Green
    (255, 0, 0),    # Red
    (255, 255, 0),  # Yellow
    (128, 0, 128)   # Purple
]

current_idx = 0
last_button_state = False

print("Starting NeoPixel Test...")
print("Press the BOOTSEL button to cycle colors.")

# On Pico 2 / RP2350, MicroPython provides access to the BOOTSEL button 
# via a special virtual pin or constant in some builds, but we can also
# use the rp2.bootsel_button() function if available.
# Note: In standard MicroPython, reading BOOTSEL is often:
# rp2.bootsel_button()

while True:
    try:
        # Check BOOTSEL button state
        # 1 = pressed, 0 = released
        button_pressed = rp2.bootsel_button()
        
        if button_pressed and not last_button_state:
            current_idx = (current_idx + 1) % len(colors)
            r, g, b = colors[current_idx]
            print(f"Button Pressed! Color: {colors[current_idx]}")
            set_pixels(r, g, b)
            
        last_button_state = button_pressed
        time.sleep(0.02) # Debounce
        
    except Exception as e:
        print(f"Error: {e}")
        break
