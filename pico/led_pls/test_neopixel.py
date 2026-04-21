import machine
import time
import rp2

NUM_PIXELS = 6
PIN_NUM = 28
BRIGHTNESS = 0.3

led = machine.Pin("LED", machine.Pin.OUT)

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

sm = rp2.StateMachine(0, ws2812, freq=8_000_000, sideset_base=machine.Pin(PIN_NUM))
sm.active(1)

def set_pixels(r, g, b):
    color = (int(g * BRIGHTNESS) << 16) | (int(r * BRIGHTNESS) << 8) | int(b * BRIGHTNESS)
    led.on()
    for _ in range(NUM_PIXELS):
        sm.put(color, 8)
    time.sleep_us(50)
    led.off()

colors = [
    (255, 0,   0),    # Red
    (0,   255, 0),    # Green
    (255, 255, 0),    # Yellow
    (128, 0,   128),  # Purple
]

current_idx = 0
last_button_state = False

set_pixels(*colors[current_idx])
print("NeoPixel ready — press BOOTSEL to cycle colors.")

while True:
    try:
        button_pressed = rp2.bootsel_button()
        if button_pressed and not last_button_state:
            current_idx = (current_idx + 1) % len(colors)
            print(f"Color: {colors[current_idx]}")
            set_pixels(*colors[current_idx])
        last_button_state = button_pressed
        time.sleep_ms(20)
    except Exception as e:
        print(f"Error: {e}")
        continue
