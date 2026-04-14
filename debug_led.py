#!/usr/bin/env python3
"""
NeoPixel Debug Script (Bit-bang version) for Jetson AGX Orin.
Requires: pip3 install adafruit-circuitpython-neopixel adafruit-blinka

Hardware:
- Connect NeoPixel DIN to Pin 7 (GPIO4 / AUD_MCLK) on the 40-pin header.
- Connect NeoPixel GND to Pin 6, 9, 14, 20, 25, 30, 34, or 39.
- Connect NeoPixel 5V to an external 5V supply.

Setup:
1. Ensure the user is in the 'gpio' group: sudo usermod -aG gpio $USER
2. Reboot or log out/in.
"""

import sys
import time
import board
import neopixel

# Using board.D4 (Pin 7 / GPIO4) for bit-banging.
# WARNING: Bit-banging WS2812 LEDs from user-space Python on Jetson/Linux is 
# extremely sensitive to timing. If the strip flickers or shows wrong colors,
# it is because the OS kernel is interrupting the Python process.
PIXEL_PIN = board.D4
NUM_PIXELS = 30  # Adjust for your strip length
ORDER = neopixel.GRB

def main():
    print(f"Initializing NeoPixel strip on GPIO4 (Pin 7) with {NUM_PIXELS} pixels...")
    
    try:
        # Standard NeoPixel constructor (attempts bit-bang/PWM via Blinka)
        pixels = neopixel.NeoPixel(
            PIXEL_PIN, 
            NUM_PIXELS, 
            brightness=0.2, 
            auto_write=False, 
            pixel_order=ORDER
        )
    except Exception as e:
        print(f"Error: Could not initialize NeoPixels. {e}")
        print("\nPossible fixes:")
        print("1. Run with sudo if you have permission errors.")
        print("2. Ensure 'adafruit-blinka' and 'adafruit-circuitpython-neopixel' are installed.")
        print("3. If you get a 'NeoPixel involves precise timing' error, the Adafruit")
        print("   library is refusing to bit-bang on this pin because it's unreliable.")
        return

    print("Commands: r (red), g (green), b (blue), w (white), o (off), rainbow, q (quit)")
    
    while True:
        cmd = input("Enter command: ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'r':
            pixels.fill((255, 0, 0))
        elif cmd == 'g':
            pixels.fill((0, 255, 0))
        elif cmd == 'b':
            pixels.fill((0, 0, 255))
        elif cmd == 'w':
            pixels.fill((255, 255, 255))
        elif cmd == 'o':
            pixels.fill((0, 0, 0))
        elif cmd == 'rainbow':
            print("Running rainbow cycle (Ctrl+C to stop)...")
            try:
                for j in range(255):
                    for i in range(NUM_PIXELS):
                        pixel_index = (i * 256 // NUM_PIXELS) + j
                        pixels[i] = wheel(pixel_index & 255)
                    pixels.show()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                pass
        else:
            print("Unknown command.")
            continue
            
        pixels.show()

def wheel(pos):
    if pos < 0 or pos > 255:
        return (0, 0, 0)
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    pos -= 170
    return (pos * 3, 0, 255 - pos * 3)

if __name__ == "__main__":
    main()
