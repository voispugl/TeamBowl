#!/usr/bin/env python3
"""
LED Controller for TeamBowl.
Controls NeoPixel RGB strip via GPIO4 (Pin 7) on Jetson AGX Orin.

Subscriptions:
- /lid_state (std_msgs/String): Changes color based on cargo lid status
- /led_rgb (std_msgs/ColorRGBA): Manual override for specific colors
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ColorRGBA
import time
import threading

try:
    import board
    import neopixel
    HAS_NEOPIXEL = True
except ImportError:
    HAS_NEOPIXEL = False

class LEDController(Node):
    def __init__(self):
        super().__init__('led_controller')
        
        # Parameters
        self.declare_parameter('num_pixels', 30)
        self.declare_parameter('brightness', 0.2)
        self.num_pixels = self.get_parameter('num_pixels').value
        brightness = self.get_parameter('brightness').value
        
        if not HAS_NEOPIXEL:
            self.get_logger().error("neopixel or blinka library not found! LEDs will not work.")
            self.pixels = None
        else:
            try:
                # Use GPIO4 (board.D4 / Pin 7) for bit-banging
                self.pixels = neopixel.NeoPixel(
                    board.D4,
                    self.num_pixels,
                    brightness=brightness,
                    auto_write=False,
                    pixel_order=neopixel.GRB
                )
                self.get_logger().info(f"Initialized {self.num_pixels} NeoPixels on GPIO4 (Pin 7).")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize NeoPixels: {e}")
                self.get_logger().error("Try running with sudo or ensuring user is in 'gpio' group.")
                self.pixels = None

        # State
        self.current_rgb = (0, 0, 0)
        self.last_lid_state = 'unknown'
        self.pulse_thread = None
        self.stop_pulse = threading.Event()
        self.debug_active = False

        # Subscriptions
        self.create_subscription(String, '/lid_state', self.lid_state_callback, 10)
        self.create_subscription(ColorRGBA, '/led_rgb', self.manual_rgb_callback, 10)

        # Initial state: Blue for startup
        self.set_color(0, 0, 255)

    def set_color(self, r, g, b, show=True):
        if self.pixels:
            try:
                self.pixels.fill((int(r), int(g), int(b)))
                if show:
                    self.pixels.show()
            except Exception as e:
                self.get_logger().error(f"Error writing to LEDs: {e}")
        self.current_rgb = (r, g, b)

    def manual_rgb_callback(self, msg):
        """Handle manual RGB override."""
        # Use alpha channel > 1.0 as a signal for debug mode if needed, 
        # or just respond to any manual RGB as an override.
        self.stop_pulsing()
        
        # Special check for a 'debug' signal (e.g. if alpha is 0.5)
        if msg.a == 0.5:
            self.get_logger().info("Debug sequence triggered via /led_rgb")
            self.run_debug_sequence()
            return

        # msg values are usually 0.0 - 1.0
        self.set_color(msg.r * 255, msg.g * 255, msg.b * 255)
        self.get_logger().info(f"Manual LED set to R:{msg.r} G:{msg.g} B:{msg.b}")

    def lid_state_callback(self, msg):
        """React to cargo bay lid state changes."""
        state = msg.data
        if state == self.last_lid_state:
            return
        
        self.last_lid_state = state
        self.stop_pulsing()

        if state == 'open':
            self.set_color(0, 255, 0) # Solid Green
        elif state == 'closed':
            self.set_color(0, 0, 255) # Solid Blue
        elif state in ('moving_open', 'moving_closed'):
            self.start_pulsing(255, 255, 0) # Pulse Yellow
        elif state == 'unknown':
            self.set_color(255, 0, 0) # Solid Red (Error/Warning)

    def run_debug_sequence(self):
        """Run a quick RGB cycle for hardware validation."""
        def _debug():
            for c in [(255,0,0), (0,255,0), (0,0,255), (0,0,0)]:
                self.set_color(*c)
                time.sleep(1.0)
            self.lid_state_callback(String(data=self.last_lid_state)) # Restore
        threading.Thread(target=_debug).start()

    def start_pulsing(self, r, g, b):
        self.stop_pulsing()
        self.stop_pulse.clear()
        self.pulse_thread = threading.Thread(target=self.pulse_worker, args=(r, g, b))
        self.pulse_thread.start()

    def stop_pulsing(self):
        if self.pulse_thread:
            self.stop_pulse.set()
            self.pulse_thread.join()
            self.pulse_thread = None

    def pulse_worker(self, r, g, b):
        """Worker thread for pulsing animations."""
        while not self.stop_pulse.is_set():
            # Fade in
            for i in range(0, 101, 5):
                if self.stop_pulse.is_set(): break
                factor = i / 100.0
                self.set_color(r * factor, g * factor, b * factor)
                time.sleep(0.05)
            # Fade out
            for i in range(100, -1, -5):
                if self.stop_pulse.is_set(): break
                factor = i / 100.0
                self.set_color(r * factor, g * factor, b * factor)
                time.sleep(0.05)

def main(args=None):
    rclpy.init(args=args)
    node = LEDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_pulsing()
        node.set_color(0, 0, 0) # Turn off on exit
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
