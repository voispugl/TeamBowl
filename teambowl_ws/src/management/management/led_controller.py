#!/usr/bin/env python3
"""
LED Controller for TeamBowl.
Controls NeoPixel RGB strip via GPIO4 (Pin 7) on Jetson AGX Orin.

Priority (highest wins):
  /estop True        → solid red
  /robot_stuck True  → pulsing red
  /yolo26/user_valid True  → solid green
  default            → solid blue (person not seen)
"""

import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import board
    import neopixel
    HAS_NEOPIXEL = True
except ImportError:
    HAS_NEOPIXEL = False


class LEDController(Node):
    def __init__(self):
        super().__init__('led_controller')

        self.declare_parameter('num_pixels', 30)
        self.declare_parameter('brightness', 0.2)
        self.num_pixels = self.get_parameter('num_pixels').value
        brightness = self.get_parameter('brightness').value

        if not HAS_NEOPIXEL:
            self.get_logger().error('neopixel or blinka library not found! LEDs will not work.')
            self.pixels = None
        else:
            try:
                self.pixels = neopixel.NeoPixel(
                    board.D4,
                    self.num_pixels,
                    brightness=brightness,
                    auto_write=False,
                    pixel_order=neopixel.GRB,
                )
                self.get_logger().info(f'Initialized {self.num_pixels} NeoPixels on GPIO4 (Pin 7).')
            except Exception as e:
                self.get_logger().error(f'Failed to initialize NeoPixels: {e}')
                self.pixels = None

        self._estop = False
        self._stuck = False
        self._user_valid = False

        self.pulse_thread = None
        self.stop_pulse = threading.Event()

        self.create_subscription(Bool, '/estop', self._estop_cb, 10)
        self.create_subscription(Bool, '/robot_stuck', self._stuck_cb, 10)
        self.create_subscription(Bool, '/yolo26/user_valid', self._user_valid_cb, 10)

        # Startup: blue (no person seen yet)
        self.set_color(0, 0, 255)

    def _estop_cb(self, msg: Bool):
        self._estop = msg.data
        self._update_leds()

    def _stuck_cb(self, msg: Bool):
        self._stuck = msg.data
        self._update_leds()

    def _user_valid_cb(self, msg: Bool):
        self._user_valid = msg.data
        self._update_leds()

    def _update_leds(self):
        if self._estop:
            self.stop_pulsing()
            self.set_color(255, 0, 0)
        elif self._stuck:
            self.start_pulsing(255, 0, 0)
        elif self._user_valid:
            self.stop_pulsing()
            self.set_color(0, 255, 0)
        else:
            self.stop_pulsing()
            self.set_color(0, 0, 255)

    def set_color(self, r, g, b):
        if self.pixels:
            try:
                self.pixels.fill((int(r), int(g), int(b)))
                self.pixels.show()
            except Exception as e:
                self.get_logger().error(f'Error writing to LEDs: {e}')

    def start_pulsing(self, r, g, b):
        self.stop_pulsing()
        self.stop_pulse.clear()
        self.pulse_thread = threading.Thread(target=self._pulse_worker, args=(r, g, b), daemon=True)
        self.pulse_thread.start()

    def stop_pulsing(self):
        if self.pulse_thread:
            self.stop_pulse.set()
            self.pulse_thread.join()
            self.pulse_thread = None

    def _pulse_worker(self, r, g, b):
        while not self.stop_pulse.is_set():
            for i in range(0, 101, 5):
                if self.stop_pulse.is_set():
                    return
                f = i / 100.0
                self.set_color(r * f, g * f, b * f)
                time.sleep(0.05)
            for i in range(100, -1, -5):
                if self.stop_pulse.is_set():
                    return
                f = i / 100.0
                self.set_color(r * f, g * f, b * f)
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
        node.set_color(0, 0, 0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
