# neopixel_write.py — Local shadow of adafruit's neopixel_write module.
#
# Placed in /home/box/TeamBowl/ so that sys.path[0] (the script's directory)
# causes Python to find this file before the broken site-packages version,
# which raises NotImplementedError("Board not supported") on Jetson AGX Orin.
#
# This module:
#   1. Stubs out digitalio.DigitalInOut for board.D4 so Blinka never tries to
#      claim gpiochip0 line 106 — which would conflict (EBUSY) with the C
#      library that opens the same line at dlopen time.
#   2. Loads libneopixel_cdev.so via ctypes.
#   3. Exposes neopixel_write(gpio, buf) matching the expected interface.

import os
import ctypes
import board
import digitalio

# ---------------------------------------------------------------------------
# Step 1: Stub DigitalInOut for our neopixel pin
#
# When ctypes.CDLL() below runs, the C library's __attribute__((constructor))
# fires immediately and claims gpiochip0 line 106.  If Blinka's DigitalInOut
# later tries to claim the same line (in NeoPixel.__init__) it will get EBUSY
# and raise an exception.  We prevent this by replacing digitalio.DigitalInOut
# with a factory that returns a lightweight stub for board.D4 and the real
# class for everything else.
# ---------------------------------------------------------------------------

_NEOPIXEL_PIN = board.D4
_OrigDigitalInOut = digitalio.DigitalInOut


class _DigitalInOutStub:
    """No-op stand-in for DigitalInOut on the neopixel pin."""

    def __init__(self, pin):
        self.pin = pin

    def switch_to_output(self, value=False,
                         drive_mode=digitalio.DriveMode.PUSH_PULL):
        pass

    def deinit(self):
        pass

    @property
    def value(self):
        return False

    @value.setter
    def value(self, val):
        pass


def _PatchedDigitalInOut(pin):
    if pin is _NEOPIXEL_PIN:
        return _DigitalInOutStub(pin)
    return _OrigDigitalInOut(pin)


# Make isinstance() checks on the class itself still work for other pins
_PatchedDigitalInOut.__name__ = "DigitalInOut"
_PatchedDigitalInOut.__qualname__ = "DigitalInOut"
digitalio.DigitalInOut = _PatchedDigitalInOut

# ---------------------------------------------------------------------------
# Step 2: Load the C shared library
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "libneopixel_cdev.so")

try:
    _lib = ctypes.CDLL(_SO)
except OSError as exc:
    raise ImportError(
        f"neopixel_write: could not load {_SO}.\n"
        f"Run:  sudo bash {_HERE}/build_neopixel.sh\n"
        f"Original error: {exc}"
    ) from exc

# void neopixel_write_c(const uint8_t *buf, int len)
_lib.neopixel_write_c.restype = None
_lib.neopixel_write_c.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int,
]

# ---------------------------------------------------------------------------
# Step 3: Public function — matches the signature neopixel.py calls
# ---------------------------------------------------------------------------


def neopixel_write(gpio, buf):
    """
    Write WS2812B pixel data via the C GPIO cdev library.

    Parameters
    ----------
    gpio : DigitalInOut (stub)
        Ignored.  The C library directly controls gpiochip0 line 106
        (board.D4 / PQ.06 / GP66 on Jetson AGX Orin).
    buf  : bytearray or bytes
        GRB-ordered pixel data as produced by adafruit_pixelbuf.
        Length = NUM_PIXELS * bytes_per_pixel.
    """
    if not buf:
        return
    data = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    _lib.neopixel_write_c(data, len(buf))
