import time
import serial
import pyvesc
from pyvesc.VESC.messages import SetRPM, SetCurrent

PORT = "/dev/ttyACM0"
BAUD = 115200

with serial.Serial(PORT, baudrate=BAUD, timeout=0.05) as ser:
    print(f"Opened {PORT}")

    # Send RPM command
    t_end = time.time() + 10.0
    while time.time() < t_end:
        ser.write(pyvesc.encode(SetRPM(6000)))
        print("Sent RPM = 2000")
        time.sleep(0.2)

    # Stop motor by commanding zero current
    ser.write(pyvesc.encode(SetCurrent(0)))
    print("Stopped motor")
