import time
import serial
import pyvesc
from pyvesc.VESC.messages import SetRPM, SetCurrent

PORT = "/dev/ttyACM0"
BAUD = 115200

with serial.Serial(PORT, baudrate=BAUD, timeout=0.05) as ser:
    print(f"Opened {PORT}")

    # Send RPM command
    ser.write(pyvesc.encode(SetRPM(2000)))
    print("Sent RPM = 2000")
    time.sleep(2.0)

    # Stop motor by commanding zero current
    ser.write(pyvesc.encode(SetCurrent(0)))
    print("Stopped motor")
