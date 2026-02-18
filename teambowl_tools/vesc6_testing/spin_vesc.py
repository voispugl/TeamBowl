import time
import serial
from pyvesc.VESC.messages import SetDutyCycle

PORT = "/dev/ttyACM0"
BAUD = 115200
DUTY = 0.08  # start low: 0.03-0.10

def send(ser, duty):
    duty = max(min(duty, 0.95), -0.95)
    ser.write(SetDutyCycle(duty).encode())

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(0.2)
    try:
        # gentle ramp
        for i in range(1, 21):
            send(ser, DUTY * i / 20.0)
            time.sleep(0.05)

        # hold 2s
        t0 = time.time()
        while time.time() - t0 < 2.0:
            send(ser, DUTY)
            time.sleep(0.05)

    finally:
        # stop (repeat a few times)
        for _ in range(10):
            send(ser, 0.0)
            time.sleep(0.02)
        ser.close()

if __name__ == "__main__":
    main()
