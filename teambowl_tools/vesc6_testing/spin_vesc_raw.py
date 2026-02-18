import time
import serial
import struct

PORT = "/dev/ttyACM0"
BAUD = 115200

COMM_SET_DUTY = 5  # VESC "set duty" command id

def crc16_ccitt(data: bytes, poly=0x1021, init=0x0000) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc

def vesc_packet(payload: bytes) -> bytes:
    l = len(payload)
    if l < 256:
        header = bytes([2, l])          # short packet
    else:
        header = bytes([3, (l>>8)&0xFF, l&0xFF])  # long packet
    crc = crc16_ccitt(payload)
    return header + payload + bytes([(crc>>8)&0xFF, crc&0xFF]) + bytes([3])

def set_duty(duty: float) -> bytes:
    duty = max(min(duty, 0.95), -0.95)
    duty_i = int(duty * 100000)         # VESC scaling
    payload = bytes([COMM_SET_DUTY]) + struct.pack(">i", duty_i)
    return vesc_packet(payload)

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(0.2)

    target = 0.08  # 8% duty (start small)
    try:
        # ramp up
        for i in range(1, 21):
            ser.write(set_duty(target * i / 20.0))
            time.sleep(0.05)

        # hold 2s
        t0 = time.time()
        while time.time() - t0 < 2.0:
            ser.write(set_duty(target))
            time.sleep(0.05)

    finally:
        # stop
        for _ in range(10):
            ser.write(set_duty(0.0))
            time.sleep(0.02)
        ser.close()

if __name__ == "__main__":
    main()
