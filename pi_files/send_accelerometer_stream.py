import board
import busio
import socket
import time
import struct
from adafruit_mpu6050 import MPU6050

# === I2C & MPU Setup ===
i2c = busio.I2C(board.SCL, board.SDA)
mpu = MPU6050(i2c)

# === UDP Setup ===
UDP_IP = "172.20.10.11"  # Replace with your PC's IP
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# === Timing Setup ===
TARGET_HZ = 100
PERIOD = 1.0 / TARGET_HZ

print(f"Streaming MPU6050 data at {TARGET_HZ}Hz to {UDP_IP}:{UDP_PORT}...")

while True:
    start_time = time.perf_counter()
    try:
        accel = mpu.acceleration  # (x, y, z) in m/s²
        gyro = mpu.gyro           # (x, y, z) in °/s
        payload = struct.pack("ffffff", *accel, *gyro)
        
        sock.sendto(payload, (UDP_IP, UDP_PORT))
    except Exception as e:
        print("Sensor read error:", e)
    
    time.sleep(max(0, PERIOD - (time.perf_counter() - start_time)))