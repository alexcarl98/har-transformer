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
UDP_IP = "192.168.86.43"  # Replace with your PC's IP
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

# import board
# import busio
# import socket
# import time
# import struct
# from adafruit_mpu6050 import MPU6050

# # === Setup I2C + Sensor ===
# i2c = busio.I2C(board.SCL, board.SDA)
# mpu = MPU6050(i2c)

# # === Setup UDP socket ===
# UDP_IP = "192.168.86.43"  # Receiver IP address (change to your PC)
# UDP_PORT = 5005
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# # === Target rate (Hz) ===
# TARGET_RATE = 100
# PERIOD = 1.0 / TARGET_RATE

# print("Streaming at ~100Hz to", UDP_IP, "on port", UDP_PORT)

# while True:
#     print("Acceleration:", mpu.acceleration)

#     start = time.time()

#     # Get acceleration data (x, y, z)
#     accel = mpu.acceleration  # in m/s²
#     gyro = mpu.gyro  # in deg/s
#     # Pack into binary format (3 floats)
#     message = struct.pack("ffffff", *accel, *gyro)

#     # Send over UDP
#     sock.sendto(message, (UDP_IP, UDP_PORT))

#     # Wait to maintain 100Hz
#     elapsed = time.time() - start
#     delay = max(0, PERIOD - elapsed)
#     time.sleep(delay)
