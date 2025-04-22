import socket
import struct
import time
import numpy as np
from datetime import datetime

# UDP Setup
UDP_IP = "0.0.0.0"  # Listen on all available interfaces
UDP_PORT = 5005
BUFFER_SIZE = 1024

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for accelerometer data on port {UDP_PORT}...")
print("Waiting for data from Raspberry Pi (172.20.10.14)...")

try:
    while True:
        # Receive data
        data, addr = sock.recvfrom(BUFFER_SIZE)
        timestamp = datetime.now()
        
        # Unpack the 6 float values (3 accel, 3 gyro)
        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = struct.unpack("ffffff", data)
        
        # Print the data
        print(f"\rTime: {timestamp.strftime('%H:%M:%S.%f')[:-3]} | "
              f"Accel (m/s²): ({accel_x:6.2f}, {accel_y:6.2f}, {accel_z:6.2f}) | "
              f"Gyro (°/s): ({gyro_x:6.2f}, {gyro_y:6.2f}, {gyro_z:6.2f})", end="")

except KeyboardInterrupt:
    print("\nStopping listener...")
finally:
    sock.close()