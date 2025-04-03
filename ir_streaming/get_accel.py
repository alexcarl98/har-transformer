import socket
import struct
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q

# === UDP Setup ===
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

# === Madgwick Filter ===
madgwick = Madgwick(sampleperiod=0.01)  # 100Hz
quaternion = None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for MPU6050 data on UDP {UDP_PORT}...")


while True:
    data, _ = sock.recvfrom(1024)
    x, y, z, gx, gy, gz = struct.unpack("ffffff", data)
    accel = np.array([x, y, z], dtype=np.float64)
    gyro = np.radians([gx, gy, gz], dtype=np.float64)

    if quaternion is None:
        quaternion = np.array(acc2q(accel), dtype=np.float64)
        print("Initialized quaternion from first reading.")
        continue

    # Update orientation with Madgwick filter
    quaternion = madgwick.updateIMU(q=quaternion, gyr=gyro, acc=accel)

    # Calculate gravity vector
    q = quaternion
    gravity = np.array([
        2 * (q[1]*q[3] - q[0]*q[2]),
        2 * (q[0]*q[1] + q[2]*q[3]),
        q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    ]) * 9.81

    # Subtract gravity
    linear_accel = accel - gravity
    
    # Print motion-only acceleration
    print("Linear Accel: x={:.2f}, y={:.2f}, z={:.2f}".format(*linear_accel))

'''
will need a multi-threaded approach:
- one thread for listening to UDP 
- one thread for processing data
- while queue is less than window size:
    - add data to queue
    - sleep for 1/100 seconds
- while queue is greater than window size:
    - process data
    - remove last two elements from from queue
    - sleep for 1/100 seconds


'''