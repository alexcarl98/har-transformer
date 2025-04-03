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


# import socket
# import struct
# import numpy as np
# from ahrs.filters import Madgwick
# from ahrs.common.orientation import acc2q


# UDP_IP = "0.0.0.0"
# UDP_PORT = 5005
# madgwick = Madgwick(sampleperiod=0.01)  # 100 Hz sample rate
# quaternion = None


# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))

# print("Listening on UDP port", UDP_PORT)



# while True:
#     data, _ = sock.recvfrom(1024)
#     x, y, z, gx, gy, gz = struct.unpack("ffffff", data)
#     accel = np.array([x,y,z], dtype=np.float64)
#     gyro = np.radians([gx,gy,gz], dtype=np.float64)  # Convert to rad/s

#     if quaternion is None: 
#         quaternion = np.array(acc2q(accel), dtype=np.float64)
#         continue
#     else:
#         quaternion = madgwick.updateIMU(q=quaternion, gyr=gyro, acc=accel)

#     gravity = np.array([
#         2 * (quaternion[1]*quaternion[3] - quaternion[0]*quaternion[2]),
#         2 * (quaternion[0]*quaternion[1] + quaternion[2]*quaternion[3]),
#         quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2
#     ]) * 9.81

#     linear_accel = accel - gravity

#     print("Linear Accel: x={:.2f}, y={:.2f}, z={:.2f}".format(*linear_accel))

# import socket
# import pickle
# import numpy as np
# import cv2
# import time
# HOST = ''  # Listen on all interfaces
# PORT = 5050

# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.bind((HOST, PORT))
# sock.listen(1)

# print(f"Listening on port {PORT}...")
# conn, addr = sock.accept()
# print(f"Connection from {addr}")

# try:
#     while True:
#         # Read the first 4 bytes for the length
#         data_len = int.from_bytes(conn.recv(4), 'big')

#         # Read the full payload
#         data = b''
#         while len(data) < data_len:
#             packet = conn.recv(data_len - len(data))
#             if not packet:
#                 break
#             data += packet

#         if not data:
#             break

#         # Deserialize
#         acceleration = pickle.loads(data)  # shape (3,), dtype float64

#         # Show with OpenCV
#         print(acceleration)

#         # Exit on 'q' key
#         time.sleep(1/100)
# except KeyboardInterrupt:
#     print("Interrupted.")
# finally:
#     conn.close()
#     sock.close()
#     cv2.destroyAllWindows()
