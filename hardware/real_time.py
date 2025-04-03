import socket
import struct
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q
from queue import Queue
from threading import Thread
from scipy.fft import fft


# === UDP Setup ===
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

# === Madgwick Filter ===
madgwick = Madgwick(sampleperiod=0.01)  # 100Hz
quaternion = None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for MPU6050 data on UDP {UDP_PORT}...")


accel_queue = Queue()
WINDOW_SIZE = 50

def extract_window_signal_features(window):
    fft_values = fft(window)
    fft_mag = np.abs(fft_values)[:WINDOW_SIZE//2]

    mean_mag = list(np.mean(window, axis=0))
    std_mag = list(np.std(window, axis=0))

    freq_energy = list(np.mean(fft_mag**2, axis=0))

    extracted = [*mean_mag, *std_mag, *freq_energy]
    return extracted


def listener():
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
        mag = np.linalg.norm(linear_accel)
        
        # Print motion-only acceleration
        print("Linear Accel: x={:.2f}, y={:.2f}, z={:.2f}".format(*linear_accel))
        accel_queue.put(linear_accel)

def inference(model):
    while True:
        if accel_queue.qsize() >= 100:
            data = accel_queue.get()
            print(data)

