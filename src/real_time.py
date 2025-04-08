"""
KNOWN BUGS: (2025-04-03):

This script is affected by several real-time data display issues:

- UDP packets may arrive out of order, disrupting the time series.
- The inference loop processes every point in the deque, even outdated ones.
  As a result, predictions may lag behind by several seconds.

Example: The model will output "upstairs" a minute after the user has stopped walking upstairs.

FIXME: (unimplemented):
- Only process the most recent 100 points.
- Drop old packets and/or use a timestamped sliding buffer.
"""

import socket
import struct
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q
from collections import deque
import torch
from threading import Thread
import time

# === Model (Optional Import Placeholder) ===
from har_model import AccelTransformer
from constants import WINDOW_SIZE, SZ_SEQ_DATA, SZ_META_DATA, NUM_CLS
from preprocessing import extract_window_signal_features

ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"

# === UDP Setup ===
UDP_IP = ""
stylized_decoder= {
    0: f'{ANSI_BLUE}downstairs      {ANSI_RESET}', 
    1: f'{ANSI_YELLOW}jog_treadmill{ANSI_RESET}', 
    2: f'{ANSI_MAGENTA}upstairs       {ANSI_RESET}', 
    3: f'{ANSI_CYAN}walk_mixed      {ANSI_RESET}', 
    4: f'{ANSI_GREEN}walk_sidewalk    {ANSI_RESET}', 
    5: f'{ANSI_RED}walk_treadmill      {ANSI_RESET}'}

UDP_PORT = 5005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening for MPU6050 data on UDP {UDP_PORT}...")

model = AccelTransformer(
    num_classes=NUM_CLS,
    in_seq_dim=SZ_SEQ_DATA,
    in_meta_dim=SZ_META_DATA
).to(DEVICE)

checkpoint = torch.load('accel_transformer.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(torch.load("accel_transformer.pth", map_location=DEVICE))
model.eval()

# === Madgwick Filter ===
madgwick = Madgwick(sampleperiod=0.01)  # 100Hz
quaternion = None

# === Buffers and Constants ===

accel_window = deque(maxlen=WINDOW_SIZE)

def listener():
    global quaternion

    while True:
        data, _ = sock.recvfrom(1024)
        x, y, z, gx, gy, gz = struct.unpack("ffffff", data)
        accel = np.array([x, y, z], dtype=np.float64)
        gyro = np.radians([gx, gy, gz], dtype=np.float64)

        if quaternion is None:
            quaternion = np.array(acc2q(accel), dtype=np.float64)
            print("Initialized quaternion from first reading.")
            continue

        quaternion = madgwick.updateIMU(q=quaternion, gyr=gyro, acc=accel)

        # Calculate gravity vector
        q = quaternion
        gravity = np.array([
            2 * (q[1]*q[3] - q[0]*q[2]),
            2 * (q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ]) * 9.81

        linear_accel = accel - gravity
        accel_window.append(linear_accel)
        

last_pred = None

def inference(model):
    global last_pred
    while True:
        if len(accel_window) == WINDOW_SIZE:
            window = list(accel_window)
            features = extract_window_signal_features(window)

            with torch.no_grad():
                x_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                x_meta = torch.tensor([features], dtype=torch.float32).to(DEVICE)
                logits = model(x_input, x_meta)
                pred = torch.argmax(logits, dim=1).item()

                if pred != last_pred:
                    print(f"\rPredicted class: {stylized_decoder[pred]}", end="", flush=True)
                    last_pred = pred

# === Launch Threads ===
Thread(target=listener, daemon=True).start()
Thread(target=inference, args=(model,), daemon=True).start()


# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")