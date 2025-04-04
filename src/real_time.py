import socket
import struct
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q
from collections import deque
import torch
from threading import Thread
from scipy.fft import fft

# === Model (Optional Import Placeholder) ===
from accel_trans import AccelTransformer
from constants import WINDOW_SIZE

ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"

# === UDP Setup ===
UDP_IP = "0.0.0.0"
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
    num_classes=6,
    n_seq_features=3,
    n_meta_features=9
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

def extract_window_signal_features(window):
    """Extracts mean, std, and FFT-based energy from window."""
    window = np.array(window)
    fft_values = fft(window, axis=0)
    fft_mag = np.abs(fft_values)[:WINDOW_SIZE//2]  # shape: (WINDOW_SIZE/2, 3)

    mean_mag = np.mean(window, axis=0).tolist()
    std_mag = np.std(window, axis=0).tolist()
    freq_energy = np.mean(fft_mag**2, axis=0).tolist()

    return mean_mag + std_mag + freq_energy  # Length: 3+3+3 = 9 features

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
        

def inference(model):
    while True:
        if len(accel_window) == WINDOW_SIZE:
            window = list(accel_window)
            features = extract_window_signal_features(window)

            # import torch
            with torch.no_grad():
                x_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 50, 3)
                x_meta = torch.tensor([features], dtype=torch.float32).to(DEVICE)           # (1, 9)
                logits = model(x_input, x_meta)
                pred = torch.argmax(logits, dim=1).item()
                print(f"\rPredicted class: {stylized_decoder[pred]}", end="", flush=True)

# === Launch Threads ===
Thread(target=listener, daemon=True).start()
Thread(target=inference, args=(model,), daemon=True).start()


# Keep main thread alive
while True:
    pass
