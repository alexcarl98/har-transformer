import time
import board
import adafruit_mpu6050
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q

import numpy as np

# === Setup ===
madgwick = Madgwick(sampleperiod=0.01)  # 100 Hz sample rate

i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

TARGET_RATE_HZ = 100
PERIOD = 1.0 / TARGET_RATE_HZ

accel = np.array(mpu.acceleration)
# function estimates quaternion from acceleration
quaternion = np.array(acc2q(accel), dtype=np.float64)
print("Starting 100Hz sensor loop...")

while True:
    start_time = time.perf_counter()  # High-res timing

    # === Read sensor data ===
    raw_accel = mpu.acceleration
    raw_gyro = mpu.gyro
    accel = np.array(raw_accel, dtype=np.float64)
    gyro = np.radians(raw_gyro, dtype=np.float64)  # Convert to rad/s

    # === Update orientation filter ===
    quaternion = madgwick.updateIMU(q=quaternion, gyr=gyro, acc=accel)

    # === Calculate gravity vector from quaternion ===
    gravity = np.array([
        2 * (quaternion[1]*quaternion[3] - quaternion[0]*quaternion[2]),
        2 * (quaternion[0]*quaternion[1] + quaternion[2]*quaternion[3]),
        quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2
    ]) * 9.81

    # === Remove gravity from acceleration ===
    linear_accel = accel - gravity

    # === Output (comment out prints to go full speed) ===
    print("Linear Accel: x={:.2f}, y={:.2f}, z={:.2f}".format(*linear_accel))

    # === Sleep to maintain 100Hz ===
    elapsed = time.perf_counter() - start_time
    time.sleep(max(0, PERIOD - elapsed))
