import board
import busio
import socket
import time
import struct
import numpy as np
from collections import deque
from adafruit_mpu6050 import MPU6050

def setup_mpu():
    """Initialize I2C and MPU6050 sensor"""
    i2c = busio.I2C(board.SCL, board.SDA)
    return MPU6050(i2c)

def setup_udp_socket():
    """Create UDP socket"""
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def collect_sensor_data(mpu, readings_buffer, window_size):
    """Collect one sensor reading and add to buffer"""
    accel = mpu.acceleration
    gyro = mpu.gyro
    readings_buffer.append((*accel, *gyro))
    return len(readings_buffer) == window_size

def send_batch(sock, readings_buffer, udp_ip, udp_port):
    """Send batch of readings over UDP"""
    payload = struct.pack(f"{len(readings_buffer) * 6}f", 
                         *[val for reading in readings_buffer for val in reading])
    sock.sendto(payload, (udp_ip, udp_port))
    readings_buffer.clear()

def run_sensor_stream(udp_ip, udp_port, window_size=100, target_hz=100):
    """Main function to run the sensor data collection and transmission"""
    mpu = setup_mpu()
    sock = setup_udp_socket()
    readings_buffer = deque(maxlen=window_size)
    period = 1.0 / target_hz

    print(f"Collecting MPU6050 data at {target_hz}Hz...")
    print(f"Will send batches of {window_size} readings to {udp_ip}:{udp_port}")

    try:
        while True:
            start_time = time.perf_counter()
            
            try:
                if collect_sensor_data(mpu, readings_buffer, window_size):
                    send_batch(sock, readings_buffer, udp_ip, udp_port)
                    print(f"\rSent batch of {window_size} readings", end="")
                    
            except Exception as e:
                print("\nSensor read error:", e)
            
            time.sleep(max(0, period - (time.perf_counter() - start_time)))
            
    except KeyboardInterrupt:
        print("\nStopping data collection...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream MPU6050 sensor data over UDP')
    parser.add_argument('--ip', default='172.20.10.11', help='UDP destination IP')
    parser.add_argument('--port', type=int, default=5005, help='UDP destination port')
    parser.add_argument('--window', type=int, default=100, help='Number of readings per batch')
    parser.add_argument('--hz', type=int, default=100, help='Target sampling rate in Hz')
    
    args = parser.parse_args()
    run_sensor_stream(args.ip, args.port, args.window, args.hz) 