import socket
import struct
import numpy as np
from datetime import datetime

def setup_udp_socket(udp_ip, udp_port):
    """Initialize and bind UDP socket"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    return sock

def calculate_buffer_size(window_size):
    """Calculate required buffer size based on window size"""
    return 4 * 6 * window_size  # 4 bytes per float * 6 values per reading * window size

def process_batch(data, window_size):
    """Process received batch data into numpy array and calculate statistics"""
    values = struct.unpack(f"{window_size * 6}f", data)
    readings = np.array(values).reshape(window_size, 6)
    
    accel_mean = readings[:, :3].mean(axis=0)
    gyro_mean = readings[:, 3:].mean(axis=0)
    
    return accel_mean, gyro_mean

def print_batch_stats(timestamp, accel_mean, gyro_mean):
    """Print formatted batch statistics"""
    print(f"\nTime: {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"Batch Statistics:")
    print(f"Accelerometer mean (m/s²): ({accel_mean[0]:6.2f}, {accel_mean[1]:6.2f}, {accel_mean[2]:6.2f})")
    print(f"Gyroscope mean (°/s):     ({gyro_mean[0]:6.2f}, {gyro_mean[1]:6.2f}, {gyro_mean[2]:6.2f})")
    print("-" * 50)

def run_receiver(udp_ip, udp_port, window_size=100):
    """Main function to run the data receiver"""
    sock = setup_udp_socket(udp_ip, udp_port)
    buffer_size = calculate_buffer_size(window_size)

    print(f"Listening for accelerometer batches on port {udp_port}...")
    print(f"Expecting batches of {window_size} readings from Raspberry Pi")
    print("Format: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] × {window_size}")

    try:
        while True:
            data, addr = sock.recvfrom(buffer_size)
            timestamp = datetime.now()
            
            accel_mean, gyro_mean = process_batch(data, window_size)
            print_batch_stats(timestamp, accel_mean, gyro_mean)

    except KeyboardInterrupt:
        print("\nStopping listener...")
    finally:
        sock.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Receive MPU6050 sensor data over UDP')
    parser.add_argument('--ip', default='0.0.0.0', help='UDP listening IP')
    parser.add_argument('--port', type=int, default=5005, help='UDP listening port')
    parser.add_argument('--window', type=int, default=100, help='Expected readings per batch')
    
    args = parser.parse_args()
    run_receiver(args.ip, args.port, args.window) 