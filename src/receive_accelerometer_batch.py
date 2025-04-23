import socket
import struct
import numpy as np
from datetime import datetime
from collections import deque
from typing import Deque, Tuple, NamedTuple
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q
import torch

# Global sensor data queue - now holds 3 batches of 101 readings
MAX_BATCHES = 3
SENSOR_BUFFER: Deque[Tuple[float, ...]] = deque(maxlen=251)  # 3 * 251 readings

class BatchStats(NamedTuple):
    accel_mean: np.ndarray
    gyro_mean: np.ndarray
    linear_accel_mean: np.ndarray
    quaternion: np.ndarray


class UDPSensorListener:
    def __init__(self, udp_ip='0.0.0.0', udp_port=5005, window_size=101):
        """Initialize UDP Sensor Listener
        
        Args:
            udp_ip (str): IP address to listen on
            udp_port (int): Port to listen on
            window_size (int): Expected number of readings per batch (default: 101)
        """
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.window_size = window_size
        self.buffer_size = self._calculate_buffer_size()
        self.sock = None
        
        # Initialize Madgwick filter
        self.madgwick = Madgwick(sampleperiod=0.01, beta=0.02)  # 100 Hz sampling rate
        # self.quaternion = np.array([1., 0., 0., 0.])  # Initial quaternion
        self.quaternion = None
        
        # Add tensor storage for latest window
        self.latest_linear_accel_window = None
        
    def _calculate_buffer_size(self):
        """Calculate required buffer size based on window size"""
        return 4 * 6 * self.window_size  # 4 bytes/float * 6 values/reading * readings
        
    def setup_socket(self):
        """Initialize and bind UDP socket"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        
    def process_batch(self, data) -> BatchStats:
        """Process received batch data and calculate orientation and linear acceleration
        
        Args:
            data (bytes): Raw UDP data received
            
        Returns:
            BatchStats: Named tuple containing means and orientation
        """
        values = struct.unpack(f"{self.window_size * 6}f", data)
        readings = np.array(values).reshape(self.window_size, 6)
        accel_data = readings[:, :3]  # Shape: (window_size, 3)
        gyro_data = readings[:, 3:]   # Shape: (window_size, 3)
        

        self.quaternion = np.array(acc2q(accel_data[0]), dtype=np.float64)
        print("Initialized quaternion from first reading.")
        # Calculate means
        accel_mean = accel_data.mean(axis=0)
        gyro_mean = gyro_data.mean(axis=0)
        
        # Process each reading for orientation and linear acceleration
        linear_accels = []
        for i in range(1, len(readings)):
            # Update orientation using Madgwick filter
            updated_q = self.madgwick.updateIMU(
                q=self.quaternion,
                gyr=gyro_data[i], 
                acc=accel_data[i]
            )

            if updated_q is not None:
                self.quaternion = updated_q
            # Rotate acceleration to remove gravity
            # Convert quaternion to rotation matrix
            q = self.quaternion
            gravity = np.array([
                2 * (q[1]*q[3] - q[0]*q[2]),
                2 * (q[0]*q[1] + q[2]*q[3]),
                q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
            ]) * 9.81
            # Remove gravity from acceleration (in world frame)
            # gravity = np.array([0, 0, 9.81])  # gravity in world frame
            linear_accel = accel_data[i] -  gravity
            linear_accels.append(linear_accel)
            
        linear_accels = np.array(linear_accels)
        linear_accel_mean = linear_accels.mean(axis=0)
        
        # Store the latest window as a PyTorch tensor and save locally
        self.latest_linear_accel_window = torch.tensor(
            linear_accels.T,  # Transpose to get (3, window_size-1)
            dtype=torch.float32
        )
        
        # Save the tensor locally
        torch.save(self.latest_linear_accel_window, 'real_time_window.pt')
        
        # Add all readings to global buffer
        for reading in readings:
            SENSOR_BUFFER.append(tuple(reading))
        
        return BatchStats(
            accel_mean=accel_mean,
            gyro_mean=gyro_mean,
            linear_accel_mean=linear_accel_mean,
            quaternion=self.quaternion
        )
    
    def print_batch_stats(self, timestamp, stats: BatchStats):
        """Print formatted batch statistics"""
        print(f"\nTime: {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"Batch Statistics:")
        print(f"Accelerometer mean (m/s²): ({stats.accel_mean[0]:6.2f}, {stats.accel_mean[1]:6.2f}, {stats.accel_mean[2]:6.2f})")
        print(f"Gyroscope mean (°/s):     ({stats.gyro_mean[0]:6.2f}, {stats.gyro_mean[1]:6.2f}, {stats.gyro_mean[2]:6.2f})")
        print(f"Linear Accel mean (m/s²): ({stats.linear_accel_mean[0]:6.2f}, {stats.linear_accel_mean[1]:6.2f}, {stats.linear_accel_mean[2]:6.2f})")
        print(f"Current Orientation (q):   ({stats.quaternion[0]:6.2f}, {stats.quaternion[1]:6.2f}, {stats.quaternion[2]:6.2f}, {stats.quaternion[3]:6.2f})")
        print(f"Buffer size: {len(SENSOR_BUFFER)} readings")
        print("-" * 50)
    
    def get_buffer_data(self) -> np.ndarray:
        """Get all data from the buffer as a numpy array"""
        return np.array(list(SENSOR_BUFFER))
    
    def clear_buffer(self):
        """Clear the global buffer"""
        SENSOR_BUFFER.clear()
    
    def start_listening(self):
        """Start the main listening loop"""
        if not self.sock:
            self.setup_socket()
            
        print(f"Listening for accelerometer batches on port {self.udp_port}...")
        print(f"Expecting batches of {self.window_size} readings")
        print(f"Format: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] × {self.window_size}")
        print(f"Buffer capacity: {SENSOR_BUFFER.maxlen} readings")
        
        try:
            while True:
                data, addr = self.sock.recvfrom(self.buffer_size)
                timestamp = datetime.now()
                
                stats = self.process_batch(data)
                self.print_batch_stats(timestamp, stats)
                
        except KeyboardInterrupt:
            print("\nStopping listener...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        if self.sock:
            self.sock.close()
            self.sock = None

    def get_latest_window(self):
        """Return the latest linear acceleration window as a tensor."""
        return self.latest_linear_accel_window

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Receive MPU6050 sensor data over UDP')
    parser.add_argument('--ip', default='0.0.0.0', help='UDP listening IP')
    parser.add_argument('--port', type=int, default=5005, help='UDP listening port')
    parser.add_argument('--window', type=int, default=101, help='Number of readings per batch')
    
    args = parser.parse_args()
    
    listener = UDPSensorListener(args.ip, args.port, args.window)
    listener.start_listening()

if __name__ == "__main__":
    main() 