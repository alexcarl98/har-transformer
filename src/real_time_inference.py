import threading
import torch
import time
from queue import Queue
import model_version.v1 as v1
from utils import TConfig
from receive_accelerometer_batch import UDPSensorListener
import socket

class RealTimeInference:
    def __init__(self, model_path, config_path='config.yml'):
        # Initialize device and load config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = TConfig.from_yaml(config_path)
        
        # Initialize model
        self.stats_pp = v1.TorchStatsPipeline(self.args.extracted_features, self.args.in_seq_dim)
        self.model = self.load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize sensor listener
        self.sensor_listener = UDPSensorListener(window_size=101)
        
        # Thread control
        self.running = False
        self.inference_queue = Queue(maxsize=1)  # Only keep latest window
        
    def load_model(self, model_path):
        model = v1.AccelTransformerV1(
            d_model=self.args.d_model,
            fc_hidden_dim=self.args.fc_hidden_dim,
            num_classes=self.args.num_classes,
            in_channels=self.args.in_seq_dim,
            in_meta_dim=self.args.in_meta_dim,
            nhead=self.args.nhead,
            dropout=self.args.dropout,
            patch_size=16,
            stride=4,
            window_size=self.args.window_size,
            torch_stats_pipeline=self.stats_pp
        ).to(self.device)
        
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def sensor_thread(self):
        """Thread for receiving sensor data"""
        print("Starting sensor thread...")
        try:
            while self.running:
                data, _ = self.sensor_listener.sock.recvfrom(self.sensor_listener.buffer_size)
                stats = self.sensor_listener.process_batch(data)
                
                # Get latest window and put in queue
                if self.sensor_listener.latest_linear_accel_window is not None:
                    # Replace old window if queue is full
                    if self.inference_queue.full():
                        self.inference_queue.get()
                    self.inference_queue.put(self.sensor_listener.latest_linear_accel_window)
        except Exception as e:
            print(f"Sensor thread error: {e}")
            self.running = False
    
    @torch.no_grad()
    def inference_thread(self):
        """Thread for running model inference"""
        print("Starting inference thread...")
        try:
            while self.running:
                if not self.inference_queue.empty():
                    # Get window and run inference
                    window = self.inference_queue.get()
                    window = window.to(self.device)
                    window = window.T.unsqueeze(0)  # Reshape for model
                    
                    # Run inference
                    outputs = self.model(window)
                    prediction = torch.argmax(outputs, dim=1)
                    activity = self.args.decoder_dict[prediction.item()]
                    
                    # Print prediction
                    print(f"\nPredicted Activity: {activity}")
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
        except Exception as e:
            print(f"Inference thread error: {e}")
            self.running = False
    
    def start(self):
        """Start the real-time inference system"""
        print("Initializing real-time inference system...")
        self.running = True
        
        # Setup UDP socket
        self.sensor_listener.setup_socket()
        
        # Create and start threads
        self.sensor_thread = threading.Thread(target=self.sensor_thread)
        self.inference_thread = threading.Thread(target=self.inference_thread)
        
        self.sensor_thread.start()
        self.inference_thread.start()
        
        print("System running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping system...")
            self.stop()
    
    def stop(self):
        """Stop the system and cleanup"""
        self.running = False
        if hasattr(self, 'sensor_thread'):
            self.sensor_thread.join()
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join()
        self.sensor_listener.cleanup()
        print("System stopped.")

def main():
    model_path = '/home/alexa/Documents/har-transformer/models/run_20250422_210242/weights/best_f1_accel_transformer.pth'
    real_time_system = RealTimeInference(model_path)
    real_time_system.start()

if __name__ == "__main__":
    main() 