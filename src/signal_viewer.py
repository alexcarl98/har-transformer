# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
import seaborn as sns
import os

π = np.pi
subject_id = "002"
data_dir = "har_data"
sensor_loc = ["waist", "ankle", "wrist"]  # All possible sensors
classes = ["upstairs", "downstairs", "jog_treadmill", 
           "walk_treadmill", "walk_mixed", "walk_sidewalk"]

time_col = 'time'

# Load the data
full_path = os.path.join(data_dir, f"{subject_id}.csv")
data = pd.read_csv(full_path)

# Convert time to datetime
data[time_col] = pd.to_datetime(data[time_col])

# Find activity transitions
activity_changes = data['activity'].ne(data['activity'].shift()).cumsum()
activity_groups = data.groupby(activity_changes)

# Create subplots for each activity
available_sensors = [sensor for sensor in sensor_loc if f'{sensor}_x' in data.columns]
n_sensors = len(available_sensors)
unique_activities = data['activity'].unique()

for activity in unique_activities:
    # Get all segments for this activity
    activity_segments = [group for _, group in activity_groups if group['activity'].iloc[0] == activity]
    
    if not activity_segments:
        continue
        
    print(f"\nPlotting {activity} segments...")
    
    # Create figure for this activity
    fig, axes = plt.subplots(n_sensors, 1, figsize=(15, 3*n_sensors))
    if n_sensors == 1:
        axes = [axes]
    fig.suptitle(f'Activity: {activity}', fontsize=16)
    
    # Plot each sensor
    for ax, sensor in zip(axes, available_sensors):
        # Plot each segment
        for segment in activity_segments:
            # Calculate relative time in seconds for this segment
            relative_time = (segment[time_col] - segment[time_col].iloc[0]).dt.total_seconds()
            
            # Plot acceleration data
            ax.plot(relative_time, segment[f'{sensor}_x'], 'red', label='X-axis', linewidth=1, alpha=0.7)
            ax.plot(relative_time, segment[f'{sensor}_y'], 'green', label='Y-axis', linewidth=1, alpha=0.7)
            ax.plot(relative_time, segment[f'{sensor}_z'], 'blue', label='Z-axis', linewidth=1, alpha=0.7)
        
        ax.set_title(f'{sensor.capitalize()} Accelerometer Data')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        
        # Only add legend for the first segment to avoid duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Print statistics for this sensor and activity
        print(f"\n{sensor.capitalize()} Sensor Statistics:")
        for axis in ['x', 'y', 'z']:
            col = f'{sensor}_{axis}'
            stats = data[data['activity'] == activity][col].describe()
            print(f"\n{axis}-axis:")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Std: {stats['std']:.2f}")
            print(f"Min: {stats['min']:.2f}")
            print(f"Max: {stats['max']:.2f}")
    
    plt.tight_layout()
    plt.show()