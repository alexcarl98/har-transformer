import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

π = np.pi
NZ_WINDOW_SIZE = 20
THRESHOLD_MULTIPLIER = 3
subject_id = "040"
data_dir = "har_data"
sensor_loc = ["waist", "ankle", "wrist"]
classes = ["upstairs", "downstairs", "jog_treadmill", 
           "walk_treadmill", "walk_mixed", "walk_sidewalk"]

time_col = 'time'

# Load the data
full_path = os.path.join(data_dir, f"{subject_id}.csv")
data = pd.read_csv(full_path)
data[time_col] = pd.to_datetime(data[time_col])

# Find activity transitions
activity_changes = data['activity'].ne(data['activity'].shift()).cumsum()
activity_groups = data.groupby(activity_changes)

# Identify available sensors
available_sensors = [sensor for sensor in sensor_loc if f'{sensor}_x' in data.columns]
n_sensors = len(available_sensors)
unique_activities = data['activity'].unique()

# Prepare to store plots
for activity in unique_activities:
    activity_segments = [group for _, group in activity_groups if group['activity'].iloc[0] == activity]
    if not activity_segments:
        continue

    fig, axes = plt.subplots(n_sensors, 1, figsize=(15, 3 * n_sensors))
    if n_sensors == 1:
        axes = [axes]
    fig.suptitle(f'Activity: {activity}', fontsize=16)

    for ax, sensor in zip(axes, available_sensors):
        for segment in activity_segments:
            relative_time = (segment[time_col] - segment[time_col].iloc[0]).dt.total_seconds()
            noisy_sections = []

            for axis_color, axis_name, color in zip(['x', 'y', 'z'], ['X-axis', 'Y-axis', 'Z-axis'], ['red', 'green', 'blue']):
                col = f'{sensor}_{axis_color}'
                signal_data = segment[col].values

                # Calculate rolling std deviation
                rolling_std = pd.Series(signal_data).rolling(window=NZ_WINDOW_SIZE).std().fillna(0)

                # Mark noisy regions based on a threshold
                threshold = THRESHOLD_MULTIPLIER * rolling_std.mean()
                noisy_idx = np.where(rolling_std > threshold)[0]
                noisy_times = relative_time.iloc[noisy_idx] if len(noisy_idx) > 0 else []

                if len(noisy_times) > 0:
                    ax.axvspan(relative_time.iloc[noisy_idx[0]], relative_time.iloc[noisy_idx[-1]], color='orange', alpha=0.2)

                ax.plot(relative_time, signal_data, color=color, label=axis_name, linewidth=1, alpha=0.7)

        ax.set_title(f'{sensor.capitalize()} Accelerometer Data')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.show()
