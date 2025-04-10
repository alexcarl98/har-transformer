# Updated implementation with toggles and parameter definitions for each heuristic

# === Configurable Flags ===
USE_ROLLING_STD = True
USE_CADENCE_VARIABILITY = True
USE_FREQ_DOMINANCE = True
NOISE_VOTE_THRESHOLD = 2  # Number of heuristics that must flag a segment as noisy

# === Parameters ===
NZ_WINDOW_SIZE = 40               # Window size for rolling std
STD_THRESHOLD_MULTIPLIER = 3      # Multiplier for rolling std threshold
PEAK_DISTANCE = 10                # Minimum distance between peaks (in samples)
CADENCE_CV_THRESHOLD = 0.3        # Coefficient of variation threshold
FREQ_DOMINANCE_PERCENTILE = 85    # Percentile for dominant frequency cutoff

# File & Sensor Info
subject_id = "019"
data_dir = "har_data"
sensor_loc = ["waist", "ankle", "wrist"]
time_col = 'time'

# Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import rfft
import os

full_path = os.path.join(data_dir, f"{subject_id}.csv")
data = pd.read_csv(full_path)
data[time_col] = pd.to_datetime(data[time_col])

# Segment by activity
activity_changes = data['activity'].ne(data['activity'].shift()).cumsum()
activity_groups = data.groupby(activity_changes)
available_sensors = [sensor for sensor in sensor_loc if f'{sensor}_x' in data.columns]
n_sensors = len(available_sensors)
unique_activities = data['activity'].unique()

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

            for axis_color, axis_name, color in zip(['x', 'y', 'z','vm'], ['X-axis', 'Y-axis', 'Z-axis', 'VM-axis'], ['red', 'green', 'blue', 'purple']):
                col = f'{sensor}_{axis_color}'
                signal_data = segment[col].values

                vote = 0

                # === Heuristic 1: Rolling Std ===
                if USE_ROLLING_STD:
                    rolling_std = pd.Series(signal_data).rolling(window=NZ_WINDOW_SIZE).std().fillna(0)
                    std_threshold = STD_THRESHOLD_MULTIPLIER * rolling_std.mean()
                    noisy_idx_std = np.where(rolling_std > std_threshold)[0]
                    if len(noisy_idx_std) > 0:
                        vote += 1

                # === Heuristic 2: Step Cadence CV ===
                if USE_CADENCE_VARIABILITY:
                    peaks, _ = find_peaks(signal_data, distance=PEAK_DISTANCE)
                    if len(peaks) > 2:
                        step_intervals = np.diff(peaks)
                        cv = np.std(step_intervals) / (np.mean(step_intervals) + 1e-6)
                        if cv > CADENCE_CV_THRESHOLD:
                            vote += 1

                # === Heuristic 3: Frequency Dominance ===
                if USE_FREQ_DOMINANCE:
                    fft_vals = np.abs(rfft(signal_data))
                    dominant_freq_strength = np.max(fft_vals)
                    if dominant_freq_strength < np.percentile(fft_vals, FREQ_DOMINANCE_PERCENTILE):
                        vote += 1

                # === Final Flag ===
                if vote >= NOISE_VOTE_THRESHOLD and USE_ROLLING_STD and len(noisy_idx_std) > 0:
                    ax.axvspan(relative_time.iloc[noisy_idx_std[0]], relative_time.iloc[noisy_idx_std[-1]], color='orange', alpha=0.2)

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
