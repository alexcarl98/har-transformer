import requests
from config import Config
from tqdm import tqdm
import os
import zipfile
import numpy as np
import pandas as pd
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2R, acc2q


def download_uci_data(filename: str):
    # raw_data_dir = config.data.raw_dir + "/uci"
    url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"  # Example URL
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        # exit()
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Save it to disk with progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

    print(f"Downloaded {filename}")


decoder_dict = [
    "other",
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "",
    "",
 	"",
 	"",
 	"",
 	"",
 	"upstairs",
 	"downstairs"
]

# to_keep = [0,1,2,3,4,5,12,13]
# There are other values above this:
# if activityID > len(decoder_dict) or decoder_dict[activityID] == "":
#     print(f"Unknown activityID: {activityID}")


sensor_locs = [
    "hand",
    "chest",
    "ankle",
]

column_names = [
    "time", # s
    "activity",
    "heart_rate", # bpm
    "hand_temp",
    "hand_acc1_x",
    "hand_acc1_y",
    "hand_acc1_z",
    "hand_acc2_x",
    "hand_acc2_y",
    "hand_acc2_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_magn_x",
    "hand_magn_y",
    "hand_magn_z",
    "hand_unk1",
    "hand_unk2",
    "hand_unk3",
    "hand_unk4",
    "chest_temp",
    "chest_acc1_x",
    "chest_acc1_y",
    "chest_acc1_z",
    "chest_acc2_x",
    "chest_acc2_y",
    "chest_acc2_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_magn_x",
    "chest_magn_y",
    "chest_magn_z",
    "chest_unk1",
    "chest_unk2",
    "chest_unk3",
    "chest_unk4",
    "ankle_temp",
    "ankle_acc1_x",
    "ankle_acc1_y",
    "ankle_acc1_z",
    "ankle_acc2_x",
    "ankle_acc2_y",
    "ankle_acc2_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_magn_x",
    "ankle_magn_y",
    "ankle_magn_z",
    "ankle_unk1",
    "ankle_unk2",
    "ankle_unk3",
    "ankle_unk4",
]

"""
timestamp activityID heart_rate 
8.38 0 104 30 2.37223 8.60074 3.51048 2.43954 8.76165 3.35465 -0.0922174 0.0568115 -0.0158445 14.6806 -69.2128 -5.58905 1 0 0 0 31.8125 0.23808 9.80003 -1.68896 0.265304 9.81549 -1.41344 -0.00506495 -0.00678097 -0.00566295 0.47196 -51.0499 43.2903 1 0 0 0 30.3125 9.65918 -1.65569 -0.0997967 9.64689 -1.55576 0.310404 0.00830026 0.00925038 -0.0175803 -61.1888 -38.9599 -58.1438 1 0 0 0
"""
def download_extract_uci_data():
    name = "PAMAP2_Dataset"
    zipped_data_name = f"{name}.zip"
    config = Config.from_yaml("config.yml")
    uci_dir = config.data.raw_dir + "/uci"
    if os.path.exists(uci_dir + "/" + name):
        print(f"Data {name} already exists. Skipping download.")
        return uci_dir + "/" + name
    
    os.makedirs(uci_dir, exist_ok=True)
    filename = uci_dir + "/" + 'pamap2.zip'
    download_uci_data(filename)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(uci_dir)  # Extracts into a folder

    # There's a zip within a zip here,
    with zipfile.ZipFile(uci_dir + "/" + zipped_data_name, 'r') as zip_ref:
        zip_ref.extractall(uci_dir)  # Extracts into a folder

    # Remove the inner zip file
    os.remove(filename)
    os.remove(uci_dir + "/" + zipped_data_name)

    return uci_dir + "/" + name


def read_first_line_as_dataframe(filepath):
    with open(filepath, 'r') as file:
        # Read the first line
        first_line = file.readline().strip()
    
    # Split the line on spaces
    entries = first_line.split()
    
    # Convert entries to floats (automatically handling 'NaN' because float('NaN') works)
    entries = [float(e) for e in entries]
    
    # Create a DataFrame (one row)
    df = pd.DataFrame([entries])
    
    return df

import matplotlib.pyplot as plt


def plot_accel_window_with_spectrogram(window, activity,axes_labels = ['x', 'y', 'z'], sampling_rate=100):
    """
    Plots waveform and spectrogram for a single accelerometer window.

    Parameters:
        window (np.ndarray): Shape (window_size, 3), where each column is x, y, z axis.
        axes_labels (list): Labels for the axes. Default is ['x', 'y', 'z'].
        sampling_rate (int): Samples per second. Default is 100Hz.
    """
    n = len(axes_labels)
    
    window_size = window.shape[0]
    time_axis = np.arange(window_size) / sampling_rate

    fig, axs = plt.subplots(2, n, figsize=(5*n, 2*n))
    fig.suptitle(f"Waveforms and Spectrograms for Accelerometer Axes for {activity}", fontsize=16)

    for i in range(n):
        signal = window[:, i]

        # Waveform
        axs[0, i].plot(time_axis, signal)
        axs[0, i].set_title(f"Waveform - {axes_labels[i]} axis")
        axs[0, i].set_xlabel("Time [s]")
        axs[0, i].set_ylabel("Amplitude")

        # Spectrogram
        axs[1, i].specgram(signal, Fs=sampling_rate, cmap='viridis')
        axs[1, i].set_title(f"Spectrogram - {axes_labels[i]} axis")
        axs[1, i].set_xlabel("Time [s]")
        axs[1, i].set_ylabel("Frequency [Hz]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f"{activity}.png")
    plt.show()

def estimate_frequency(data, time_column):
    """
    Estimate the frequency of the accelerometer data.
    """
    # Calculate the time difference between consecutive samples
    
    data[time_column] = data[time_column].astype(float)
    n = len(data[time_column])

    # Convert the time differences to seconds
    time_diff = np.diff(data[time_column].values)  # Time differences between consecutive rows

    # Calculate the sampling rate
    sampling_rate = (n-1)/np.sum(time_diff)

    return sampling_rate
    

def demo_activity_existing_data():
    subject_number = '001'
    data_path = f"/home/alexa/Documents/har-transformer/har_data/{subject_number}.csv"
    df = pd.read_csv(data_path)
    current_activity = df['activity'].iloc[0]
    activity_window = []  # Store samples for current activity
    sensor_loc = "wrist"
    feature_cols = [f'{sensor_loc}_{ft}' for ft in ["x","y","z"]]
    
    # Load and sort data
    df = pd.read_csv(data_path, parse_dates=['time']).sort_values('time')
    
    # Convert datetime to float (seconds since first timestamp)
    df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    
    df['class_change'] = (df['activity'] != df['activity'].shift()).cumsum()
    # Process each activity segment
    for _, group in df.groupby('class_change'):
        activity = group['activity'].iloc[0]
        features = group[feature_cols].values
        frequency = estimate_frequency(group, 'time')
        
        plot_accel_window_with_spectrogram(features, f"{subject_number}_{activity}_freq={frequency:.1f}", axes_labels=feature_cols, sampling_rate=100)


def calculate_linear_acceleration_windows(df, acc1, gyr, mag, frequency=100, max_time_gap=0.1):
    """
    Calculate linear acceleration windows from raw sensor data.
    
    Args:
        df (pd.DataFrame): DataFrame containing sensor data
        frequency (float): Sampling frequency in Hz
        max_time_gap (float): Maximum allowed time gap between samples in seconds
    
    Returns:
        tuple: (windows, jump_timestamps) where:
            - windows: List of windows, each window is array of shape (window_size, 4)
                      where columns are [timestamp, lin_acc_x, lin_acc_y, lin_acc_z]
            - jump_timestamps: List of timestamps where time jumps or activity changes occurred
    """
    # Setup filter
    madgwick = Madgwick(frequency=frequency)
    df = df.dropna()

    windows = []
    current_window = []
    jump_timestamps = []
    current_activity = df['activity'].iloc[0]
    last_time = df['time'].iloc[0]
    
    # Initialize first quaternion
    q = np.array(acc2q(acc1[0]), dtype=np.float64)

    for i in range(1, len(df)):
        current_time = df['time'].iloc[i]
        time_delta = current_time - last_time
        
        acc_sample = np.array(acc1[i])
        gyr_sample = np.array(gyr[i])
        mag_sample = np.array(mag[i])
        
        # Check for window breaks
        if time_delta > max_time_gap or df['activity'].iloc[i] != current_activity:
            if len(current_window) > 0:
                windows.append(np.array(current_window))
            
            # Record the timestamp where the jump occurred
            jump_timestamps.append({
                'timestamp': current_time,
                'reason': 'time_gap' if time_delta > max_time_gap else 'activity_change',
                'gap_size': time_delta if time_delta > max_time_gap else None,
                'old_activity': current_activity,
                'new_activity': df['activity'].iloc[i]
            })
            
            # Reset for new window
            q = np.array(acc2q(acc_sample), dtype=np.float64)
            current_activity = df['activity'].iloc[i]
            current_window = []

        # Update orientation
        q = madgwick.updateIMU(q, gyr=gyr_sample, acc=acc_sample) if mag_sample is None else madgwick.updateMARG(q, gyr=gyr_sample, acc=acc_sample, mag=mag_sample)
        
        if q is None:
            continue
            
        # Calculate gravity in sensor frame
        gravity = np.array([
            2 * (q[1]*q[3] - q[0]*q[2]),
            2 * (q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ]) * 9.81
        
        # Calculate linear acceleration
        lin_acc_sample = acc_sample - gravity
        
        # Store [timestamp, lin_acc_x, lin_acc_y, lin_acc_z]
        current_window.append(np.array([current_time, *lin_acc_sample]))
        last_time = current_time

    # Add final window
    if len(current_window) > 0:
        windows.append(np.array(current_window))
    
    return windows, jump_timestamps

if __name__ == "__main__":
    # demo_activity_existing_data()
    # exit()
    config = Config.from_yaml("config.yml")
    data_raw_path = config.data.raw_dir + "/uci"
    sensor_wise_columns_to_drop = ['temp', 'unk1', 'unk2', 'unk3', 'unk4']
    drop_columns = ['heart_rate'] + [f"{sensor}_{col}" for sensor in sensor_locs for col in sensor_wise_columns_to_drop]
    data_path = download_extract_uci_data()
    # print(f"{data_path=}")
    # print(f"{data_raw_path=}")
    print(f"Data extracted to {data_path}")
    allowed_activities = [4,5,12,13]
    # allowed_activities = [0,1,2,3,4,5,12,13]
    
    all_data = []
    files = os.listdir(data_path + "/Protocol")
    
    for i, file in enumerate(files):
        subject_number = files[i][7:10]
        print(f"{subject_number=}")
        df = pd.read_csv(data_path + "/Protocol/" + file, 
                        sep=' ', 
                        header=None,
                        names=column_names)
        df = df.drop(columns=drop_columns)
        df = df[df['activity'].isin(allowed_activities)]
        df = df.dropna()
        output_csv = f"{data_raw_path}/{subject_number}.csv"
        df.to_csv(output_csv, index=False)

    first_activity = df['activity'].iloc[0]
    current_activity = first_activity

    # Setup filter
    frequency=100
    sensor_locs = [
        "hand",
        "chest",
        "ankle",
    ]
    clsmap = {
        4: 'walk_sidewalk',
        5: 'jogging_treadmill',
        12: 'upstairs',
        13: 'downstairs',
    }

    sensor_lin_acc_windows = []
    mpsn = {
        'hand': 'wrist',
        'chest': 'waist',
        'ankle': 'ankle',
    }
    # we are going to look past it for now
    subjects = list(range(101, 110))

    for subject in subjects:
        subject_df = pd.read_csv(f"{data_raw_path}/{subject}.csv")
        if len(subject_df) == 0:
            continue
        new_df = subject_df[['time', 'activity']]
        for sensor in sensor_locs:
            acc_cols = [f'{sensor}_acc1_x', f'{sensor}_acc1_y', f'{sensor}_acc1_z']
            gyr_cols = [f'{sensor}_gyro_x', f'{sensor}_gyro_y', f'{sensor}_gyro_z']
            mag_cols = [f'{sensor}_magn_x', f'{sensor}_magn_y', f'{sensor}_magn_z']
            acc1 = subject_df[acc_cols].values
            gyr = subject_df[gyr_cols].values
            mag = subject_df[mag_cols].values
            
            # Get windows of linear acceleration data
            windows, jump_timestamps = calculate_linear_acceleration_windows(subject_df, acc1, gyr, mag, frequency=frequency)
            print(f"{len(windows)=}")
            print(f"{jump_timestamps=}")
            # Concatenate all windows into a single array
            all_data = np.concatenate(windows, axis=0)  # Shape: (total_samples, 4)

            # Create the new columns using the timestamps to align data
            new_df.loc[new_df['time'].isin(all_data[:, 0]), [f'{mpsn[sensor]}_x', f'{mpsn[sensor]}_y', f'{mpsn[sensor]}_z']] = all_data[:, 1:]
            
        print(f"{subject=}")
        new_df.dropna(inplace=True)
        new_df['activity'] = new_df['activity'].map(clsmap)
        print(f"{new_df.head()}")
        new_df.to_csv(f"{config.data.raw_dir}/uci/{subject}.csv", index=False)

        # for activity in activities:
        #     activity_distribution = subject_df[subject_df['activity'] == activity]['activity'].value_counts()
        #     count = int(activity_distribution.sum())
        #     percentage = float(count / total_samples)
        #     a = clsmap[activity]
        #     header += f"{a}_count,{a}_percentage,"
        #     row += f"{count},{percentage:.2f},"
        # header += "total_samples,has_waist,has_ankle,has_wrist"
        # row += f"{total_samples},1.0,1.0,1.0"

    # exit()


    # for sensor in sensor_locs:
    #     acc_cols = [f'{sensor}_acc1_x', f'{sensor}_acc1_y', f'{sensor}_acc1_z']
    #     gyr_cols = [f'{sensor}_gyro_x', f'{sensor}_gyro_y', f'{sensor}_gyro_z']
    #     mag_cols = [f'{sensor}_magn_x', f'{sensor}_magn_y', f'{sensor}_magn_z']
    #     acc1 = df[acc_cols].values
    #     gyr = df[gyr_cols].values
    #     mag = df[mag_cols].values
        
    #     # Get windows of linear acceleration data
    #     windows = calculate_linear_acceleration_windows(df, acc1, gyr, mag, frequency=frequency)
        
    #     # Concatenate all windows into a single array
    #     all_data = np.concatenate(windows, axis=0)  # Shape: (total_samples, 4)

        
    #     # Create the new columns using the timestamps to align data
    #     new_df.loc[new_df['time'].isin(all_data[:, 0]), [f'{mpsn[sensor]}_x', f'{mpsn[sensor]}_y', f'{mpsn[sensor]}_z']] = all_data[:, 1:]

    exit()
    gyr = df.iloc[:, 8:11].values   # hand_gyro_x to hand_gyro_z
    mag = df.iloc[:, 11:14].values  # hand_magn_x to hand_magn_z
    acc1 = acc[:, 3:]  # columns hand_acc1_x, hand_acc1_y, hand_acc1_z


    # Get windows of linear acceleration data
    windows, jump_timestamps = calculate_linear_acceleration_windows(df, acc1, gyr, mag, frequency=frequency)
    
    # Plot each window
    for i, window in enumerate(windows):
        print(f"Window {i}: shape={window.shape}")
        if window.shape[0] > 100:
            plot_accel_window_with_spectrogram(
                window[:, 1:],  # exclude timestamp column for plotting
                activity=decoder_dict[current_activity],
                axes_labels=['Linear Acc X', 'Linear Acc Y', 'Linear Acc Z'],
                sampling_rate=frequency
            )


'''
timestamp (s)
activityID (see below for the mapping to the activities)
heart rate (bpm)

IMU hand
IMU chest
IMU ankle

'''