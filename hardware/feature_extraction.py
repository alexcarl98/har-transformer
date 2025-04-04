import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm
π = np.pi

def zero_crossing(df, column_name):
    df[f"{column_name}_zero_crossing"] = df[column_name].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return df


def derive_periodic_features(t, period):
    ω = (2*π) / period
    return np.sin(ω*t), np.cos(ω*t)

def tensors_equal(new_data, old_data):
    for i, (new, old) in enumerate(zip(new_data, old_data)):
        if isinstance(new, torch.Tensor):
            if not torch.equal(new, old):
                # Find where they differ
                differences = (new != old)
                for row in range(differences.shape[0]):
                    for col in range(differences.shape[1]):
                        if differences[row, col]:
                            print(f"Difference in tensor {i}:")
                            print(f"Row {row}, Column {col}")
                            print(f"New value: {new[row, col]}")
                            print(f"Old value: {old[row, col]}")
                            print("---")
                return False
        elif new != old:
            print(f"Non-tensor difference in position {i}:")
            print(f"New value: {new}")
            print(f"Old value: {old}")
            return False
    return True



FILE_PATH = "https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/016_labeled.csv"
URL_BASE = "https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/"
url_suffix = "_labeled.csv"

# Define the dataset numbers you want to process
dataset_numbers = ['001', '002', '004', '008','010','011','012','013','015','016']
dataset_numbers = ['017', '018', '019', '020','021','022','024','025','026']
dataset_numbers = ['027', '028', '029', '030', '031', '032', '033', '034','035', '036', '037']
dataset_numbers = ['038', '039', '040', '041']

target = 'activity'

# Initialize dictionary to store combined data
combined_data = {}

# First pass: collect all data and see total counts per activity
for dataset_num in tqdm(dataset_numbers):
    try:
        df = pd.read_csv(f"{URL_BASE}{dataset_num}{url_suffix}")
        person_dict = {}
        min_sample_sz = float('inf')
        for activity, group in df.groupby(target):
            if activity not in person_dict:
                person_dict[activity] = []
            person_dict[activity].append(group)
            sample_num = len(group)
            if sample_num < min_sample_sz:
                min_sample_sz = sample_num
        
        min_sample_sz = (9 * min_sample_sz)//10
        half_min = min_sample_sz // 2

        combined_activities = pd.DataFrame()

        for activity, dataframe in person_dict.items():
            df = dataframe[0]
            current_activity_sz = len(df)
            if current_activity_sz > min_sample_sz:
                middle_index = current_activity_sz // 2
                start_index = max(0, middle_index - half_min)
                end_index = min(current_activity_sz, middle_index + half_min)
                selected_data = df.iloc[start_index:end_index].copy()
                person_dict[activity] = selected_data
                combined_activities = pd.concat([combined_activities, selected_data], ignore_index=True)
        
        combined_data[dataset_num] = combined_activities
            
    except Exception as e:
        print(f"Error processing dataset {dataset_num}: {str(e)}")
        continue

for person, activities in combined_data.items():
    activities.to_csv(f'HAR_DATA/{person}.csv', index=False)

exit()

# Combine all dataframes for each activity
for activity in combined_data:
    combined_data[activity] = pd.concat(combined_data[activity], ignore_index=True)

# Print original distribution
print("\nOriginal Combined Distribution:")
total_counts = pd.Series({act: len(df) for act, df in combined_data.items()})
print(pd.DataFrame({
    'Count': total_counts,
    'Percentage (%)': (total_counts / total_counts.sum() * 100).round(2)
}))

# Create balanced dataset
min_samples = min(len(df) for df in combined_data.values())
print(f"\nBalancing all classes to {min_samples} samples each")

balanced_data = {}
for activity, df in combined_data.items():
    # Randomly sample min_samples from each activity
    balanced_data[activity] = df.sample(n=min_samples, random_state=42)

# Print new distribution
print("\nBalanced Distribution:")
balanced_counts = pd.Series({act: len(df) for act, df in balanced_data.items()})
print(pd.DataFrame({
    'Count': balanced_counts,
    'Percentage (%)': (balanced_counts / balanced_counts.sum() * 100).round(2)
}))

# Combine into final balanced dataset
balanced_df = pd.concat(balanced_data.values(), ignore_index=True)
print(f"\nFinal balanced dataset shape: {balanced_df.shape}")

# Optional: shuffle the final dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# You can save the balanced dataset if needed
# balanced_df.to_csv('balanced_dataset.csv', index=False)

'''
NOTES:

import numpy as np
import pandas as pd


FILE_PATH = "https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/016_labeled.csv"
URL_BASE = "https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data/"
url_suffix = "_labeled.csv"

# Define the dataset numbers you want to process
# dataset_numbers = ['001', '002', '004', '008','010','011','012','013','015','016' ]
dataset_numbers = ['016', '002']

target = 'activity'

# Initialize dictionary to store combined data
combined_data = {}

# First pass: collect all data and see total counts per activity
for dataset_num in dataset_numbers:
    try:
        df = pd.read_csv(f"{URL_BASE}{dataset_num}{url_suffix}")
        min_sample_sz = float('inf')
        person_dict = {}
        for activity, group in df.groupby(target):
            if activity not in person_dict:
                person_dict[activity] = []
            person_dict[activity].append(group)
            # sample_num = len(group)
            # if sample_num < min_sample_sz:
            #     min_sample_sz = sample_num
        
        # half_min = min_sample_sz // 2

        # for activity, dataframe in person_dict.items():
        #     current_activity_sz = len(dataframe)
        #     if current_activity_sz > min_sample_sz:
        #         middle_index = current_activity_sz // 2
        #         start_index = max(0, middle_index - half_min)
        #         end_index = min(current_activity_sz, middle_index + half_min)
        #         selected_data = dataframe.iloc[start_index:end_index].copy()
        #         person_dict[activity] = selected_data
        
        # combined_data[dataset_num] = person_dict

    except Exception as e:
        print(f"Error processing dataset {dataset_num}: {str(e)}")
        continue

print(combined_data)
exit()

# Combine all dataframes for each activity
for activity in combined_data:
    combined_data[activity] = pd.concat(combined_data[activity], ignore_index=True)

# Print original distribution
print("\nOriginal Combined Distribution:")
total_counts = pd.Series({act: len(df) for act, df in combined_data.items()})
print(pd.DataFrame({
    'Count': total_counts,
    'Percentage (%)': (total_counts / total_counts.sum() * 100).round(2)
}))

# Create balanced dataset
min_samples = min(len(df) for df in combined_data.values())
print(f"\nBalancing all classes to {min_samples} samples each")

balanced_data = {}
for activity, df in combined_data.items():
    # Randomly sample min_samples from each activity
    balanced_data[activity] = df.sample(n=min_samples, random_state=42)

# Print new distribution
print("\nBalanced Distribution:")
balanced_counts = pd.Series({act: len(df) for act, df in balanced_data.items()})
print(pd.DataFrame({
    'Count': balanced_counts,
    'Percentage (%)': (balanced_counts / balanced_counts.sum() * 100).round(2)
}))

# Combine into final balanced dataset
balanced_df = pd.concat(balanced_data.values(), ignore_index=True)
print(f"\nFinal balanced dataset shape: {balanced_df.shape}")

# Optional: shuffle the final dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# You can save the balanced dataset if needed
# balanced_df.to_csv('balanced_dataset.csv', index=False)

'''