import pandas as pd
from tqdm import tqdm
from constants import data_dir, URL_BASE, dataset_numbers, LABELS_COL

url_suffix = "_labeled.csv"
target = LABELS_COL[0]
combined_data = {}

# first-collect all data and see total counts per activity
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
        # take 90% of the smallest activity
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
    activities.to_csv(f'{data_dir}{person}.csv', index=False)