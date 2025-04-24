import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# This is where all the data is stored online
def download_data(url, data_dir):
    print("Downloading data from", url)
    URL_BASE = url

    all_subjects = ['001', '002', '004', '008','010','011','012',
                    '013','015','016','017', '018', '019', '020',
                    '021','022','024','025', '026','027', '028','029',
                    '030', '031', '032', '033', '034', '035', '036', '037', 
                    '038', '039', '040', '041']
    
    sensor_loc = ["waist", "ankle", "wrist"]
    
    classes = ["upstairs", "downstairs","jog_treadmill", "walk_treadmill", "walk_mixed", "walk_sidewalk"]
    valid_columns= ['time','wrist_x','wrist_y','wrist_z','wrist_vm','ankle_x','ankle_y','ankle_z','ankle_vm','waist_x','waist_y','waist_z','waist_vm','person','activity']

    all_subjects_int = [int(subject) for subject in all_subjects]
    label = "activity"

    # This is the biometrics file that contains the information about the subjects
    biometrics = pd.read_csv("https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/biometrics/biometrics.csv")

    # remove all subjects not in all_subject_int
    biometrics = biometrics[biometrics['ID'].isin(all_subjects_int)]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for dataset_num in all_subjects:
        try:
            df = pd.read_csv(f"{URL_BASE}/{dataset_num}_labeled.csv")

            for col in valid_columns:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' missing from dataset {dataset_num}")
            
            y = df[label]

            # Create a DataFrame with both counts and percentages
            counts = y.value_counts()
            percentages = y.value_counts(normalize=True) * 100
            
            # Add counts and percentages to biometrics
            subject_idx = biometrics['ID'] == int(dataset_num)
            
            # Add counts and percentages for each class
            for class_name in classes:
                if class_name in counts:
                    biometrics.loc[subject_idx, f'{class_name}_count'] = counts[class_name]
                    biometrics.loc[subject_idx, f'{class_name}_percent'] = percentages[class_name].round(2)
                else:
                    biometrics.loc[subject_idx, f'{class_name}_count'] = 0
                    biometrics.loc[subject_idx, f'{class_name}_percent'] = 0.0
            
            # Add total samples
            biometrics.loc[subject_idx, 'total_samples'] = counts.sum()
            
            # Add sensor information

            summary_df = pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percentages.round(2)  # Round to 2 decimal places
        })
            
            summary_df.loc['Total'] = [summary_df['Count'].sum(), 100.0]
            
            print(f"\nDataset {dataset_num} Distribution:")
            print(summary_df.to_string(), "\nSensors:")
            
            for sensor in sensor_loc:
                print(f"- {sensor}: {int(f"{sensor}_x" in df.columns)}")
                biometrics.loc[subject_idx, f'has_{sensor}'] = int(f"{sensor}_x" in df.columns)
            
            print("-" * 50)  # Separator for better readability
            
            if dataset_num == "002":
                # only applies to subject 002
                df = df.iloc[:, 1:]

            df.to_csv(f'{data_dir}{dataset_num}.csv', index=False)
        except Exception as e:
            continue


    biometrics.to_csv(f'{data_dir}000_biometrics.csv', index=False)


    # Set plot style
    sns.set(style="whitegrid")

    # Convert class-based columns to ensure no NaNs
    for cls in classes:
        biometrics[f"{cls}_count"] = biometrics[f"{cls}_count"].fillna(0)
        biometrics[f"{cls}_percent"] = biometrics[f"{cls}_percent"].fillna(0)

    # Plot 1: Total sample count per subject
    plt.figure(figsize=(12, 6))
    sns.barplot(x="ID", y="total_samples", data=biometrics)
    plt.title("Total Sample Count per Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Total Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{data_dir}subject_barplot.png')

    # Plot 2: Distribution of activity percentages across all subjects
    activity_percentages = biometrics[[f"{cls}_percent" for cls in classes]]
    activity_percentages.columns = classes
    activity_percentages = activity_percentages.melt(var_name="Activity", value_name="Percent")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Activity", y="Percent", data=activity_percentages)
    plt.title("Activity Percentage Distribution Across Subjects")
    plt.ylabel("Percent (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{data_dir}activity_percentage_boxplot.png')

    # Plot 3: Heatmap of activity coverage per subject
    heatmap_data = biometrics.set_index("ID")[[f"{cls}_percent" for cls in classes]]
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Activity Percentage Per Subject (Heatmap)")
    plt.xlabel("Activity")
    plt.ylabel("Subject ID")
    plt.tight_layout()
    # plt.show()

    plt.savefig(f'{data_dir}subject_activity_distribution.png')


if __name__ == "__main__":
    download_data("https://raw.githubusercontent.com/Har-Lab/HumanActivityData/refs/heads/main/data/labeled_activity_data", "hhhhhh/")