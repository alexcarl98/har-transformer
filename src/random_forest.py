import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


# Data Loading and Preparation
df = pd.read_csv('raw_data/016.csv')

# Display all the unique activites
print(df['activity'].unique())
y_labels = df['activity']

# One hot encoding:
# Using pandas get_dummies
categorical_col = ['activity']

# Or using sklearn
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(df[categorical_col])
