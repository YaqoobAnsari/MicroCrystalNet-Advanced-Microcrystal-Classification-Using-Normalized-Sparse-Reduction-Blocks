# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:43:29 2024

@author: User
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from datetime import datetime 
from sklearn.manifold import TSNE  
import pandas as pd

# Define the directory containing model results
model_results_dir = '../Model Results'

# Path to the merged results CSV file
merged_results_path = os.path.join(model_results_dir, 'merged_results.csv')

def update_merged_file():
    # Load the merged DataFrame
    merged_df = pd.read_csv(merged_results_path)

    # Initialize a counter for new entries added
    new_entries_added = 0

    # Loop through each subfolder in the Model Results directory
    for subdir, dirs, files in os.walk(model_results_dir):
        for file in files:
            if file.endswith('.csv') and os.path.join(subdir, file) != merged_results_path:
                # Construct the full file path
                file_path = os.path.join(subdir, file)
                
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Check if the first row's model name is already in the merged_df
                if not df.iloc[0]['Model Name'] in merged_df['Model Name'].values:
                    # Append the new DataFrame to the merged_df
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
                    new_entries_added += 1

    # Save the updated merged DataFrame if new entries were added
    if new_entries_added > 0:
        merged_df.to_csv(merged_results_path, index=False)
        print(f"Updated the merged results file with {new_entries_added} new entries.")
    else:
        print("No new entries were added to the merged results file.")
        
def merge_results():
    # Check if the merged results file already exists
    if not os.path.exists(merged_results_path):
        # If the file does not exist, use the initial creation logic
        all_data_frames = []
    
        # Loop through each subfolder and collect CSV files as before
        for subdir, dirs, files in os.walk(model_results_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    df = pd.read_csv(file_path)
                    all_data_frames.append(df)
    
        # Concatenate all DataFrames
        merged_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(merged_results_path, index=False)
        print("Results CSV Created. Data added.")
    else:
        # If the file exists, update it with new data
        update_merged_file()
        print("Results CSV updated.")
merge_results()