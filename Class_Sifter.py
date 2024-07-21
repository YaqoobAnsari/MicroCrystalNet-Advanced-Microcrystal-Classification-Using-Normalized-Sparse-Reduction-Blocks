import os
import pandas as pd
import shutil

 
# Define the paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
DATASET_PATH = os.path.join(BASE_PATH, '..', 'Labelled Dataset')
OUTPUT_PATH = os.path.join(BASE_PATH, '..', 'Dataset')

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def count_folders_and_process_dataset():
    all_classes = set()
    unique_classes_found = set()  # To store unique classes found
    folders_with_missing_csv = []  # To store folders with missing '_Issac.csv' files
    
    # Count and display the total number of folders in the 'Labelled Dataset' directory
    dataset_folders = os.listdir(DATASET_PATH)
    total_folders = len(dataset_folders)
    folder_count = 1
    print(f"Total number of folders in 'Labelled Dataset': {total_folders}\n")
    
    class_label_count = {}

    for folder in dataset_folders:
        print(f"Folder {folder_count}/{total_folders} :: {folder}")
        label_to_class_map = {}  # To map each label to its respective class for this specific folder
        folder_path = os.path.join(DATASET_PATH, folder)
        
        if os.path.isdir(folder_path):
            # Find the CSV file in the sub-folder
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('_Issac.csv')]

            if not csv_files:
                print(f"Warning: No '_Issac.csv' file found in folder '{folder}' \n")
                folder_count += 1
                folders_with_missing_csv.append(folder)
                continue
            
            csv_path = os.path.join(folder_path, csv_files[0])
            df = pd.read_csv(csv_path)
                    
            # Filter rows based on the 'Class' column
            df = df[df['Class'].isin(['0', '1'])]
                    
            # Change 'rounded' Sec-class values to 'spherical', strip any extra spaces
            df['Form'] = df['Form'].str.strip().str.replace('^rounded$', 'spherical', case=False).str.strip()                        
            # Map labels to classes AFTER changing the values
            for index, row in df.iterrows():
                class_name = row['Form']  # Convert to lowercase and strip whitespace
                if pd.isna(class_name) or isinstance(class_name, float):
                    
                    print(f"Warning: Invalid class name for folder '{folder}', label '{row['Label No.']}'. Skipping...")
                    folder_count += 1
                    continue
                label_to_class_map[row['Label No.']] = class_name
                unique_classes_found.add(class_name)  # Add the class name to the set
                    
            # Get unique class names
            unique_classes = df['Form'].unique()  # Convert to lowercase and strip whitespace
            all_classes.update(unique_classes)

            # Print info
            #print(f'Processing Folder: {folder}')
            print(f'Number of Labels: {len(df["Label No."])}')
            #print(f'Number of Unique Classes: {len(unique_classes)}')

            
            moved_count = 0

            # Move images to corresponding class folders
            for label_no, class_name in label_to_class_map.items():
                image_name = f'{folder}_{label_no}.tif'
                image_path = os.path.join(folder_path, '_Non-uniform', image_name)
        
                class_folder = os.path.join(OUTPUT_PATH, class_name)  # No need to strip here, as it's done earlier
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
        
                if os.path.exists(image_path):
                    shutil.copy(image_path, class_folder)
                    moved_count += 1
        
                class_label_count[class_name] = class_label_count.get(class_name, 0) + 1

            print('\nLabels detected for each class:')
            total_labels_detected = 0
            for k, v in class_label_count.items():
                total_labels_detected += v
                #print(f'Class: {k}, Labels: {v}')
            
            print(f'\nTotal Labels Detected: {total_labels_detected}')
            #print(f'Number of images matched and moved: {moved_count}') 
        folder_count += 1

    print("\nFolders without '_Issac.csv' files:")
    for folder_name in folders_with_missing_csv:
        print(folder_name)

    print("\nUnique classes found:")
    for unique_class in unique_classes_found:
        print(unique_class)

    print(f'\nProcessed all folders. Total unique classes found: {len(all_classes)}')

count_folders_and_process_dataset()



def move_and_delete_rounded_folder():
    # Define the path for the "rounded" and "spherical" folders
    rounded_folder_path = os.path.join(OUTPUT_PATH, "rounded")
    spherical_folder_path = os.path.join(OUTPUT_PATH, "spherical")

    # Check if the "rounded" folder exists
    if os.path.exists(rounded_folder_path):
        # Iterate over all files in the "rounded" folder
        for file_name in os.listdir(rounded_folder_path):
            # Define the source path (inside "rounded" folder) and destination path (inside "spherical" folder)
            source_path = os.path.join(rounded_folder_path, file_name)
            destination_path = os.path.join(spherical_folder_path, file_name)

            # Move the file
            shutil.move(source_path, destination_path)
        
        # After moving all files, delete the "rounded" folder
        os.rmdir(rounded_folder_path)
        print("\nMoved all images from 'rounded' to 'spherical' and deleted the 'rounded' folder.")

move_and_delete_rounded_folder()
