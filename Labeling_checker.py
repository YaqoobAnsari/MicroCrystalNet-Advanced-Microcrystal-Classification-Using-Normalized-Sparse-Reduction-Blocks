
import os
import csv
import pandas as pd


def count_csv_files(directory):
    c =0
    for root, dirs, files in os.walk(directory):
        
        csv_count = sum(1 for file in files if file.endswith('_Issac.csv'))
        if csv_count > 0:
            print(f"{c+1} Directory {root} contains {csv_count} CSV files")
            c = c+ 1

# Use the function
directory = "/path/to/directory"  # replace with your directory path
count_csv_files('Dataset')


# folder = folder.split('\\')[-1]


# specify the directory you want to start from
root_dir = 'Dataset'

subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
required_columns = ["Label No.", "Class", "Sec-class", "Ter-class", "Edges", "Corners", "Pitted", "Ledges", "Comments"]

with open('subfolders.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["File Name", "Issac Labeling", "Number of Crystals", "Required Columns", "Discarded Crystals", "Crystals Remaining"])
    
    for folder in subfolders:
        issac_labeling = any(fname.endswith('Issac.csv') for fname in os.listdir(folder))

        non_uniform_folder = os.path.join(folder, "_Non-uniform")
        if os.path.exists(non_uniform_folder):
            num_of_crystals = len([name for name in os.listdir(non_uniform_folder) if os.path.isfile(os.path.join(non_uniform_folder, name))])
        else:
            num_of_crystals = 'N/A'  # or some other value indicating lack of the "_Non-uniform" folder

        # Check if Issac.csv has required columns and count non-0 and non-1 entries in Class column
        issac_files = [fname for fname in os.listdir(folder) if fname.endswith('Issac.csv')]
        if len(issac_files) > 0:
            df = pd.read_csv(os.path.join(folder, issac_files[0]))
            has_required_columns = all(item in df.columns for item in required_columns)
            non_zero_one = df['Class'].apply(lambda x: x not in ['0', '1']).sum()

            # Print unique entries in "Sec-class" and "Ter-class"
            print(f"Unique entries in 'Sec-class' for {folder}: {df['Sec-class'].unique()}")
            print(f"Unique entries in 'Ter-class' for {folder}: {df['Ter-class'].unique()}")
        else:
            has_required_columns = 'N/A'  # or some other value indicating lack of Issac.csv
            non_zero_one = 'N/A'

        # Calculate crystals remaining
        if num_of_crystals != 'N/A' and non_zero_one != 'N/A':
            crystals_remaining = num_of_crystals - non_zero_one
        else:
            crystals_remaining = 'N/A'
            
        writer.writerow([folder, issac_labeling, num_of_crystals, has_required_columns, non_zero_one, crystals_remaining])