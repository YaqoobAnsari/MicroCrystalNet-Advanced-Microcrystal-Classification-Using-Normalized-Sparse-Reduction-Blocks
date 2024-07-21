import os
import shutil
import numpy as np

def split_dataset(root_dir, output_dir, train_ratio=0.75, val_ratio=0.10):
    # Ensure the output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List the classes
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.tif'))]
        
        # Shuffle the image files
        np.random.shuffle(image_files)
        
        # Split the image files
        train_end = int(train_ratio * len(image_files))
        val_end = train_end + int(val_ratio * len(image_files))
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Save the images to the new directory structure
        for dataset_type, dataset_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            dataset_dir = os.path.join(output_dir, dataset_type, class_name)
            
            # Ensure the dataset directory exists
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            for image_file in dataset_files:
                src_path = os.path.join(class_dir, image_file)
                dst_path = os.path.join(dataset_dir, image_file)
                shutil.copyfile(src_path, dst_path)

# Use the function
root_dir = "../Dataset"
output_dir = "../Split Dataset"
split_dataset(root_dir, output_dir)
