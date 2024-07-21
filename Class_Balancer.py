import os
import random
from PIL import Image

def augment_by_rotation(image_path, output_dir):
    """Augment an image by orthogonal rotations and save them."""
    img = Image.open(image_path)
    angles = [90, 180, 270]

    augmented_images = []
    for angle in angles:
        rotated_img = img.rotate(angle)
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{angle}{ext}"
        rotated_img.save(os.path.join(output_dir, new_filename))
        augmented_images.append(new_filename)

    return augmented_images

def balance_classes(train_dir, margin=10):
    # Find the class with the smallest number of images
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))

    # Print initial class counts
    print("Initial class counts:", class_counts)
            
    min_class = min(class_counts, key=class_counts.get)
    min_count = class_counts[min_class]

    # Augment every image of the class with the smallest number
    print(f"Augmenting class {min_class} with orthogonal rotations...")
    min_class_dir = os.path.join(train_dir, min_class)
    for image_file in os.listdir(min_class_dir):
        image_path = os.path.join(min_class_dir, image_file)
        augment_by_rotation(image_path, min_class_dir)

    # Update and print the count for the class with the smallest number
    updated_count = len(os.listdir(min_class_dir))
    print(f"Updated count for class {min_class}: {updated_count}")

    target_count = 4 * min_count

    # Augment other classes
    for class_name, count in class_counts.items():
        if class_name == min_class:
            continue
        
        class_dir = os.path.join(train_dir, class_name)
        images = os.listdir(class_dir)
        
        # Continue augmentation until the count of images is within the margin of the target count
        while abs(len(os.listdir(class_dir)) - target_count) > margin:
            image_file = random.choice(images)  # Pick a random image to augment
            image_path = os.path.join(class_dir, image_file)
            augment_by_rotation(image_path, class_dir)

        # Print the updated count for the class
        updated_count = len(os.listdir(class_dir))
        print(f"Updated count for class {class_name}: {updated_count}")

# Use the function
train_dir = "../Split Dataset/train"
balance_classes(train_dir)

 