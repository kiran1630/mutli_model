import os
import random
import shutil

# Paths
source_dir = r"D:\download\EuroSATd"  # Main dataset directory
output_dir = r"D:\download\euro1"  # Directory where train/test will be saved

train_ratio = 0.8  # 80% training, 20% testing
images_per_class = 2000  # Select 650 random images per class

# Create train and test directories
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Process each category
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    
    if os.path.isdir(category_path):
        images = os.listdir(category_path)
        selected_images = random.sample(images, images_per_class)  # Select 650 images

        # Split into train & test
        train_count = int(train_ratio * images_per_class)
        train_images = selected_images[:train_count]
        test_images = selected_images[train_count:]

        # Create category subdirectories
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))

        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

print("Data successfully split into train and test directories.")
