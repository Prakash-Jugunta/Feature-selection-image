import os
import shutil
import random

# Set the absolute path to your Malayalam character dataset
dataset_dir = "Telugu"  # Update this to your dataset path
train_dir = "dataset\\telugu_chars\\train"
test_dir = "dataset\\telugu_chars\\test"

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Debugging: Print the directory structure
print(f"Source directory: {dataset_dir}")
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Only process subfolders that are not 'train' or 'test'
for char_folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, char_folder)
    if os.path.isdir(folder_path) and char_folder not in ['train', 'test']:
        images = os.listdir(folder_path)
        random.shuffle(images)
        split_idx = int(0.8 * len(images))  # 80% train, 20% test
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        os.makedirs(os.path.join(train_dir, char_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, char_folder), exist_ok=True)

        for img in train_images:
            src = os.path.join(dataset_dir, char_folder, img)
            dst = os.path.join(train_dir, char_folder, img)
            print(f"Copying from {src} to {dst}")  # Debugging print
            if os.path.exists(src):  # Check if source file exists
                shutil.copy(src, dst)
            else:
                print(f"Source file not found: {src}")

        for img in test_images:
            src = os.path.join(dataset_dir, char_folder, img)
            dst = os.path.join(test_dir, char_folder, img)
            print(f"Copying from {src} to {dst}")  # Debugging print
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Source file not found: {src}")

print("Dataset split into train and test successfully!")