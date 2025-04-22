import os
import random
import shutil

def split_single_folder(input_folder, output_folder, train_ratio=0.8):
    """
    Splits a single folder of images into train and test folders.
    Args:
        input_folder: Path to the folder containing images.
        output_folder: Path to save train and test folders.
        train_ratio: Ratio of images to include in the train set (default: 0.8).
    """
    # Create output directories
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all image files from the input folder
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Shuffle the images randomly

    # Calculate split index
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Move or copy images to respective folders
    for img in train_images:
        src_path = os.path.join(input_folder, img)
        dst_path = os.path.join(train_folder, img)
        shutil.copy(src_path, dst_path)

    for img in test_images:
        src_path = os.path.join(input_folder, img)
        dst_path = os.path.join(test_folder, img)
        shutil.copy(src_path, dst_path)

    print(f"Dataset split completed! Train: {len(train_images)}, Test: {len(test_images)}")

# Example usage
if __name__ == "__main__":
    input_directory = "C:/Users/rohit/OneDrive/Desktop/ADARTS project/char/char/3407"  # Replace with your folder path
    output_directory = "C:/Users/rohit/OneDrive/Desktop/ADARTS project/char/char/1_split"  # Replace with your desired output path

    split_single_folder(input_directory, output_directory)
