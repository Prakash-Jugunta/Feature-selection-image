# Check if files exist in class_0 folder
import os


train_files = os.listdir("C:/Users/rohit/Downloads/ADARTS project/char/char/1_split/train/class_0")
print(f"Found {len(train_files)} training images")

test_files = os.listdir("C:/Users/rohit/Downloads/ADARTS project/char/char/1_split/test/class_0")
print(f"Found {len(test_files)} testing images")
