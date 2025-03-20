import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
base_dir = "D:/plant_disease_detection/model/dataset"
all_images_dir = os.path.join(base_dir, "all_images")  # Original dataset folder
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")

# Create train, test, and val directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split ratio
train_ratio = 0.7  # 70% for training
val_ratio = 0.15   # 15% for validation
test_ratio = 0.15  # 15% for testing

# Iterate through each class folder
for class_name in os.listdir(all_images_dir):
    class_path = os.path.join(all_images_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get all image file paths for the current class
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

    # Split into train, test, and val
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # Create class subdirectories in train, test, and val folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move images to respective folders
    for img in train_images:
        shutil.copy(img, os.path.join(train_dir, class_name))
    for img in test_images:
        shutil.copy(img, os.path.join(test_dir, class_name))
    for img in val_images:
        shutil.copy(img, os.path.join(val_dir, class_name))

    print(f"Class '{class_name}' - Train: {len(train_images)}, Test: {len(test_images)}, Val: {len(val_images)}")

print("Dataset split completed!")