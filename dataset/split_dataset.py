# D:/plant_disease_detection/model/dataset/split_dataset.py

import os
import shutil
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure(base_dir):
    """Create the necessary directory structure."""
    directories = ['train', 'test', 'val']
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def get_image_files(directory):
    """Get all image files from a directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(valid_extensions)]

def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split the dataset into train, test, and validation sets."""

    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Train, validation, and test ratios must sum to 1")

    # Create directory structure
    create_directory_structure(base_dir)

    # Source directory (where your original images are)
    source_dir = os.path.join(base_dir, "all_images")

    # If source directory doesn't exist, try to find images directly in the base_dir
    if not os.path.exists(source_dir):
        logger.warning(f"'all_images' directory not found in {base_dir}")
        source_dir = base_dir

    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir)
                 if os.path.isdir(os.path.join(source_dir, d))]

    if not class_dirs:
        logger.error(f"No class directories found in {source_dir}")
        return

    total_images = 0
    for class_name in class_dirs:
        class_path = os.path.join(source_dir, class_name)

        # Get all images in the class directory
        images = [os.path.join(class_path, img)
                 for img in get_image_files(class_path)]

        if not images:
            logger.warning(f"No images found in class directory: {class_name}")
            continue

        total_images += len(images)
        logger.info(f"Processing class '{class_name}' with {len(images)} images")

        # Create class directories in train, test, and val
        for split in ['train', 'test', 'val']:
            os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

        try:
            # First split: train and temp (test + val)
            train_images, temp_images = train_test_split(
                images,
                train_size=train_ratio,
                random_state=42
            )

            # Second split: test and val from temp
            val_images, test_images = train_test_split(
                temp_images,
                train_size=val_ratio/(val_ratio + test_ratio),
                random_state=42
            )

            # Copy images to respective directories
            for img in train_images:
                shutil.copy2(img, os.path.join(base_dir, 'train', class_name))
            for img in test_images:
                shutil.copy2(img, os.path.join(base_dir, 'test', class_name))
            for img in val_images:
                shutil.copy2(img, os.path.join(base_dir, 'val', class_name))

            logger.info(f"Class '{class_name}' split complete:")
            logger.info(f"  Train: {len(train_images)}")
            logger.info(f"  Test: {len(test_images)}")
            logger.info(f"  Validation: {len(val_images)}")

        except Exception as e:
            logger.error(f"Error processing class '{class_name}': {str(e)}")
            continue

    if total_images == 0:
        logger.error("No images were found to process!")
    else:
        logger.info(f"Dataset split completed! Total images processed: {total_images}")

if __name__ == "__main__":
    # Base directory where your dataset is located
    base_dir = "D:/plant_disease_detection/model/dataset"

    try:
        split_dataset(base_dir)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise