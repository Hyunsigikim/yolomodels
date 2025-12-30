import os
import random
import shutil
import json
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define paths
project_dir = Path('project4') # 여기를 변경
dataset_dir = Path('yolo_dataset4')

# Create output directories
for split in ['train', 'val', 'test']:
    (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

# Get all image files (assuming .jpg extension)
image_files = list(project_dir.glob('images/*.jpg'))
print(f"Found {len(image_files)} images")

# Shuffle the dataset
random.shuffle(image_files)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

total = len(image_files)
train_end = int(train_ratio * total)
val_end = train_end + int(val_ratio * total)

# Split the dataset
train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

print(f"Splitting into: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def copy_files(files, split):
    copied_labels = 0
    for img_path in files:
        # Copy image
        dest_img = dataset_dir / split / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Try different label file naming patterns
        possible_label_names = [
            f"{img_path.stem}_backup.txt",  # For Label Studio format
            f"{img_path.stem}.txt",         # Standard YOLO format
            f"{img_path.stem}.txt"           # Another common format
        ]
        
        label_copied = False
        for label_name in possible_label_names:
            label_path = project_dir / 'labels' / label_name
            if label_path.exists():
                dest_label = dataset_dir / split / 'labels' / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dest_label)
                copied_labels += 1
                label_copied = True
                break
        
        if not label_copied:
            print(f"Warning: No label file found for {img_path.name}")
    
    print(f"Copied {len(files)} images and {copied_labels} labels to {split} set")
    if copied_labels < len(files):
        print(f"Warning: Only found labels for {copied_labels} out of {len(files)} images in {split} set")

# Copy files to their respective directories
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset splitting complete!")

# Read class information from notes.json
try:
    with open(project_dir / 'notes.json', 'r') as f:
        notes = json.load(f)
    
    # Sort categories by id to ensure correct order
    categories = sorted(notes['categories'], key=lambda x: x['id'])
    class_names = {str(cat['id']): cat['name'] for cat in categories}
    
    # Generate names section for YAML
    names_section = '\n'.join([f'  {id_}: {name}' for id_, name in class_names.items()])
    
    # Create dataset.yaml file
    dataset_yaml = f"""path: {str(dataset_dir.resolve())}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(class_names)}

# Classes
names:
{names_section}
"""
    
    with open(dataset_dir / 'dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(dataset_yaml)
    
    print(f"Loaded {len(class_names)} classes from notes.json")
    
except Exception as e:
    print(f"Error reading notes.json: {e}")
    print("Using default class mapping")
    # Fallback to default classes if notes.json is not available
    dataset_yaml = f"""path: {str(dataset_dir.resolve())}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 4

# Classes
names:
  0: bike
  1: car
  2: other
  3: truck
"""
    with open(dataset_dir / 'dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(dataset_yaml)

print("Created dataset.yaml configuration file.")
print("\nDataset structure:")
print(f"{dataset_dir}")
print("├── train/")
print("│   ├── images/")
print("│   └── labels/")
print("├── val/")
print("│   ├── images/")
print("│   └── labels/")
print("├── test/")
print("│   ├── images/")
print("│   └── labels/")
print("└── dataset.yaml")
