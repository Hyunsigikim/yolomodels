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

# Label Studio export uses 1-based ids: 1=car, 2=bike, 3=truck, 4=other.
# Ultralytics YOLO requires zero-based contiguous ids: 0=car, 1=bike, 2=truck, 3=other.
LABEL_ID_REMAP = {
    '1': '0',
    '2': '1',
    '3': '2',
    '4': '3',
}

DATASET_CLASS_NAMES = {
    0: 'car',
    1: 'bike',
    2: 'truck',
    3: 'other',
}

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
                remap_label_file(label_path, dest_label)
                copied_labels += 1
                label_copied = True
                break
        
        if not label_copied:
            print(f"Warning: No label file found for {img_path.name}")
    
    print(f"Copied {len(files)} images and {copied_labels} labels to {split} set")
    if copied_labels < len(files):
        print(f"Warning: Only found labels for {copied_labels} out of {len(files)} images in {split} set")

def remap_label_file(src_label, dest_label):
    remapped_lines = []
    with open(src_label, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            class_id = parts[0].lstrip('\ufeff')
            if class_id not in LABEL_ID_REMAP:
                raise ValueError(
                    f"Unsupported Label Studio class id '{class_id}' in {src_label}:{line_number}. "
                    f"Expected one of {', '.join(LABEL_ID_REMAP)}."
                )

            parts[0] = LABEL_ID_REMAP[class_id]
            remapped_lines.append(' '.join(parts))

    with open(dest_label, 'w', encoding='utf-8') as f:
        f.write('\n'.join(remapped_lines))
        if remapped_lines:
            f.write('\n')

# Copy files to their respective directories
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset splitting complete!")

# Read class information from notes.json
try:
    with open(project_dir / 'notes.json', 'r', encoding='utf-8-sig') as f:
        notes = json.load(f)
    
    class_names = DATASET_CLASS_NAMES
    
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
    
    print(f"Loaded {len(class_names)} classes using fixed Label Studio -> YOLO mapping")
    
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
  0: car
  1: bike
  2: truck
  3: other
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
