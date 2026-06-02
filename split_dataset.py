import os
import random
import shutil
import json
import argparse
from pathlib import Path

def detect_input_directory():
    exports_dir = Path('exports')
    if not exports_dir.exists() or not exports_dir.is_dir():
        return None
    
    # Find all subdirectories under exports/
    subdirs = [d for d in exports_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Check if a subdirectory contains 'images' directory
    valid_subdirs = []
    for sd in subdirs:
        if (sd / 'images').exists() and (sd / 'images').is_dir():
            valid_subdirs.append(sd)
            
    if valid_subdirs:
        # Sort by modification time, latest first
        valid_subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return valid_subdirs[0]
        
    # Check if exports/ itself has images/
    if (exports_dir / 'images').exists() and (exports_dir / 'images').is_dir():
        return exports_dir
        
    return None

def detect_classes_and_mapping(input_dir):
    # Scan label files first to see what class IDs are actually present
    label_file_ids = set()
    labels_dir = input_dir / 'labels'
    if labels_dir.exists():
        for lf in labels_dir.glob('*.txt'):
            try:
                with open(lf, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id_str = parts[0].lstrip('\ufeff')
                            label_file_ids.add(class_id_str)
            except Exception:
                pass

    # Try notes.json first
    notes_path = input_dir / 'notes.json'
    if notes_path.exists():
        try:
            with open(notes_path, 'r', encoding='utf-8-sig') as f:
                notes = json.load(f)
            
            # Check for categories (common in label-studio-converter YOLO)
            if isinstance(notes, dict) and 'categories' in notes:
                categories = notes['categories']
                categories = sorted(categories, key=lambda x: x.get('id', 0))
                
                class_names = {}
                label_id_remap = {}
                
                # Check if label files use 0-based indexing (e.g. contains '0' or max index < len(categories))
                has_zero_in_labels = '0' in label_file_ids
                
                # If there are no label files or they don't contain '0', check if category IDs themselves are 0-based
                cat_ids = [cat.get('id') for cat in categories]
                categories_are_zero_based = 0 in cat_ids if cat_ids else True
                
                use_zero_based = has_zero_in_labels or (not label_file_ids and categories_are_zero_based)
                
                if use_zero_based:
                    print("Label files are 0-indexed. Mapping categories sequentially starting from 0.")
                    for idx, cat in enumerate(categories):
                        label_id_remap[str(idx)] = str(idx)
                        class_names[idx] = cat.get('name', f"class_{idx}")
                else:
                    print("Label files are 1-indexed (or custom-indexed). Remapping declared IDs to 0-based indices.")
                    for idx, cat in enumerate(categories):
                        cat_id = str(cat.get('id'))
                        label_id_remap[cat_id] = str(idx)
                        class_names[idx] = cat.get('name', f"class_{idx}")
                
                print(f"Detected class mapping from notes.json (categories): {class_names}")
                return label_id_remap, class_names
            
            # Check for classes (list of strings)
            if isinstance(notes, dict) and 'classes' in notes and isinstance(notes['classes'], list):
                class_names = {idx: name for idx, name in enumerate(notes['classes'])}
                print(f"Detected class names from notes.json (classes list): {class_names}")
                
                has_zero = '0' in label_file_ids
                num_classes = len(class_names)
                has_num_classes = str(num_classes) in label_file_ids
                
                if has_num_classes and not has_zero:
                    print("Label files are 1-indexed. Remapping 1..N to 0..N-1.")
                    label_id_remap = {str(i): str(i-1) for i in range(1, num_classes + 1)}
                else:
                    print("Label files are 0-indexed. Keeping 0..N-1.")
                    label_id_remap = {str(i): str(i) for i in range(num_classes)}
                    
                return label_id_remap, class_names
        except Exception as e:
            print(f"Warning: Failed to parse notes.json: {e}")

    # Try classes.txt
    classes_txt_path = input_dir / 'classes.txt'
    if classes_txt_path.exists():
        try:
            with open(classes_txt_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            class_names = {idx: name for idx, name in enumerate(lines)}
            print(f"Detected class names from classes.txt: {class_names}")
            
            has_zero = '0' in label_file_ids
            num_classes = len(class_names)
            has_num_classes = str(num_classes) in label_file_ids
            
            if has_num_classes and not has_zero:
                print("Label files are 1-indexed. Remapping 1..N to 0..N-1.")
                label_id_remap = {str(i): str(i-1) for i in range(1, num_classes + 1)}
            else:
                print("Label files are 0-indexed. Keeping 0..N-1.")
                label_id_remap = {str(i): str(i) for i in range(num_classes)}
                
            return label_id_remap, class_names
        except Exception as e:
            print(f"Warning: Failed to parse classes.txt: {e}")

    # Fallback: Scan label files directly
    print("No notes.json or classes.txt found. Scanning label files to detect classes...")
    if not label_file_ids:
        print("Warning: No annotations found in label files. Defaulting to single class 'object'.")
        return {'0': '0'}, {0: 'object'}
        
    try:
        sorted_ids = sorted(list(label_file_ids), key=lambda x: int(x))
    except ValueError:
        sorted_ids = sorted(list(label_file_ids))
        
    label_id_remap = {}
    class_names = {}
    for idx, orig_id in enumerate(sorted_ids):
        label_id_remap[str(orig_id)] = str(idx)
        class_names[idx] = f"class_{orig_id}"
        
    print(f"Scanned unique class IDs: {sorted_ids}. Mapping to: {class_names}")
    return label_id_remap, class_names

def remap_label_file(src_label, dest_label, label_id_remap):
    remapped_lines = []
    with open(src_label, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            class_id = parts[0].lstrip('\ufeff')
            if class_id not in label_id_remap:
                raise ValueError(
                    f"Unsupported class id '{class_id}' in {src_label}:{line_number}. "
                    f"Expected one of {', '.join(label_id_remap)}."
                )

            parts[0] = label_id_remap[class_id]
            remapped_lines.append(' '.join(parts))

    with open(dest_label, 'w', encoding='utf-8') as f:
        f.write('\n'.join(remapped_lines))
        if remapped_lines:
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Split image/label dataset into train, val, and test sets for YOLO.")
    parser.add_argument('--input', '-i', type=str, default=None,
                        help="Input project directory containing 'images' and 'labels'. Defaults to the latest subfolder under 'exports/'.")
    parser.add_argument('--output', '-o', type=str, default='datasets/yolo_dataset',
                        help="Output directory for the processed YOLO dataset. Defaults to 'datasets/yolo_dataset'.")
    parser.add_argument('--train-ratio', type=float, default=0.7, help="Ratio of data for training (default: 0.7)")
    parser.add_argument('--val-ratio', type=float, default=0.15, help="Ratio of data for validation (default: 0.15)")
    parser.add_argument('--test-ratio', type=float, default=0.15, help="Ratio of data for testing (default: 0.15)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--exts', type=str, default='jpg,jpeg,png', help="Comma-separated image extensions to search (default: 'jpg,jpeg,png')")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Resolve input directory
    if args.input:
        project_dir = Path(args.input)
    else:
        detected = detect_input_directory()
        if detected:
            project_dir = detected
            print(f"Auto-detected input directory: {project_dir}")
        else:
            print("Error: No valid project directory found in 'exports/'.")
            print("Please place your Label Studio export inside 'exports/' or specify --input.")
            return
            
    if not project_dir.exists():
        print(f"Error: Input directory does not exist: {project_dir}")
        return
        
    dataset_dir = Path(args.output)
    
    # Verify split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        print(f"Error: Split ratios sum to {total_ratio:.2f}, but must equal 1.0")
        return

    # Detect classes and construct label remap
    label_id_remap, class_names = detect_classes_and_mapping(project_dir)
    if not class_names:
        print("Error: No classes could be resolved. Cannot construct dataset.")
        return
        
    print(f"Active Label Remap: {label_id_remap}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Get all image files matching the extensions
    exts = [e.strip().lower() for e in args.exts.split(',')]
    image_files = []
    for ext in exts:
        image_files.extend(list(project_dir.glob(f'images/*.{ext}')))
        image_files.extend(list(project_dir.glob(f'images/*.{ext.upper()}')))
        
    print(f"Found {len(image_files)} image files")
    if not image_files:
        print("Error: No images found. Exiting.")
        return

    # Shuffle the dataset
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(args.train_ratio * total)
    val_end = train_end + int(args.val_ratio * total)

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
            ]
            
            label_copied = False
            for label_name in possible_label_names:
                label_path = project_dir / 'labels' / label_name
                if label_path.exists():
                    dest_label = dataset_dir / split / 'labels' / f"{img_path.stem}.txt"
                    try:
                        remap_label_file(label_path, dest_label, label_id_remap)
                        copied_labels += 1
                        label_copied = True
                        break
                    except Exception as e:
                        print(f"Error processing label file {label_path}: {e}")
            
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

    # Generate dataset.yaml file
    names_section = '\n'.join([f'  {id_}: {name}' for id_, name in class_names.items()])
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

    print(f"Created dataset.yaml configuration file at {dataset_dir / 'dataset.yaml'}")
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

if __name__ == '__main__':
    main()
