from ultralytics import YOLO
import argparse
from pathlib import Path
import shutil
from datetime import datetime

def _default_device():
    try:
        import torch

        return '0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return ''

def _detect_dataset_yaml():
    datasets_dir = Path('datasets')
    if not datasets_dir.exists() or not datasets_dir.is_dir():
        return None
    
    # Find all dataset.yaml files recursively
    yaml_files = list(datasets_dir.glob('**/dataset.yaml'))
    if not yaml_files:
        return None
        
    # Sort by modification time, latest first
    yaml_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return yaml_files[0]

def train_model(
    data_yaml=None,
    model_size='yolov8s.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    device=None,
    project='runs/detect',
    name='yolov8_custom',
    export_format='onnx',
    no_export=False,
    models_dir='models',
    save_best=True,
    save_best_name=None,
):
    if data_yaml is None:
        detected = _detect_dataset_yaml()
        if detected:
            data_yaml = str(detected)
            print(f"Auto-detected dataset configuration: {data_yaml}")
        else:
            raise FileNotFoundError(
                "dataset yaml not specified and none found in 'datasets/' directory. "
                "Please specify path to dataset.yaml with --data-yaml."
            )

    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")

    if device is None:
        device = _default_device()

    # If model_size is a simple filename (no directory separator), place it under models_dir
    model_size_path = Path(model_size)
    if len(model_size_path.parts) == 1:
        models_dir_path = Path(models_dir)
        models_dir_path.mkdir(parents=True, exist_ok=True)
        model_load_path = models_dir_path / model_size
    else:
        model_load_path = model_size_path

    print(f"Loading/downloading pre-trained model: {model_load_path}")
    model = YOLO(str(model_load_path))
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )

    if save_best:
        run_dir = Path(project) / name
        best_pt = run_dir / 'weights' / 'best.pt'
        last_pt = run_dir / 'weights' / 'last.pt'
        src_pt = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)
        if src_pt is not None:
            models_dir_path = Path(models_dir)
            models_dir_path.mkdir(parents=True, exist_ok=True)

            today = datetime.now().strftime('%Y%m%d')
            base = save_best_name.strip() if isinstance(save_best_name, str) and save_best_name.strip() else name
            dst = models_dir_path / f'{base}_{today}.pt'
            if dst.exists():
                i = 2
                while True:
                    candidate = models_dir_path / f'{base}_{today}_{i}.pt'
                    if not candidate.exists():
                        dst = candidate
                        break
                    i += 1
            shutil.copy2(src_pt, dst)
            print(f"Saved model: {dst}")
        else:
            print(f"Warning: weights not found in: {run_dir / 'weights'}")

    if not no_export:
        print(f"Exporting model to format: {export_format}")
        exported_path_str = model.export(format=export_format)
        if exported_path_str:
            exported_path = Path(exported_path_str)
            if exported_path.exists():
                models_dir_path = Path(models_dir)
                models_dir_path.mkdir(parents=True, exist_ok=True)
                
                ext = exported_path.suffix
                today = datetime.now().strftime('%Y%m%d')
                base = save_best_name.strip() if isinstance(save_best_name, str) and save_best_name.strip() else name
                dst_export = models_dir_path / f'{base}_{today}{ext}'
                
                if dst_export.exists():
                    i = 2
                    while True:
                        candidate = models_dir_path / f'{base}_{today}_{i}{ext}'
                        if not candidate.exists():
                            dst_export = candidate
                            break
                        i += 1
                shutil.copy2(exported_path, dst_export)
                print(f"Saved exported model: {dst_export}")

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on custom dataset')
    parser.add_argument('--data-yaml', type=str, default=None, help='Path to dataset YAML file. Defaults to auto-detecting in datasets/ directory.')
    parser.add_argument(
        '--model-size',
        type=str,
        default='yolov8s.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLOv8 model size',
    )
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help="Device: '0', 'cpu', etc. Default: auto")
    parser.add_argument('--project', type=str, default='runs/detect', help='Output project directory')
    parser.add_argument('--name', type=str, default='yolov8_custom', help='Run name')
    parser.add_argument('--export-format', type=str, default='onnx', help='Export format')
    parser.add_argument('--no-export', action='store_true', help='Disable export step')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to copy trained .pt file into')
    parser.add_argument('--no-save-best', action='store_true', help='Disable copying best.pt to models directory')
    parser.add_argument('--save-best-name', type=str, default=None, help='Override base name for saved model file')

    args = parser.parse_args()

    train_model(
        data_yaml=args.data_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        export_format=args.export_format,
        no_export=args.no_export,
        models_dir=args.models_dir,
        save_best=not args.no_save_best,
        save_best_name=args.save_best_name,
    )
