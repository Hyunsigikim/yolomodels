from ultralytics import YOLO
import argparse
from pathlib import Path
import shutil
import json
import yaml
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

def _read_classes_from_yaml(yaml_path: Path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data and 'names' in data:
                return data['names']
    except Exception:
        pass
    return {}

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
    resume=False,
    simplify=True,
    dynamic=False,
    opset=None,
):
    """
    Train YOLOv8 model, export to ONNX (with simplify/dynamic/opset options),
    and save final weights with metadata JSON.
    """
    run_dir = Path(project) / name
    last_pt = run_dir / 'weights' / 'last.pt'

    if resume:
        resume_target = last_pt if last_pt.exists() else Path(model_size)
        if not resume_target.exists():
            raise FileNotFoundError(f"Resume requested but checkpoint not found: {resume_target}")
        print(f"🔄 Resuming training from checkpoint: {resume_target}")
        model = YOLO(str(resume_target))
        results = model.train(resume=True)
    else:
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
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
        )

    today = datetime.now().strftime('%Y%m%d')
    base = save_best_name.strip() if isinstance(save_best_name, str) and save_best_name.strip() else name
    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)

    saved_pt_path = None
    saved_onnx_path = None

    if save_best:
        best_pt = run_dir / 'weights' / 'best.pt'
        src_pt = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)
        if src_pt is not None:
            dst = models_dir_path / f'{base}_{today}.pt'
            shutil.copy2(src_pt, dst)
            saved_pt_path = dst
            print(f"💾 Saved PyTorch model: {dst}")
        else:
            print(f"Warning: weights not found in: {run_dir / 'weights'}")

    if not no_export:
        print(f"📦 Exporting model to format: {export_format} (simplify={simplify}, dynamic={dynamic}, opset={opset})")
        export_kwargs = {
            'format': export_format,
            'simplify': simplify,
            'dynamic': dynamic,
        }
        if opset:
            export_kwargs['opset'] = opset

        exported_path_str = model.export(**export_kwargs)
        if exported_path_str:
            exported_path = Path(exported_path_str)
            if exported_path.exists():
                ext = exported_path.suffix
                dst_export = models_dir_path / f'{base}_{today}{ext}'
                shutil.copy2(exported_path, dst_export)
                saved_onnx_path = dst_export
                print(f"💾 Saved exported model: {dst_export}")

    # Generate metadata JSON
    yaml_target = Path(data_yaml) if data_yaml else _detect_dataset_yaml()
    classes_dict = _read_classes_from_yaml(yaml_target) if yaml_target else {}
    
    metadata = {
        "project_name": base,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_architecture": "YOLOv8",
        "base_model": model_size,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "classes": classes_dict,
        "pt_file": saved_pt_path.name if saved_pt_path else None,
        "onnx_file": saved_onnx_path.name if saved_onnx_path else None,
    }

    meta_json_path = models_dir_path / f'{base}_{today}_meta.json'
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"📋 Saved deployment metadata: {meta_json_path}")

    print("🎉 Training and export complete!")
    return {
        "pt_path": saved_pt_path,
        "onnx_path": saved_onnx_path,
        "meta_path": meta_json_path,
        "classes": classes_dict
    }

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
    parser.add_argument('--batch', type=int, default=16, help='Batch size (-1 for auto batch)')
    parser.add_argument('--device', type=str, default=None, help="Device: '0', 'cpu', etc. Default: auto")
    parser.add_argument('--project', type=str, default='runs/detect', help='Output project directory')
    parser.add_argument('--name', type=str, default='yolov8_custom', help='Run name')
    parser.add_argument('--export-format', type=str, default='onnx', help='Export format')
    parser.add_argument('--no-export', action='store_true', help='Disable export step')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to copy trained .pt file into')
    parser.add_argument('--no-save-best', action='store_true', help='Disable copying best.pt to models directory')
    parser.add_argument('--save-best-name', type=str, default=None, help='Override base name for saved model file')
    parser.add_argument('--resume', action='store_true', help='Resume training from previous checkpoint')
    parser.add_argument('--no-simplify', action='store_true', help='Disable ONNX simplifier')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic ONNX axes')
    parser.add_argument('--opset', type=int, default=None, help='ONNX opset version')

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
        resume=args.resume,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        opset=args.opset,
    )
