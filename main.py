import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from split_dataset import split_dataset, detect_input_directory
from train import train_model

def get_available_projects():
    """Find all valid project directories under exports/"""
    exports_dir = Path('exports')
    if not exports_dir.exists() or not exports_dir.is_dir():
        return []

    subdirs = [d for d in exports_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    valid = []
    for sd in subdirs:
        if (sd / 'images').exists() and (sd / 'images').is_dir():
            valid.append(sd)

    # Sort by modification time (most recent first)
    valid.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return valid

def resolve_project_dir(user_input=None):
    """Resolve project directory from user input or interactive prompt"""
    if user_input:
        p = Path(user_input)
        if p.exists() and p.is_dir():
            return p
        
        # Check under exports/
        p_under_exports = Path('exports') / user_input
        if p_under_exports.exists() and p_under_exports.is_dir():
            return p_under_exports

        raise FileNotFoundError(f"프로젝트 폴더를 찾을 수 없습니다: '{user_input}' (또는 'exports/{user_input}')")

    # No input provided: discover projects in exports/
    projects = get_available_projects()
    if not projects:
        # Check if exports/ itself has images/
        exports_dir = Path('exports')
        if (exports_dir / 'images').exists() and (exports_dir / 'images').is_dir():
            return exports_dir
        
        raise FileNotFoundError(
            "exports/ 폴더 내에 유효한 라벨스튜디오 프로젝트(images/ 폴더 포함)가 존재하지 않습니다.\n"
            "Label Studio에서 YOLO 형식으로 Export한 압축 파일을 exports/<프로젝트명>/ 에 풀어주세요."
        )

    if len(projects) == 1:
        print(f"📦 단일 프로젝트 감지: '{projects[0].name}' (경로: {projects[0]})")
        return projects[0]

    # Multiple projects found
    print("\n=======================================================")
    print("📁 exports/ 폴더 내 발견된 프로젝트 목록:")
    print("=======================================================")
    for idx, proj in enumerate(projects, start=1):
        mtime = datetime.fromtimestamp(proj.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        default_mark = " [최신/기본값]" if idx == 1 else ""
        print(f"  {idx}) {proj.name:<20} (수정일: {mtime}){default_mark}")
    print("=======================================================")

    # Interactive input if terminal is interactive
    if sys.stdin.isatty():
        try:
            choice = input(f"실행할 프로젝트 번호를 입력하세요 (기본값: 1): ").strip()
            if not choice:
                return projects[0]
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(projects):
                return projects[choice_idx]
            else:
                print("잘못된 번호입니다. 가장 최신 프로젝트(1번)로 진행합니다.")
                return projects[0]
        except Exception:
            return projects[0]
    else:
        print(f"자동으로 최신 프로젝트 '{projects[0].name}'를 선택합니다.")
        return projects[0]

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 올인원 파이프라인 (데이터 분할 + 모델 학습 + ONNX 변환 + 가중치 저장)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Positional or named project argument
    parser.add_argument('project_name', nargs='?', default=None,
                        help="exports/ 하위의 프로젝트 폴더명 또는 경로 (예: topveiw, exports/project4)")
    parser.add_argument('--project', '-p', type=str, default=None,
                        help="프로젝트 폴더명 (예: topveiw)")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="분할 데이터셋 저장 경로 (기본값: datasets/<프로젝트명>_dataset)")
    parser.add_argument('--model-size', type=str, default='yolov8m.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help="YOLOv8 모델 크기")
    parser.add_argument('--epochs', type=int, default=100,
                        help="학습 에포크 수")
    parser.add_argument('--batch', type=int, default=16,
                        help="배치 크기")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="입력 이미지 해상도")
    parser.add_argument('--device', type=str, default=None,
                        help="학습 장치 ('0', 'cpu', 등. 기본값: 자동 감지)")
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help="학습 데이터 비율")
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help="검증 데이터 비율")
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help="테스트 데이터 비율")

    args = parser.parse_args()

    project_query = args.project or args.project_name
    project_dir = resolve_project_dir(project_query)
    project_name = project_dir.name

    if args.output:
        output_dataset_dir = Path(args.output)
    else:
        output_dataset_dir = Path('datasets') / f"{project_name}_dataset"

    print(f"\n🚀 [YOLO Pipeline 시작] 프로젝트: '{project_name}'")
    print(f"📍 입력 데이터 경로: {project_dir.resolve()}")
    print(f"📍 출력 데이터셋 경로: {output_dataset_dir.resolve()}")
    print(f"⚙️  모델 설정: {args.model_size} | Epochs: {args.epochs} | Batch: {args.batch}")

    # Step 1: Split Dataset
    print("\n-------------------------------------------------------")
    print("🔄 [Step 1/2] 데이터셋 분할 및 dataset.yaml 생성 중...")
    print("-------------------------------------------------------")
    dataset_yaml_path, class_names = split_dataset(
        input_dir=str(project_dir),
        output_dir=str(output_dataset_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"✅ 데이터셋 준비 완료: {dataset_yaml_path}")
    print(f"🏷️  인식된 클래스 목록: {class_names}")

    # Step 2: Train Model & Export ONNX
    print("\n-------------------------------------------------------")
    print("🚂 [Step 2/2] YOLOv8 모델 학습 및 ONNX 변환 시작...")
    print("-------------------------------------------------------")
    run_name = f"{project_name}_run"
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')
    base_save_name = f"{project_name}_best"

    train_model(
        data_yaml=str(dataset_yaml_path),
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project='runs/detect',
        name=run_name,
        export_format='onnx',
        no_export=False,
        models_dir=str(models_dir),
        save_best=True,
        save_best_name=base_save_name,
    )

    # Summary
    pt_model_path = models_dir / f"{base_save_name}_{today}.pt"
    onnx_model_path = models_dir / f"{base_save_name}_{today}.onnx"

    print("\n" + "=" * 70)
    print("🎉 [파이프라인 완료] 모델 학습 및 ONNX 변환이 성공적으로 끝났습니다!")
    print("=" * 70)
    print(f"📁 프로젝트명: {project_name}")
    print("🎯 클래스 정보:")
    for cid, cname in class_names.items():
        print(f"   {cid}: {cname}")
    print("\n💾 생성된 최종 모델 가중치:")
    if pt_model_path.exists():
        print(f"   - PyTorch 모델 (.pt):   {pt_model_path.resolve()}")
    if onnx_model_path.exists():
        print(f"   - ONNX 배포 모델 (.onnx): {onnx_model_path.resolve()}")
    print("\n🚀 인공지능 프로그램 사용 안내:")
    print(f"   위 생성된 '{onnx_model_path.name}' (또는 .pt) 파일을 복사하여")
    print(f"   인공지능 추론/관제 프로그램에 업로드하여 사용하시면 됩니다.")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
