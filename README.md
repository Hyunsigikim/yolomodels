# YOLOv8 Object Detection Training Pipeline
 
 이 프로젝트는 이미지/라벨 데이터셋을 YOLO 학습용 디렉토리 구조로 구성하고, Ultralytics YOLOv8로 학습/추론/ONNX 내보내기를 수행하기 위한 스크립트들을 제공합니다.
 
 ## 📂 프로젝트 구조
 ```
 yolomodels/
 ├── yolo_dataset4/                # YOLO 형식 데이터셋 (split_dataset.py 결과)
 │   ├── train/
 │   │   ├── images/
 │   │   └── labels/
 │   ├── val/
 │   │   ├── images/
 │   │   └── labels/
 │   ├── test/
 │   │   ├── images/
 │   │   └── labels/
 │   └── dataset.yaml              # 데이터셋 설정 파일
 ├── runs/                        # 학습 결과 저장 디렉토리
 ├── split_dataset.py             # 데이터셋 분할 스크립트
 ├── train.py                     # 모델 학습 스크립트
 ├── python_test.py               # CUDA/GPU 확인 스크립트
 └── requirements.txt             # 필요한 파이썬 패키지
 ```

 ## 🎯 주요 기능
- 데이터셋을 학습/검증/테스트 세트로 자동 분할
- YOLOv8 모델 학습
- 학습된 모델을 ONNX 형식으로 내보내기
- 다양한 YOLOv8 모델 크기 지원 (nano ~ xlarge)

 ## 📊 입력/출력
### 입력
1. **이미지 데이터**: 객체가 포함된 이미지 파일들 (JPG, PNG 등)
2. **라벨 데이터**: 이미지에 대한 라벨 파일들 (YOLO 형식 .txt)

### 출력
1. **YOLO 형식 데이터셋**: 변환된 이미지와 라벨 파일들
2. **학습된 모델**: PyTorch(.pt) 및 ONNX 형식의 모델 가중치
3. **학습 결과**: 학습 메트릭, 시각화 자료, 예측 결과

 ## 🚀 시작하기
 YOLO 학습/추론은 **Ultralytics(Pytorch)** 기반입니다. TensorFlow와 같은 환경에서 같이 쓰면 의존성이 충돌할 수 있으니, YOLO용 가상환경을 별도로 구성하는 것을 권장합니다.

 ## 🛠️ 스크립트 설명
 
 ### 1. split_dataset.py

#### 개요
이미지와 라벨 데이터셋을 학습, 검증, 테스트 세트로 분할하고 YOLO 형식에 맞게 디렉토리 구조를 구성합니다.

#### 주요 기능
- 사용자 정의 비율로 데이터셋 분할 (기본값: 학습 70%, 검증 15%, 테스트 15%)
- 이미지-라벨 쌍을 유지하며 분할
- 중복 없이 무작위 샘플링
- YOLO 학습을 위한 디렉토리 구조 자동 생성

 #### 사용 방법
 ```bash
 python split_dataset.py --project-dir project4 --output-dir yolo_dataset4
 ```

 `project_dir` 하위에 다음 구조가 있다고 가정합니다.

 - `images/*.jpg`
 - `labels/*.txt`
 - `notes.json` (클래스 이름을 읽어 `dataset.yaml` 생성 시 사용)

 분할 비율/시드/이미지 확장자는 옵션으로 조정할 수 있습니다.

 ```bash
 python split_dataset.py --project-dir project4 --output-dir yolo_dataset4 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42 --exts jpg,jpeg,png
 ```

 ### 2. train.py
 
#### 개요
 Ultralytics YOLOv8 모델을 사용하여 객체 감지 모델을 학습시키고, 학습 완료 후 ONNX로 내보내는 스크립트입니다.
 
#### 주요 기능
- 다양한 크기의 사전 학습된 YOLOv8 모델 지원 (nano ~ xlarge)
- ONNX 형식으로 모델 내보내기
 - 학습 파라미터 조정 가능 (에포크, 배치 크기, 이미지 크기)

 #### 사용 예시
 ```bash
 # 기본 학습 (YOLOv8 medium 모델, 100 에포크)
 python train.py --data-yaml yolo_dataset4/dataset.yaml --model-size yolov8m.pt --epochs 100
 ```

 ## ⚙️ 설치 및 설정
 
 ### 1. 가상 환경 설정 (권장)
 ```bash
 # 가상 환경 생성
 python -m venv .yolov8
 
 # Windows
 .\.yolov8\Scripts\activate
 
 # Linux/Mac
 source .yolov8/bin/activate
 ```
 
 ### 2. PyTorch 설치 (GPU 사용 시 권장)
 PyTorch는 설치 환경(CUDA)에 따라 휠이 다르므로 먼저 설치합니다.
 
 ```bash
 # CUDA 11.8 예시
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 ```
 
 ### 3. 필요한 패키지 설치
 ```bash
 pip install -r requirements.txt
 ```
 
 ### 4. 설치 확인
 ```bash
 python -c "import numpy; import cv2; print('numpy', numpy.__version__); print('cv2', cv2.__version__)"
 python -c "from ultralytics import YOLO; print('ultralytics import OK')"
 ```

 ## 🔄 데이터셋 준비 (YOLO 형식)
 1. 원본 데이터 폴더(`project_dir`)에 `images/`, `labels/`, `notes.json`을 준비합니다.
 2. `split_dataset.py`의 `project_dir`, `dataset_dir`를 본인 경로로 수정합니다.
 3. `python split_dataset.py` 실행 후 생성된 `dataset.yaml` 경로를 학습에 사용합니다.

 ## 🚂 YOLOv8 모델 학습
 
 ### 학습 실행
 ```bash
 python train.py --data-yaml ./yolo_dataset4/dataset.yaml --model-size yolov8s.pt --epochs 50
 ```

### 주요 매개변수
| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--data-yaml` | 데이터셋 설정 파일 경로 | 필수 |
| `--model-size` | 사용할 모델 크기 | yolov8s.pt |
| `--epochs` | 학습 에포크 수 | 50 |
| `--imgsz` | 입력 이미지 크기 | 640 |
| `--batch` | 배치 크기 | 16 |
| `--device` | 학습 디바이스 (예: 0, cpu) | 자동 |
| `--project` | 결과 저장 상위 폴더 | runs/detect |
| `--name` | 실행 이름 | yolov8_custom |
| `--export-format` | 내보내기 포맷 | onnx |
| `--no-export` | 내보내기 비활성화 | false |

 ### 학습 모니터링
학습 중에 TensorBoard를 사용하여 실시간으로 메트릭을 확인할 수 있습니다:
```bash
tensorboard --logdir runs/detect
```

 ## 📊 학습 결과 분석

 ### 출력 디렉토리 구조
```
runs/detect/yolov8_custom/
├── weights/
│   ├── best.pt         # 최고 성능 모델
│   └── last.pt         # 마지막 에포크 모델
├── events.out.tfevents.*  # TensorBoard 로그
├── args.yaml           # 학습 설정
├── confusion_matrix.png   # 혼동 행렬
├── F1_curve.png        # 정밀도-재현율 곡선
├── P_curve.png         # 정밀도-신뢰도 곡선
├── R_curve.png         # 재현율-신뢰도 곡선
├── results.png         # 학습 메트릭 시각화
└── val_batchX_labels.jpg  # 검증 배치 예시
```

 ### 주요 메트릭
- **mAP@0.5**: IoU=0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 0.5~0.95(0.05 간격)에서의 평균 mAP
- **Precision**: 정밀도 (True Positives / (True Positives + False Positives))
- **Recall**: 재현율 (True Positives / (True Positives + False Negatives))
- **mAP**: 모든 클래스의 평균 정밀도

 ## 🚀 학습된 모델 사용하기

 ### 추론 실행 예시
```python
from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('runs/detect/yolov8_custom/weights/best.pt')

# 이미지에 대한 추론
results = model('test_image.jpg')

# 결과 시각화
results[0].show()
```

 ### ONNX 형식으로 내보내기
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('runs/detect/yolov8_custom/weights/best.pt')

# ONNX 형식으로 내보내기
model.export(format='onnx')
```

 ### 배치 추론
```python
# 여러 이미지에 대한 추론
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# 결과 반복 처리
for result in results:
    boxes = result.boxes  # 바운딩 박스 객체
    print(boxes.xyxy)     # 바운딩 박스 좌표
    print(boxes.conf)     # 신뢰도 점수
    print(boxes.cls)      # 클래스 인덱스
```

 ## 📝 참고 자료
- [YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [Label Studio 문서](https://labelstud.io/guide/)

 ## 📄 라이선스
 이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
