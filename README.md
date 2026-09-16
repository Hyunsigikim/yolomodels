# YOLOv8 Object Detection Training Pipeline

이 프로젝트는 Label Studio에서 라벨링된 데이터셋을 YOLO 학습용 디렉토리 구조로 구성하고, Ultralytics YOLOv8로 학습/검증/ONNX 내보내기를 수행하기 위한 올인원 파이프라인을 제공합니다.

---

## 🏷️ Label Studio 연동 및 데이터셋 준비 가이드

이 프로젝트의 데이터 수집 및 라벨링은 **Label Studio**를 사용합니다.
10.1.100.82 서버 터미널에 'label-studio' 입력 후 실행
http://10.1.100.82:8080/ 을 브라우저 입력하여 접속


### 1. Label Studio 환경 정보
- **계정**: `eunee0521@goodnet.co.kr`
- **대상 프로젝트**: `topveiw`

### 2. 클래스 매핑 정보 (Class Mapping)
| Class ID | Class Name | Description |
| :---: | :--- | :--- |
| `0` | **bike** | 오토바이 / 자전거 |
| `1` | **car** | 일반 승용차 |
| `2` | **other** | 기타 미분류 차량/객체 |
| `3` | **truck** | 트럭 / 대형 화물차 |

---

### 3. Label Studio에서 데이터 내보내기 (Export)
1. **Label Studio 로그인**: `eunee0521@goodnet.co.kr` 계정으로 접속합니다.
2. **프로젝트 선택**: 대시보드에서 `topveiw` 프로젝트를 선택합니다.
3. **데이터 내보내기**:
   - 우측 상단의 **[Export]** 버튼을 클릭합니다.
   - 내보내기 포맷 목록에서 **`YOLO`** 형식을 선택합니다.
   - **[Export]** 버튼을 눌러 생성된 ZIP 파일(예: `project-X-at-YYYY-MM-DD.zip`)을 다운로드합니다.

---

### 4. 프로그램 내 데이터 업로드 (배치 위치)
1. 다운로드받은 ZIP 파일의 압축을 해제합니다.
2. 프로젝트 디렉토리(`D:\Developer\yolomodels` 또는 `D:\yolomodels`)의 `exports/` 하위에 프로젝트 이름으로 폴더를 생성하고 파일들을 배치합니다.
   
   **폴더 구조 예시:**
   ```
   yolomodels/
   └── exports/
       └── topveiw/                 # 프로젝트 폴더 (예: topveiw, project4 등)
           ├── images/              # 원본 이미지 파일들 (.jpg, .png 등)
           │   ├── img1.jpg
           │   └── img2.jpg
           ├── labels/              # YOLO 라벨 파일들 (*.txt, *_backup.txt)
           │   ├── img1.txt
           │   └── img2.txt
           ├── notes.json           # Label Studio 메타데이터 및 클래스 정보
           └── classes.txt          # (선택) 클래스 이름 목록
   ```

---

## 🚀 전체 작업 워크플로우 (Quick Start)

### Step 1. 가상환경 활성화
터미널을 열고 YOLO용 가상환경을 활성화합니다.
```cmd
# Windows
.\.yolov8\Scripts\activate

# Linux / Mac
source .yolov8/bin/activate
```

### Step 2. 데이터셋 분할 (Dataset Split)
`exports/` 하위의 원본 데이터를 YOLO 학습용(train / val / test)으로 자동 분할합니다.
```bash
# 기본 실행 (exports/ 폴더 내 최신 프로젝트 폴더를 자동 탐색하여 datasets/yolo_dataset에 생성)
python split_dataset.py

# 특정 프로젝트 및 출력 폴더 지정 실행 예시
python split_dataset.py --input exports/topveiw --output datasets/yolo_dataset4
```
> **실행 결과 확인:**  
> `datasets/yolo_dataset4/` 폴더 하위에 `train/`, `val/`, `test/` 및 `dataset.yaml`이 정상 생성되었는지 확인합니다.

### Step 3. YOLOv8 모델 학습 (Train)
생성된 `dataset.yaml`을 지정하여 모델 학습을 시작합니다.
```bash
# 기본 학습 (최근 생성된 dataset.yaml 자동 인식)
python train.py --model-size yolov8m.pt --epochs 100

# 특정 데이터셋 지정 학습
python train.py --data-yaml datasets/yolo_dataset4/dataset.yaml --model-size yolov8m.pt --epochs 100
```

### Step 4. 학습 완료 및 베스트 모델 추출 & 사용
1. 학습이 완료되면 자동으로 PyTorch 가중치(`.pt`)와 ONNX(`.onnx`) 파일이 생성됩니다.
2. **결과 가중치 경로**:
   - 최고 성능 모델 (PyTorch): `runs/detect/<실행명>/weights/best.pt`
   - 배포용 모델 (ONNX): `runs/detect/<실행명>/weights/best.onnx`
3. 생성된 `best.pt` 또는 `best.onnx` 모델 파일을 복사/추출하여 실제 인공지능 관제/추론 프로그램에 업로드하여 사용합니다.

---

## ⚡ 원클릭 파이프라인 (자동화 스크립트)
데이터셋 분할부터 학습/ONNX 변환까지 한 번에 실행하려면 배치 스크립트를 사용할 수 있습니다.

```cmd
# Windows (최신 exports 폴더 자동 탐색 및 학습)
run_pipeline.bat

# 특정 입력 폴더 지정 시
run_pipeline.bat --input exports/topveiw
```

---

## 📂 전체 프로젝트 디렉토리 구조
```
yolomodels/
├── exports/                     # Label Studio에서 내보낸 데이터 저장 폴더
│   └── topveiw/                 # 개별 프로젝트 폴더
│       ├── images/
│       ├── labels/
│       └── notes.json
├── datasets/                    # 분할 완료된 YOLO 데이터셋 저장소
│   └── yolo_dataset/            # split_dataset.py의 출력 위치
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       ├── test/
│       └── dataset.yaml
├── models/                      # 기본 모델 가중치 파일 보관
├── runs/                        # 학습 결과 및 모델 가중치 저장 디렉토리
│   └── detect/
│       └── yolov8_custom/
│           ├── weights/
│           │   ├── best.pt      # 🌟 최고 성능 모델
│           │   ├── best.onnx    # 🌟 ONNX 배포 모델
│           │   └── last.pt
│           ├── results.png      # 학습 메트릭 그래프
│           └── confusion_matrix.png
├── split_dataset.py             # 데이터셋 분할 스크립트
├── train.py                     # 모델 학습 및 ONNX 변환 스크립트
├── run_pipeline.bat             # 원클릭 파이프라인 실행 배치 파일 (Windows)
├── run_pipeline.sh              # 원클릭 파이프라인 실행 셸 스크립트 (Linux)
├── python_test.py               # CUDA / GPU 연동 확인 스크립트
└── requirements.txt             # 의존성 패키지 목록
```

---

## ⚙️ 초기 환경 설치 방법

### 1. 가상 환경 생성 및 활성화
```bash
python -m venv .yolov8

# Windows
.\.yolov8\Scripts\activate

# Linux/Mac
source .yolov8/bin/activate
```

### 2. PyTorch 설치 (CUDA/GPU 권장)
```bash
# CUDA 11.8 기준
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. GPU 및 패키지 정상 확인
```bash
python python_test.py
```

---

## 🛠️ 주요 매개변수 옵션

### train.py 옵션
| 매개변수 | 설명 | 기본값 |
|---|---|---|
| `--data-yaml` | 데이터셋 설정 YAML 파일 경로 | 최신 탐색 |
| `--model-size` | YOLOv8 모델 크기 (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`) | `yolov8s.pt` |
| `--epochs` | 학습 에포크 수 | `50` |
| `--batch` | 배치 크기 (GPU 메모리에 맞춰 조정) | `16` |
| `--imgsz` | 입력 이미지 해상도 | `640` |
| `--device` | 학습 디바이스 (예: `0`, `cpu`) | 자동 감지 |
| `--project` | 학습 결과 저장 상위 디렉토리 | `runs/detect` |
| `--name` | 실행 결과 폴더명 | `yolov8_custom` |
| `--export-format` | 내보내기 포맷 (`onnx`, `torchscript` 등) | `onnx` |

---

## 🚀 Python 코드에서 학습 모델 추론 예시

### PyTorch (`.pt`) 추론
```python
from ultralytics import YOLO

# 최고 성능 모델 로드
model = YOLO('runs/detect/yolov8_custom/weights/best.pt')

# 이미지 추론
results = model('test_image.jpg')

# 결과 시각화
results[0].show()
```

### ONNX (`.onnx`) 직접 내보내기 / 변환
```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolov8_custom/weights/best.pt')
model.export(format='onnx')
```

---

## 📝 참고 자료
- [Ultralytics YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [Label Studio 공식 가이드](https://labelstud.io/guide/)

## 📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
