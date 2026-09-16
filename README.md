# YOLOv8 Object Detection Training Pipeline

이 프로젝트는 Label Studio에서 라벨링된 데이터셋을 YOLO 학습용 디렉토리 구조로 구성하고, Ultralytics YOLOv8로 학습/검증/ONNX 내보내기까지 **단 하나의 명령어로 한 번에 수행**할 수 있는 올인원 파이프라인을 제공합니다.

---

## 🏷️ Label Studio 연동 및 데이터셋 준비 가이드

이 프로젝트의 데이터 수집 및 라벨링은 **Label Studio**를 사용합니다.

### 1. Label Studio 서버 접속 및 환경 정보
- **서버 실행**: `10.1.100.82` 서버 터미널에서 `label-studio` 명령어 입력 후 실행
- **웹 접속 주소**: [http://10.1.100.82:8080/](http://10.1.100.82:8080/)
- **로그인 계정**: `eunee0521@goodnet.co.kr`
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

## ⚡ 올인원 원클릭 실행 (추천)

프로젝트 폴더만 지정하면 **[데이터셋 분할 ➜ YOLOv8m 모델 학습 ➜ ONNX 변환 ➜ 최종 가중치 저장]**이 자동으로 한 번에 완료됩니다.

### 방법 1. 원클릭 배치 파일 실행 (가장 간편)
가상환경 활성화 및 필수 패키지 검사 후 바로 실행됩니다.
```cmd
# 1) 더블 클릭 또는 명령어 실행 (최신 프로젝트 자동 선택)
run_pipeline.bat

# 2) 특정 프로젝트 지정 실행
run_pipeline.bat topveiw
```

### 방법 2. Python 올인원 단일 실행 (`main.py`)
```bash
# 기본 실행 (exports 폴더 내 최신 프로젝트 자동 감지 또는 목록에서 선택)
python main.py

# 특정 프로젝트 폴더 지정 실행
python main.py topveiw

# 옵션 커스텀 실행 예시 (에포크, 모델 크기, 재개, 고속 하드링크 등)
python main.py topveiw --model-size yolov8m.pt --epochs 100 --batch 16

# 중단된 학습 이어하기 (Resume)
python main.py topveiw --resume
```

---

## 🎯 학습 완료 후 최종 모델 확인 및 배포

학습과 ONNX 변환이 완료되면 `models/` 디렉토리에 모델 가중치와 배포용 메타데이터 명세서가 자동 저장됩니다.

```
models/
├── topveiw_best_20260916.pt       # PyTorch 가중치 파일
├── topveiw_best_20260916.onnx     # 🌟 인공지능 프로그램 배포/업로드용 ONNX 모델 (최적화 완료)
└── topveiw_best_20260916_meta.json # 📋 클래스 ID-이름 맵핑 및 모델 사양 메타데이터
```

> **🚀 인공지능 프로그램 배포 방법**:  
> 생성된 `models/topveiw_best_YYYYMMDD.onnx` 모델과 `_meta.json` 명세서를 복사하여 관제/추론 프로그램에 업로드하여 즉시 사용하시면 됩니다.

---

## 🛠️ 세부 단계별 수동 실행 (필요 시)

전체 파이프라인 대신 각 단계를 개별적으로 실행하고자 할 경우 아래 명령어를 사용합니다.

### 1단계: 데이터셋 분할 (`split_dataset.py`)
```bash
# 기본 실행 (최신 exports 폴더 자동 탐색)
python split_dataset.py

# 특정 프로젝트 및 출력 폴더 지정
python split_dataset.py --input exports/topveiw --output datasets/topveiw_dataset
```

### 2단계: YOLOv8 모델 학습 (`train.py`)
```bash
# 기본 실행 (최근 생성된 dataset.yaml 자동 인식)
python train.py --model-size yolov8m.pt --epochs 100

# 특정 dataset.yaml 지정 실행
python train.py --data-yaml datasets/topveiw_dataset/dataset.yaml --model-size yolov8m.pt --epochs 100
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
│   └── topveiw_dataset/         # split_dataset.py의 출력 위치
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       ├── test/
│       └── dataset.yaml
├── models/                      # 🌟 최종 완성된 best 모델(.pt / .onnx) 보관소
│   ├── topveiw_best_20260916.pt
│   └── topveiw_best_20260916.onnx
├── runs/                        # YOLO 학습 로그 및 상세 메트릭 결과
│   └── detect/
│       └── topveiw_run/
├── main.py                      # 🌟 올인원 단일 실행 스크립트 (분할+학습+ONNX)
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

## 📝 참고 자료
- [Ultralytics YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [Label Studio 공식 가이드](https://labelstud.io/guide/)

## 📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
