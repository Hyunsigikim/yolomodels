# 📜 yolomodels 프로젝트 장기기록 마스터 작업 문서 (PROJECT_LONGTERM_MEMORY.md)

## 📌 1. 프로젝트 개요 (Overview)
- **프로젝트 명칭**: YOLOv8 객체 인식 파이프라인 (YOLOv8 Object Detection Pipeline)
- **프로젝트 목적**: Label Studio에서 라벨링된 데이터를 기반으로 YOLOv8 학습용 데이터셋 분할, 모델 학습, ONNX 포맷 변환 및 최고 성능 가중치 파일 추출을 원스톱으로 처리하는 통합 파이프라인 구축.
- **핵심 원칙 및 운영 정책**:
  - **학습 전용 머신 분리 운영**: 대용량 이미지, 라벨 데이터셋(`exports/`, `datasets/`), 학습 결과물(`runs/`), 가중치 바이너리(`models/*.pt`, `*.onnx`) 등은 Git 형상관리에서 엄격히 제외하고 코드 및 구조(`.gitkeep`)만 추적.
  - **단일 진입점 올인원 실행**: `main.py` 및 `run_pipeline.bat`을 통해 데이터셋 분할부터 학습, 변환, 모델 추출까지 일원화.
  - **Label Studio 표준 연동**:
    - **서버 IP**: `10.1.100.82` (포트 `8080`)
    - **계정**: `eunee0521@goodnet.co.kr`
    - **기본 프로젝트**: `topveiw`
    - **표준 클래스 정의**: `0: bike`, `1: car`, `2: other`, `3: truck`

---

## 📐 2. 시스템 아키텍처 및 디렉토리 구조 (Structure)
```files
yolomodels/
├── PROJECT_LONGTERM_MEMORY.md       # 📜 단일 프로젝트 장기기록 마스터 작업 문서
├── README.md                        # 📖 사용자 가이드 및 연동 매뉴얼
├── .gitignore                       # 🛡️ 대용량 데이터/가중치 Git 제외 설정
├── requirements.txt                 # 📦 필수 파이썬 패키지 의존성 목록
├── main.py                          # 🌟 올인원 단일 실행 파이프라인 엔진
├── split_dataset.py                 # ✂️ 데이터셋 분할 및 dataset.yaml 생성 모듈
├── train.py                         # 🚂 Ultralytics YOLOv8 학습 및 ONNX 변환 모듈
├── python_test.py                   # 🔍 CUDA 및 GPU 환경 검증 스크립트
├── run_pipeline.bat                 # ⚡ Windows 원클릭 파이프라인 배치 스크립트
├── run_pipeline.sh                  # ⚡ Linux 원클릭 파이프라인 셸 스크립트
├── setup_env.bat                    # ⚙️ Windows 가상환경 설치 스크립트
├── setup_env.sh                     # ⚙️ Linux 가상환경 설치 스크립트
├── exports/                         # 📥 Label Studio YOLO Export 데이터 수신함 (Git 제외)
│   └── .gitkeep
├── datasets/                        # 🗃️ 분할 완료된 YOLO 학습 데이터셋 (Git 제외)
│   └── .gitkeep
├── models/                          # 💾 최종 추출된 best 가중치(.pt, .onnx) 보관소 (Git 제외)
│   └── .gitkeep
└── runs/                            # 📊 YOLO 상세 학습 메트릭 및 로그 (Git 제외)
    └── .gitkeep
```

---

## 🗂️ 3. 비즈니스 요구사항 및 인터페이스 스키마 (Requirements & Schemas)

### 3.1 클래스 라벨 매핑 규격 (Class Mapping Schema)
| Class ID | Class Name | 한글 설명 |
| :---: | :--- | :--- |
| `0` | `bike` | 오토바이 / 자전거 |
| `1` | `car` | 일반 승용차 |
| `2` | `other` | 기타 미분류 차량/물체 |
| `3` | `truck` | 트럭 / 대형 화물차량 |

### 3.2 Label Studio 연동 규격
- **Export 포맷**: YOLO 포맷 (images, labels, notes.json / classes.txt 포함)
- **배치 경로**: `exports/<프로젝트명>/` (예: `exports/topveiw/`)

### 3.3 파이프라인 실행 인터페이스
```bash
# 기본 실행 (최신 exports 자동 감지)
python main.py

# 특정 프로젝트 지정 및 모델 하이퍼파라미터 전달
python main.py <프로젝트명> --model-size yolov8m.pt --epochs 100 --batch 16
```

---

## 📝 4. 프로젝트 변경 이력 (Change Log)
| 일자 | 구분 | 변경 내용 | 비고 |
| :--- | :---: | :--- | :--- |
| 2026-09-16 | Feat | Label Studio 연동 가이드 및 서버 정보(`10.1.100.82`), 클래스 매핑 수록 | README.md 최신화 |
| 2026-09-16 | Refactor | 올인원 단일 실행 파이프라인 `main.py` 구축 및 `split_dataset.py`, `run_pipeline.bat` 연동 개선 | 사용성 대폭 향상 |
| 2026-09-16 | Chore | 대용량 가중치/데이터셋 Git 분리 설정(`.gitignore`, `.gitkeep`) 및 `PROJECT_LONGTERM_MEMORY.md` 마스터 문서 수립 | 형상관리 최적화 |
