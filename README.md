# ai-lens-helper

> 장소별 전시품 이미지를 학습하고, 추론 시 **장소명 + 이미지** 입력으로 해당 전시품(혹은 “재촬영 권유”)을 판별하는 경량 CLI/SDK.

---

## ✨ 주요 기능 (Features)

- **CLI 학습 파이프라인**: 폴더/메타에서 자동 로딩 → 전처리 → 증강 → 학습/검증 → 체크포인트/리포트 저장
- **추론 API/CLI**: `place + image` 입력 → 전시품 후보/신뢰도 반환, 임계치 미달 시 “새로 촬영 권고”
- **임베딩 기반 검색 + 분류 하이브리드**: 라벨 추론의 견고성과 OOD(전시품 외물체) 거절(Reject) 성능 강화
- **데이터 유효성 검사**: 이미지 수/해상도/중복 검사, 클래스 불균형 경고
- **모델/데이터 버저닝**: `semver` + 메타(JSON) 기록, 재현 가능한 런(Seed 고정, 환경 스냅샷)
- **경량 배포**: ONNX/TorchScript 내보내기, CPU·Edge 우선 설계

---

## 📦 설치 (Installation)

```bash
pip install ai-lens-helper
# or (개발)
pip install -e .[dev]
```

---

## 🗂️ 데이터 디렉터리 규약 (Dataset Layout)

```
./data/
  A/
    ㄱ/
      img_001.jpg
      ... (최소 10장 권장)
    ㄴ/
    ...
  B/
  C/
```

- 폴더명: **장소/전시품**
- 각 전시품 최소 10장, 다양한 각도/거리/조명 포함 권장
- (선택) `metadata.json`: 촬영기기/날짜/라벨 설명 등

---

## 🚀 빠른 시작 (Quickstart)

### 0) GUI 애플리케이션 (추천!)

**YOLO+CLIP 기반 전시품 인식 시스템을 GUI에서 쉽게 사용하세요!**

```bash
# GUI 실행
python3 gui_app.py
# 또는
./run_gui.sh
```

**기능:**
- 📚 **Tab 1: 인덱스 빌드 (학습)** - 수집한 이미지로 YOLO+CLIP 인덱스 생성
- 🔍 **Tab 2: Inference (테스트)** - 실시간 이미지 테스트 및 결과 확인
- 🤖 **Tab 3: OpenAI Vision** - OpenAI GPT-4 Vision API로 즉시 테스트 (학습 불필요)

**자세한 사용법:**
- [GUI_USAGE.md](./GUI_USAGE.md) - GUI 전체 사용법
- [OPENAI_VISION_GUIDE.md](./OPENAI_VISION_GUIDE.md) - OpenAI Vision API 가이드

### 1) CLI로 YOLO+CLIP 인덱스 빌드

**YOLO+CLIP 모델 사용 (권장):**

```bash
# 인덱스 빌드
ai-lens build-clip-index \
  --data-root ./data \
  --place "경복궁" \
  --save ./indexes/경복궁_clip \
  --device cpu \
  --yolo-model yolov8n.pt \
  --clip-model ViT-B-16
```

옵션 요약:
- `--data-root`: 데이터 루트 경로 (data/{place}/{item}/*.jpg 구조)
- `--place`: 장소명 (예: "경복궁", "국립중앙박물관")
- `--save`: 저장 경로 (확장자 제외, .json/.faiss 자동 생성)
- `--device`: `cpu` 또는 `cuda`
- `--yolo-model`: YOLO 모델 크기 (`yolov8n.pt`~`yolov8l.pt`)
- `--clip-model`: CLIP 모델 (`ViT-B-16` 권장)

**기존 Color-based 모델 (프로토타입):**

```bash
ai-lens build-index \
  --model model.json \
  --data-root ./data \
  --place A \
  --save ./A.index
```

### 2) Inference (단일 이미지)

```bash
ai-lens infer \
  --model ./indexes/경복궁_clip.json \
  --place "경복궁" \
  --image ./sample.jpg \
  --reject-threshold 0.7 \
  --topk 5
```

**참고:** 모델 타입(YOLO+CLIP vs Color-based)은 자동 감지됩니다.

출력 예시(JSON):

```json
{
  "place": "A",
  "predictions": [
    { "item": "ㄱ", "score": 0.93 },
    { "item": "ㄴ", "score": 0.71 },
    { "item": "ㅁ", "score": 0.4 }
  ],
  "decision": "accept",
  "selected_item": "ㄱ",
  "confidence": 0.93
}
```

임계치 미달 시:

```json
{
  "place": "A",
  "predictions": [
    { "item": "ㄱ", "score": 0.55 },
    { "item": "ㄴ", "score": 0.5 }
  ],
  "decision": "recollect",
  "message": "전시품이 아닌 가능성이 높습니다. 초점/구도/조명 개선 후 재촬영 해주세요.",
  "hints": [
    "작품이 프레임 중앙에 오도록",
    "반사/글레어 줄이기",
    "더 가까이 촬영"
  ]
}
```

### 3) 배치 추론(폴더)

```bash
ai-lens infer-batch \
  --model ./runs/2025-10-16/best.ckpt \
  --place A \
  --input-dir ./captures/A \
  --output ./captures/A_results.jsonl
```

---

## 🔧 Python SDK

```python
from ai_lens_helper import Lens

lens = Lens(model_path="./runs/2025-10-16/best.onnx")
result = lens.infer(place="A", image_path="./sample.jpg", reject_threshold=0.62)
print(result.decision, result.selected_item, result.confidence)
```

---

## ⚙️ 구성(설정) 파일 예시

```yaml
# configs/default.yaml
seed: 42
img_size: 224
backbone: vit_b16
optimizer: adamw
lr: 3e-4
weight_decay: 0.05
scheduler: cosine
loss: arcface
embedding_dim: 512
augment:
  random_resized_crop: true
  color_jitter: 0.2
  horizontal_flip: true
  gaussian_blur: true
reject:
  threshold: 0.62
  min_top1_margin: 0.12
export:
  onnx: true
  opset: 17
```

---

## 🧪 평가 (Evaluation)

- **Split**: Stratified K-fold(5x) + 장소별 Holdout(신규 방문 일반화 체크)
- **지표**: Top-1/Top-3 Acc, mAP, AUROC(Reject), EER, FPR@TPR
- **리포트**: `runs/*/report.html` (혼동행렬, PR/ROC, 임베딩 t-SNE/UMAP)

---

## 🛠️ 에러 코드

| 코드 | 의미                   | 해결                              |
| ---- | ---------------------- | --------------------------------- |
| E100 | 데이터 레이아웃 불일치 | `data/place/item/*.jpg` 구조 점검 |
| E110 | 클래스당 이미지 부족   | 각 전시품 ≥10장 수집              |
| E200 | 모델 로드 실패         | 경로/권한/버전 확인               |
| E310 | Reject 임계치 미설정   | `--reject-threshold` 지정         |

---

## 🔭 로드맵

- 모바일 온디바이스( CoreML / NNAPI ) 내보내기
- 하드 네거티브 마이닝 자동화
- 멀티장소 동시 추론 + 캐싱
- 온-사이트 증분학습(소량 신샘플로 업데이트)

---

## 📝 라이선스

#### AI 이미지 수집 CLI

ai-lens collect-data museum_spec.json --num-images 20 --output-root ./data --engine naver --headless --report crawl_report.json

ai-lens collect-data museum_spec.json \
 --num-images 20 \
 --output-root ./data \
 --engine naver \
 --headless \
 --report crawl_report.json
