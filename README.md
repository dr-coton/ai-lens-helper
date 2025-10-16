# ai-lens-helper

> 장소별 전시품 이미지를 학습하고, 추론 시 **장소명 + 이미지** 입력으로 해당 전시품(혹은 “재촬영 권유”)을 판별하는 경량 CLI/SDK.

---

## ✨ 주요 기능 (Features)

* **CLI 학습 파이프라인**: 폴더/메타에서 자동 로딩 → 전처리 → 증강 → 학습/검증 → 체크포인트/리포트 저장
* **추론 API/CLI**: `place + image` 입력 → 전시품 후보/신뢰도 반환, 임계치 미달 시 “새로 촬영 권고”
* **임베딩 기반 검색 + 분류 하이브리드**: 라벨 추론의 견고성과 OOD(전시품 외물체) 거절(Reject) 성능 강화
* **데이터 유효성 검사**: 이미지 수/해상도/중복 검사, 클래스 불균형 경고
* **모델/데이터 버저닝**: `semver` + 메타(JSON) 기록, 재현 가능한 런(Seed 고정, 환경 스냅샷)
* **경량 배포**: ONNX/TorchScript 내보내기, CPU·Edge 우선 설계

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

* 폴더명: **장소/전시품**
* 각 전시품 최소 10장, 다양한 각도/거리/조명 포함 권장
* (선택) `metadata.json`: 촬영기기/날짜/라벨 설명 등

---

## 🚀 빠른 시작 (Quickstart)

### 1) 학습

```bash
ai-lens train \
  --data-root ./data \
  --place A --place B --place C \
  --backbone vit_b16 \
  --epochs 20 --batch-size 32 \
  --save-dir ./runs/2025-10-16
```

옵션 요약:

* `--data-root`: 데이터 루트 경로
* `--place`: 대상 장소(여러 개 가능)
* `--backbone`: `vit_b16|resnet50|efficientnet_b3|convnext_tiny`
* `--epochs`, `--batch-size`, `--lr`, `--img-size`
* `--embedding`: 임베딩 차원 (예: 512)
* `--loss`: `ce|arcface|triplet`
* `--export-onnx`: ONNX 내보내기 여부

### 2) 추론 (단일 이미지)

```bash
ai-lens infer \
  --model ./runs/2025-10-16/best.ckpt \
  --place A \
  --image ./sample.jpg \
  --reject-threshold 0.62 \
  --topk 3
```

출력 예시(JSON):

```json
{
  "place": "A",
  "predictions": [
    {"item": "ㄱ", "score": 0.93},
    {"item": "ㄴ", "score": 0.71},
    {"item": "ㅁ", "score": 0.40}
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
    {"item": "ㄱ", "score": 0.55},
    {"item": "ㄴ", "score": 0.50}
  ],
  "decision": "recollect",
  "message": "전시품이 아닌 가능성이 높습니다. 초점/구도/조명 개선 후 재촬영 해주세요.",
  "hints": ["작품이 프레임 중앙에 오도록", "반사/글레어 줄이기", "더 가까이 촬영"]
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

* **Split**: Stratified K-fold(5x) + 장소별 Holdout(신규 방문 일반화 체크)
* **지표**: Top-1/Top-3 Acc, mAP, AUROC(Reject), EER, FPR@TPR
* **리포트**: `runs/*/report.html` (혼동행렬, PR/ROC, 임베딩 t-SNE/UMAP)

---

## 🛠️ 에러 코드

| 코드   | 의미             | 해결                            |
| ---- | -------------- | ----------------------------- |
| E100 | 데이터 레이아웃 불일치   | `data/place/item/*.jpg` 구조 점검 |
| E110 | 클래스당 이미지 부족    | 각 전시품 ≥10장 수집                 |
| E200 | 모델 로드 실패       | 경로/권한/버전 확인                   |
| E310 | Reject 임계치 미설정 | `--reject-threshold` 지정       |

---

## 🔭 로드맵

* 모바일 온디바이스( CoreML / NNAPI ) 내보내기
* 하드 네거티브 마이닝 자동화
* 멀티장소 동시 추론 + 캐싱
* 온-사이트 증분학습(소량 신샘플로 업데이트)

---

## 📝 라이선스

MIT

---

# PLAN.md — 개발 상세 계획

## 1. 문제 재정의

* 입력: (장소명, 이미지)
* 출력: {전시품 라벨, 신뢰도, 결정(accept | recollect), 이유/힌트}
* 제약: 전시품 외 물체/OOD에 대한 거절 성능 필수

## 2. 아키텍처 개요

* **Backbone**: ViT-B/16(기본) 또는 ConvNeXt-Tiny — 전이학습
* **헤드**: (A) Softmax 분류 헤드 + (B) Metric 임베딩 헤드(ArcFace/Triplet)
* **결정 로직**:

  1. Top-1 점수 ≥ T이면 Accept
  2. 아니면 임베딩 최근접 거리 d < D이면 Accept(세컨드 오피니언)
  3. 둘 다 미달 시 Recollect
* **인덱스**: 전시품별 대표 임베딩(또는 모든 샘플)으로 FAISS/ScaNN 인덱스 구축

## 3. 데이터 파이프라인

* 로더: `data/{place}/{item}/*.jpg`
* 전처리: EXIF 정규화, 해상도 제한(긴 변 1024), 라벨 검증, 중복 제거(pHash)
* 증강: RandomResizedCrop, ColorJitter, HFlip, GaussianBlur, RandomGray (실전 촬영 다양성 반영)
* 불균형 보정: 클래스 가중치 + 샘플러 + MixUp/CutMix(선택)

## 4. 학습 전략

* 손실: `CE + ArcFace(또는 Triplet)` 멀티-오브젝티브
* 스케줄러: Cosine with warmup, EMA(선택)
* 체크포인트: `best_top1`, `best_auroc_reject`, `last`
* 검증: 장소 내/장소 간 분리(도메인 일반화 체크)
* 하드 네거티브 마이닝: Epoch N 이후, 상호 혼동 쌍 재가중

## 5. 추론 로직 세부

1. Softmax 확률 `p_top1`, `margin = p1 - p2`
2. 임베딩 거리 `d_top1`
3. 규칙:

```text
if p_top1 >= T and margin >= M: accept
elif d_top1 <= D: accept
else: recollect
```

* 기본값: `T=0.62, M=0.12, D=0.65` (데이터로 재튜닝)
* 힌트 생성: 감지된 블러/노출/프레임오프센터 기반 규칙

## 6. 패키지 구조

```
ai_lens_helper/
  __init__.py
  cli.py
  config/
  data/
  models/
    backbones.py
    heads.py
    losses.py
  train/
    datamodule.py
    engine.py
    evaluator.py
    export.py
  infer/
    runner.py
    postprocess.py
    quality_check.py
    index.py
  utils/
    io.py
    metrics.py
    vis.py
```

## 7. CLI 설계

* `ai-lens validate-data --data-root ./data`
* `ai-lens train ...` (위 README 참조)
* `ai-lens build-index --model ckpt --data-root ./data/A --place A --save ./A.index`
* `ai-lens infer --model ckpt --place A --image path.jpg --reject-threshold 0.62`
* `ai-lens infer-batch --model ckpt --place A --input-dir ./folder`
* `ai-lens export --ckpt best.ckpt --onnx out.onnx`

## 8. 평가/리포팅

* 혼동행렬 + 오분류 Top-N 시각화(리콜 낮은 클래스 점검)
* OOD 샘플(관람객, 벽, 바닥, 안내문) 세트로 Reject ROC 측정
* 리포트 HTML 자동 생성 + JSON 요약(배포 파이프라인 입력)

## 9. 배포 & 최적화

* ONNX 변환, Dynamic Quantization
* 배치 추론 시 이미지 리사이즈 캐시 / 멀티프로세스
* 인덱스 메모리맵 + warmup

## 10. 품질/재현성

* Seed 고정, 버전/의존성 Lock, 모델/설정/데이터 해시 기록
* CI: lint, type-check, unit/CLI 테스트, 샘플 데이터로 1epoch 스모크 테스트

## 11. 보안/프라이버시

* 민감정보 미수집, 파일 경로/EXIF에서 PII 제거 옵션

## 12. 일정(예시)

* W1: 스캐폴딩/데이터 밸리데이터/기본 로더
* W2: 학습엔진(분류+임베딩), 기본 CLI
* W3: 평가/리포트/Reject 로직 고도화
* W4: 배포(ONNX)/인덱스/문서화

---

## 🤖 AI에게 맡길 프롬프트 (예시)

### (A) 데이터 정제/증강 제안 프롬프트

```
당신은 컴퓨터비전 학습 데이터 큐레이터입니다. 아래 폴더 구조의 전시품 이미지를 검토하고,
1) 중복/저화질 탐지 기준, 2) 권장 증강 파이프라인(촬영 환경 다양화 중심),
3) 클래스 불균형 보정 전략을 YAML로 제안하세요.
제약: 각 전시품 최소 10장, 실내 박물관 조명/반사 고려.
```

### (B) 하이브리드 추론 규칙 튜닝 프롬프트

```
당신은 모델평가 엔지니어입니다. Softmax 점수(p_top1, margin)와 임베딩 거리(d_top1) 기반의
accept/recollect 결정 임계치 (T, M, D)를 개발/검증 세트 분포를 근거로 산출하고,
Reject AUROC를 최대화하는 조합을 제안하세요. 최종 표 형태와 근거 그래프 설명을 포함하세요.
```

### (C) 하드 네거티브 마이닝 프롬프트

```
당신은 모델훈련 고도화 엔지니어입니다. 아래 혼동행렬과 Top-오분류 샘플 묶음을 분석하여,
서로 자주 혼동되는 전시품 쌍에 대한 하드 네거티브 마이닝 절차(采样/증강/로스 가중)를 제시하세요.
Epoch별 적용 스케줄과 기대 성능 향상 폭을 수치로 추정하세요.
```

### (D) 온디바이스 최적화 프롬프트

```
당신은 모델경량화 엔지니어입니다. 현재 PyTorch ckpt를 ONNX로 변환한 후,
CPU/모바일에서의 지연시간과 메모리를 줄이기 위한 후처리(양자화/연산 Fusion/입력크기 조정)를
벤치마크 테이블과 함께 제안하세요. 품질 저하를 최소화하는 설정을 우선하세요.
```

---

## ✅ 검수 체크리스트

* [ ] 데이터 폴더 규약 준수, 클래스당 ≥10장
* [ ] `ai-lens validate-data` 통과
* [ ] K-fold + 장소 Holdout 평가 OK
* [ ] Reject 임계치 튜닝 및 보고서 아카이브
* [ ] ONNX 내보내기 및 샘플 추론 성공
* [ ] README 배포 가이드 최신화
