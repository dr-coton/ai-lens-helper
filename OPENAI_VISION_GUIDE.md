# OpenAI Vision API 사용 가이드

## 🤖 Tab 3: OpenAI Vision

GUI의 세 번째 탭에서 OpenAI의 Vision API를 사용하여 전시품을 인식할 수 있습니다.

---

## 📋 기능

- OpenAI GPT-4 Vision 모델을 사용한 이미지 인식
- 학습 없이 즉시 사용 가능
- JSON 형식의 장소/건물 데이터로 쿼리
- 다양한 모델 선택 가능

---

## 🚀 사용 방법

### 1단계: OpenAI API Key 입력

1. GUI Tab 3 (🤖 OpenAI Vision) 선택
2. "API Key" 필드에 OpenAI API Key 입력
   - https://platform.openai.com/api-keys 에서 발급
   - 예: `sk-proj-...`
3. 모델 선택:
   - `gpt-5`: 최신 GPT-5 모델 (권장, 고성능)
   - `gpt-5-mini`: GPT-5 경량 버전 (빠르고 저렴)
   - `gpt-5-nano`: GPT-5 초경량 버전 (가장 빠름)
   - `gpt-4o`: GPT-4 Omni
   - `gpt-4o-mini`: GPT-4 경량 버전
   - `gpt-4-vision-preview`: 이전 GPT-4 Vision
4. GPT-5 전용 설정 (GPT-5 모델 선택 시):
   - **Reasoning**: 추론 강도 (minimal/low/medium/high)
     - `minimal`: 최소 추론 (가장 빠름)
     - `low`: 낮은 추론
     - `medium`: 중간 추론 (권장)
     - `high`: 높은 추론 (정확도 우선)
   - **Verbosity**: 응답 상세도 (low/medium/high)
     - `low`: 간결한 응답
     - `medium`: 적절한 응답 (권장)
     - `high`: 상세한 응답

### 2단계: 이미지 선택

1. "📷 이미지 선택" 버튼 클릭
2. 테스트할 이미지 파일 선택
3. 이미지가 미리보기에 표시됨

### 3단계: 장소/건물 데이터 편집 (선택)

좌측의 JSON 편집기에서 데이터를 수정할 수 있습니다:

```json
{
  "경복궁": ["근정전", "경회루", "향원정", "강녕전", "교태전"],
  "국립중앙박물관": ["금동미륵보살반가사유상", "백제금동대향로"],
  "덕수궁": ["석조전", "중명전"]
}
```

**포맷:**
```json
{
  "장소명": ["건물1", "건물2", "건물3"],
  "다른장소": ["건물A", "건물B"]
}
```

### 4단계: OpenAI Vision 실행

1. "🤖 OpenAI Vision 실행" 버튼 클릭
2. API 호출 대기 (수 초 소요)
3. 결과 확인

---

## 📊 결과 해석

### 성공 예시

```
============================================================
OpenAI Vision API 결과
============================================================
이미지: gyeongbokgung.jpg
모델: gpt-5

🏛️  장소: 경복궁
🏢  건물: 근정전
============================================================
```

팝업 메시지:
```
✓ 인식 성공
장소: 경복궁
건물: 근정전
```

### 실패 예시

```
🏛️  장소: 알 수 없음
🏢  건물: 알 수 없음
```

팝업 메시지:
```
⚠ 인식 실패
해당하는 전시품을 찾을 수 없습니다.
```

---

## 💡 장단점 비교

### OpenAI Vision vs YOLO+CLIP

| 항목 | OpenAI Vision | YOLO+CLIP |
|------|--------------|-----------|
| **학습 필요** | ❌ 불필요 | ✅ 필요 (인덱스 빌드) |
| **속도** | 🐢 느림 (API 호출) | 🚀 빠름 (로컬) |
| **비용** | 💰 유료 (API 호출당) | 💚 무료 (로컬 실행) |
| **정확도** | 🎯 높음 (범용 AI) | 📊 학습 데이터 품질에 의존 |
| **오프라인** | ❌ 인터넷 필요 | ✅ 완전 오프라인 가능 |
| **확장성** | ⚠️ API 제한 | ✅ 무제한 |
| **데이터 추가** | 🔄 즉시 (JSON 수정) | ⏳ 재학습 필요 |

### 언제 OpenAI Vision을 사용?

**추천 상황:**
- ✅ 빠른 프로토타입/테스트
- ✅ 학습 데이터가 없을 때
- ✅ 다양한 전시품을 즉시 추가/변경
- ✅ 소량의 추론만 필요

**비추천 상황:**
- ❌ 대량의 inference 필요 (비용)
- ❌ 오프라인 환경
- ❌ 실시간 응답 필요
- ❌ 데이터 프라이버시 중요

---

## 🔧 고급 사용법

### 커스텀 프롬프트 (코드)

Python에서 직접 사용:

```python
from pathlib import Path
from ai_lens_helper.infer.openai_vision import OpenAIVisionInference

# Initialize (GPT-5)
vision = OpenAIVisionInference(
    api_key="sk-proj-...",
    model="gpt-5",
    reasoning_effort="medium",  # minimal/low/medium/high
    verbosity="medium"  # low/medium/high
)

# Or GPT-4 (old models)
vision_gpt4 = OpenAIVisionInference(
    api_key="sk-proj-...",
    model="gpt-4o"
)

# Custom prompt
custom_prompt = """
이 이미지가 어떤 한국 전통 건축물인지 식별하고,
다음 JSON 형식으로 응답하세요:
{"건물명": "...", "시대": "...", "특징": "..."}
"""

# Run inference
result = vision.infer(
    image_path=Path("./photo.jpg"),
    places_data={"경복궁": ["근정전", "경회루"]},
    custom_prompt=custom_prompt  # Optional
)

print(result)
```

### 배치 처리 (여러 이미지)

```python
import json
from pathlib import Path
from ai_lens_helper.infer.openai_vision import OpenAIVisionInference

vision = OpenAIVisionInference(api_key="sk-proj-...")

places_data = {
    "경복궁": ["근정전", "경회루"],
    "덕수궁": ["석조전"]
}

results = []
for image_path in Path("./images").glob("*.jpg"):
    result = vision.infer(image_path, places_data)
    results.append({
        "image": image_path.name,
        **result
    })

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

---

## 💰 비용 예상

### OpenAI API 가격 (2025년 기준)

| 모델 | 입력 (per 1K tokens) | 예상 비용/이미지 | 특징 |
|------|---------------------|----------------|------|
| **GPT-5 모델** ||||
| gpt-5 | ~$0.015 | ~$0.015 | 최고 성능, 고급 추론 |
| gpt-5-mini | ~$0.005 | ~$0.005 | 균형잡힌 성능/비용 |
| gpt-5-nano | ~$0.0005 | ~$0.0005 | 초경량, 빠른 응답 |
| **GPT-4 모델** ||||
| gpt-4-vision-preview | $0.01 | ~$0.01 | 이전 버전 |
| gpt-4o | $0.0025 | ~$0.003 | GPT-4 Omni |
| gpt-4o-mini | $0.00015 | ~$0.0002 | 경량 버전 |

**참고:**
- 이미지 1장 ≈ 500-1000 tokens (해상도에 따라)
- 프롬프트 + JSON 데이터 ≈ 100-300 tokens

**예시:**
- 100장 inference with gpt-5-nano: ~$0.05
- 100장 inference with gpt-5-mini: ~$0.5
- 100장 inference with gpt-5: ~$1.5
- 1000장 inference with gpt-5: ~$15

**추천:**
- 프로토타입/테스트: `gpt-5-nano` (속도 우선)
- 일반 사용: `gpt-5-mini` (균형)
- 높은 정확도 필요: `gpt-5` + reasoning_effort="high"

---

## ⚠️ 주의사항

1. **API Key 보안**
   - API Key를 절대 공유하지 마세요
   - 코드에 하드코딩하지 마세요
   - 환경 변수 사용 권장

2. **Rate Limiting**
   - OpenAI API는 요청 제한이 있습니다
   - 대량 처리 시 딜레이 추가 필요

3. **데이터 프라이버시**
   - 이미지가 OpenAI 서버로 전송됩니다
   - 민감한 데이터는 YOLO+CLIP 사용 권장

4. **인터넷 연결**
   - 안정적인 인터넷 연결 필요
   - 오프라인 환경에서는 사용 불가

---

## 🐛 문제 해결

### "OpenAI package not installed"
```bash
pip install openai
```

### "Invalid API Key"
- API Key가 올바른지 확인
- https://platform.openai.com/api-keys 에서 재발급

### "Rate limit exceeded"
- 요청 속도를 줄이세요
- 유료 플랜으로 업그레이드

### "JSON parsing error"
- JSON 데이터 형식 확인
- 중괄호/대괄호/따옴표 오타 확인

---

## 📖 참고 자료

- OpenAI API 문서: https://platform.openai.com/docs
- Vision API 가이드: https://platform.openai.com/docs/guides/vision
- API 가격: https://openai.com/pricing

---

**준비 완료!** GUI Tab 3에서 OpenAI Vision을 테스트해보세요! 🎉
