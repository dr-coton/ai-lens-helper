"""OpenAI Vision API를 사용한 전시품 인식."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIVisionInference:
    """OpenAI Vision API를 사용한 전시품 식별."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5"
    ):
        """
        Initialize OpenAI Vision inference.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gpt-4o-mini)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def infer(
        self,
        image_path: Path,
        place: str,
        items: list,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Run inference using OpenAI Vision API.

        Args:
            image_path: Path to image file
            place: Place name (e.g., "경복궁")
            items: List of items at this place (e.g., ["근정전", "경회루", "향원정"])
            custom_prompt: Optional custom prompt (overrides default)

        Returns:
            Dictionary with "장소" and "건물" keys
            Example: {"장소": "경복궁", "건물": "근정전", "index": 0}
        """
        # Start timing
        start_time = time.time()

        # Encode image
        image_base64 = self._encode_image(image_path)

        # Build prompt
        if custom_prompt is None:
            prompt = self._build_default_prompt(place, items)
        else:
            prompt = custom_prompt

        # Call OpenAI API
        try:
            api_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            )
            api_time = time.time() - api_start

            # Parse response
            content = response.choices[0].message.content.strip()

            # Calculate total time
            total_time = time.time() - start_time

            # Get token usage and calculate cost
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculate cost based on model
            cost = self._calculate_cost(input_tokens, output_tokens)

            # Try to parse as integer
            try:
                # Try to parse as plain integer first
                idx = int(content.strip())

                # Convert index to item name
                result = {"index": idx}
                if idx == -1:
                    result["장소"] = place
                    result["건물"] = "알 수 없음"
                elif 0 <= idx < len(items):
                    result["장소"] = place
                    result["건물"] = items[idx]
                else:
                    result["장소"] = place
                    result["건물"] = "알 수 없음 (잘못된 인덱스)"

                result["_metadata"] = {
                    "api_time": round(api_time, 2),
                    "total_time": round(total_time, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": round(cost, 6)
                }
                return result
            except ValueError:
                # Fallback: try JSON parsing
                try:
                    result = json.loads(content)
                    if "index" in result:
                        idx = result["index"]
                        if idx == -1:
                            result["장소"] = place
                            result["건물"] = "알 수 없음"
                        elif 0 <= idx < len(items):
                            result["장소"] = place
                            result["건물"] = items[idx]
                        else:
                            result["장소"] = place
                            result["건물"] = "알 수 없음 (잘못된 인덱스)"

                    result["_metadata"] = {
                        "api_time": round(api_time, 2),
                        "total_time": round(total_time, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "cost_usd": round(cost, 6)
                    }
                    return result
                except json.JSONDecodeError:
                    # Fallback: return raw text
                    return {
                        "raw_response": content,
                        "_metadata": {
                            "api_time": round(api_time, 2),
                            "total_time": round(total_time, 2),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cost_usd": round(cost, 6)
                        }
                    }

        except Exception as e:
            # Handle encoding issues with exception messages
            try:
                error_msg = str(e)
            except:
                error_msg = repr(e)
            return {
                "error": error_msg,
                "_metadata": {
                    "total_time": round(time.time() - start_time, 2)
                }
            }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on model and token usage.

        Pricing (per 1M tokens):
        - GPT-5: Input $1.25, Output $10.00
        - GPT-5-mini: Input $0.25, Output $2.00
        - GPT-5-nano: Input $0.05, Output $0.40
        - GPT-4o: Input $2.50, Output $10.00
        - GPT-4o-mini: Input $0.15, Output $0.60
        """
        pricing = {
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
        }

        # Get pricing for current model (default to gpt-4o if unknown)
        model_pricing = pricing.get(self.model, pricing["gpt-4o"])

        # Calculate cost (price is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def _build_default_prompt(self, place: str, items: list) -> str:
        """Build default prompt with place and items list."""
        # Build numbered list
        items_list = "\n".join([f"{i}. {item}" for i, item in enumerate(items)])

        prompt = f"""
이미지를 분석하고, 아래 리스트 중 어떤 항목에 해당하는지 인덱스 번호만 반환하세요.

장소: {place}
항목 리스트:
{items_list}

규칙:
1. 반드시 숫자 하나만 응답하세요.
2. 0부터 시작하는 정수입니다.
3. 해당하는 항목이 없으면 -1을 반환하세요.
4. JSON, 텍스트, 설명 등 다른 내용을 포함하지 마세요.
5. 오직 숫자 하나만 반환하세요.

예시: 0 또는 1 또는 2 또는 -1
"""
        return prompt
