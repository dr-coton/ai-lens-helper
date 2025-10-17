#!/usr/bin/env python3
"""Simple test script for YOLO+CLIP inference."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_lens_helper import Lens

def main():
    print("=== Testing YOLO+CLIP Inference ===\n")

    # Initialize
    print("1. Loading model...")
    lens = Lens(model_path="./indexes/경복궁_clip.json", device="cpu")
    print("   ✓ Model loaded\n")

    # Run inference
    print("2. Running inference...")
    result = lens.infer(
        place="경복궁",
        image_path="./data/경복궁/근정전/016.jpg",
        topk=5
    )
    print("   ✓ Inference complete\n")

    # Print results
    print("3. Results:")
    print(f"   Decision: {result.decision}")
    print(f"   Selected Item: {result.selected_item}")
    print(f"   Confidence: {result.confidence:.4f}")
    print(f"   Message: {result.message}")
    print(f"\n   Top 5 Predictions:")
    for i, pred in enumerate(result.predictions, 1):
        print(f"      {i}. {pred.item}: {pred.score:.4f}")

    if result.hints:
        print(f"\n   Hints:")
        for hint in result.hints:
            print(f"      - {hint}")

if __name__ == "__main__":
    main()
