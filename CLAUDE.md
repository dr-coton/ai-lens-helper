# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ai-lens-helper** is a hybrid classification + retrieval toolkit for museum exhibit identification. The system accepts `(place, image)` pairs and returns exhibit predictions with confidence scores, recommending "recollect" when confidence is below threshold.

The project is currently in early development. The core infrastructure is in place with a color-based inference prototype that validates the end-to-end workflow. The planned architecture includes deep learning models (ViT-B/16, ConvNeXt), metric learning heads (ArcFace/Triplet), and FAISS/ScaNN indexing for production deployment.

## Development Commands

### Installation
```bash
pip install -e .           # Install package in editable mode
pip install -e ".[dev]"    # Install with development dependencies
```

### Testing
```bash
pytest                     # Run all tests
pytest tests/test_data_validator.py  # Run specific test file
pytest -v                  # Verbose output
pytest --cov               # Run with coverage
```

### Code Quality
```bash
black .                    # Format code
ruff check .               # Lint code
mypy src/                  # Type check
```

### CLI Commands

The project provides an `ai-lens` CLI (defined in src/ai_lens_helper/cli.py):

```bash
# Validate dataset structure
ai-lens validate-data ./data --min-images 10

# Build color-based retrieval index (current prototype)
ai-lens build-index --model model.json --data-root ./data --place A --save ./A.index

# Single image inference
ai-lens infer --model ./A.index --place A --image ./sample.jpg --reject-threshold 0.62 --topk 3

# Batch inference (stub)
ai-lens infer-batch --model ./runs/model.ckpt --place A --input-dir ./captures/A --output ./results.jsonl

# Training (stub)
ai-lens train --data-root ./data --output-dir ./runs

# Export (stub)
ai-lens export --ckpt ./runs/best.ckpt --onnx ./model.onnx --opset 17
```

## Architecture

### Package Structure
```
src/ai_lens_helper/
  __init__.py              # SDK facade (Lens class)
  cli.py                   # Typer-based CLI entry point
  config/                  # Configuration loading (planned)
  data/                    # Dataset utilities
  models/                  # Backbones, heads, losses (stubs)
    backbones.py
    heads.py
    losses.py
  train/                   # Training pipeline
    datamodule.py          # DataValidator for dataset layout checking
    engine.py              # TrainingEngine (stub)
    evaluator.py           # Evaluation metrics (stub)
    export.py              # Model export to ONNX/TorchScript (stub)
  infer/                   # Inference pipeline
    runner.py              # InferenceRunner (color-based prototype)
    index.py               # IndexBuilder for retrieval index
    postprocess.py         # Decision logic (stub)
    quality_check.py       # Image quality hints (stub)
  utils/                   # Shared utilities
    io.py                  # JSON/YAML I/O
    image.py               # Image loading and color extraction
    metrics.py             # Evaluation metrics (stub)
    vis.py                 # Visualization (stub)
```

### Dataset Layout Convention

The project expects data organized as:
```
./data/
  {place}/           # Museum location/room (e.g., "A", "B", "C")
    {item}/          # Exhibit identifier (e.g., "ㄱ", "ㄴ", "ㄷ")
      *.jpg          # Images of this exhibit (≥10 recommended)
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.ppm`

Use `ai-lens validate-data` to check compliance before training.

### Current Inference Implementation

The current InferenceRunner (src/ai_lens_helper/infer/runner.py) is a **color-based prototype** that:
1. Loads a JSON index file containing mean RGB values for each exhibit
2. Computes mean RGB of the input image
3. Ranks exhibits by Euclidean distance in normalized RGB space
4. Returns "accept" if top score ≥ threshold, else "recollect" with hints

This validates the end-to-end workflow (CLI → SDK → inference → structured output) while the ML components are being developed.

### Planned Production Architecture

Per PLAN.md, the production system will use:
- **Backbone**: ViT-B/16 (default) or ConvNeXt-Tiny with transfer learning
- **Dual heads**: (A) Softmax classification + (B) Metric embedding (ArcFace/Triplet)
- **Decision logic**: Hybrid scoring combining softmax confidence, margin, and embedding distance
- **Indexing**: FAISS/ScaNN for efficient nearest-neighbor search
- **Loss**: Multi-objective `CE + ArcFace/Triplet`
- **Augmentation**: RandomResizedCrop, ColorJitter, HFlip, GaussianBlur, RandomGray

## Key Design Patterns

### SDK Facade
The `Lens` class (src/ai_lens_helper/__init__.py) provides a minimal public API for Python integration:
```python
from ai_lens_helper import Lens

lens = Lens(model_path="./model.index")
result = lens.infer(place="A", image_path="./sample.jpg", reject_threshold=0.62)
print(result.decision, result.selected_item, result.confidence)
```

### Inference Result Structure
All inference returns an `InferenceResult` dataclass with:
- `place`: Location identifier
- `predictions`: List of `Prediction(item, score)` sorted by score
- `decision`: "accept" or "recollect"
- `selected_item`: Top prediction if accepted
- `confidence`: Top score if accepted
- `message`: Human-readable explanation
- `hints`: List of improvement suggestions (for "recollect")

### Threshold Resolution Priority
Reject thresholds are resolved in this order:
1. Explicit parameter in `infer()` call
2. Config file (`reject_threshold` or `reject.threshold`)
3. Index metadata (`metadata.reject_threshold`)
4. Default fallback: 0.62

## Configuration

The project uses YAML configuration (see README for example configs/default.yaml):
- Seed, image size, backbone, optimizer, scheduler
- Augmentation pipeline settings
- Reject threshold and margin parameters
- Export options (ONNX opset, quantization)

Config loading is implemented in utils/io.py via `load_yaml_config()`.

## Testing Strategy

Tests use pytest with fixtures in tests/conftest.py:
- `test_data_validator.py`: Dataset layout validation logic
- `test_inference_workflow.py`: End-to-end color-based inference

When adding tests:
- Place fixtures in `tests/conftest.py` for reuse
- Use `tmp_path` fixture for file system operations
- Mock external dependencies (future PyTorch models, FAISS indices)

## Important Implementation Notes

1. **Early Development Stage**: Many modules (models/, train/engine.py, train/evaluator.py, train/export.py, infer/postprocess.py, infer/quality_check.py) are stubs awaiting implementation per PLAN.md.

2. **Color Prototype**: The current inference uses mean RGB matching as a placeholder. Do not optimize this approach; it will be replaced with the planned deep learning pipeline.

3. **Image Normalization**: The runner automatically normalizes RGB values to [0,1] range if detected as [0,255].

4. **Place-Scoped Inference**: All inference requires a `place` parameter to narrow the search space. The system does not perform cross-place ranking.

5. **Error Codes**: The project defines error codes in README (E100-E310) for common failures. Match these in exception messages.

6. **Korean Language**: README and PLAN.md are in Korean, reflecting the target deployment environment. Code, comments, and variable names remain in English.

7. **Test Path Setup**: Tests modify sys.path in conftest.py to import from src/. This is required because the project uses src-layout packaging.

## Next Development Phases

Based on PLAN.md timeline:
1. **W1**: Scaffolding/data validation/loaders (✓ mostly complete)
2. **W2**: Training engine (classification + embedding), full CLI
3. **W3**: Evaluation/reporting/reject logic refinement
4. **W4**: ONNX export/indexing/documentation

Current status: In W1/W2 transition. Data validation and color-based inference prototype are complete. Next priorities are implementing the training engine with real model backbones and the index builder with proper embedding computation.

## References

- PLAN.md contains detailed architecture decisions and AI prompts for optimization tasks
- README.md provides user-facing documentation and CLI examples
- pyproject.toml defines all dependencies and tool configurations
