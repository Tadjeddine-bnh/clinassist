"""
Phase 4: chest X-ray classifier (Vision Transformer fine-tuned on NORMAL vs PNEUMONIA).

classify_image() is used by the agent's classify_xray tool.
Running this file directly classifies one image and prints the result.

Usage:
    uv run python -m cv.classify data\raw\pneumo.jpg
"""
import sys
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"


@lru_cache
def _load():
    """Load the model once, then reuse it."""
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def classify_image(path) -> list[tuple[str, float]]:
    """Return [(label, percent), ...], highest first."""
    processor, model = _load()
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    results = [(model.config.id2label[i], round(probs[i].item() * 100, 2)) for i in range(len(probs))]
    return sorted(results, key=lambda r: r[1], reverse=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cv.classify path\\to\\image.jpg")
        return
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return
    print(f"--- Classification results for {path.name} ---")
    for label, confidence in classify_image(path):
        print(f"  {label}: {confidence}%")


if __name__ == "__main__":
    main()