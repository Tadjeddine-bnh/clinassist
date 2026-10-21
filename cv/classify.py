"""
Phase 4 — Computer Vision classifier.

Takes a medical image and classifies it using a Vision Transformer (ViT)
pretrained on chest X-rays. Returns a label + confidence score.

The output is also saved as a .txt in data/processed/ so the agent
(Phase 5) can read the classification result and factor it into its
reasoning — same pattern as OCR and ASR outputs.

Usage:
    python classify.py path\\to\\image.jpg
"""

import sys
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = settings.processed_dir

# Pretrained ViT fine-tuned on chest X-rays (normal vs pneumonia)
# Source: nickmuchi/vit-finetuned-chest-xray-pneumonia on HuggingFace
MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"


def main():
    if len(sys.argv) < 2:
        print('Usage: python classify.py path\\to\\image.jpg')
        return

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model ({MODEL_NAME})...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    image = Image.open(input_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    labels = model.config.id2label
    results = [(labels[i], round(probs[i].item() * 100, 2)) for i in range(len(labels))]
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Classification results for {input_path.name} ---")
    for label, confidence in results:
        print(f"  {label}: {confidence}%")

    top_label, top_conf = results[0]
    summary = (
        f"Image: {input_path.name}\n"
        f"Classification: {top_label}\n"
        f"Confidence: {top_conf}%\n"
        f"All scores: {', '.join(f'{l}: {c}%' for l, c in results)}\n"
    )

    output_path = PROCESSED_DIR / (input_path.stem + "_classification.txt")
    output_path.write_text(summary, encoding="utf-8")

    print(f"\nSaved to: {output_path}")
    print(f"Next step: re-run knowledge_base\\ingest.py to make this queryable.")


if __name__ == "__main__":
    main()