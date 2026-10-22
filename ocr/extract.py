"""Phase 2: text extraction from documents and images."""
import sys
from pathlib import Path

import pymupdf

from config import settings
from knowledge_base.resources import get_ocr_reader


def extract_text(path) -> str:
    """Text layer first (fast, exact). If empty, run real OCR with EasyOCR."""
    try:
        with pymupdf.open(str(path)) as doc:
            text = "\n\n".join(page.get_text() for page in doc)
        if text.strip():
            return text
    except Exception:
        pass
    return "\n".join(get_ocr_reader().readtext(str(path), detail=0))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ocr.extract <file>")
        return
    input_path = Path(sys.argv[1])
    text = extract_text(input_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.processed_dir / (input_path.stem + "_ocr.txt")
    output_path.write_text(text, encoding="utf-8")
    print("--- Extracted text preview ---")
    print(text[:500])
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()