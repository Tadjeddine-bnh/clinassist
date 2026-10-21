import sys
from pathlib import Path
from config import settings
import fitz  # pymupdf

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = settings.processed_dir

def main():
    if len(sys.argv) < 2:
        print('Usage: python extract.py path\\to\\document.pdf')
        return

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(input_path))
    text = "\n\n".join(page.get_text() for page in doc)

    output_path = PROCESSED_DIR / (input_path.stem + "_ocr.txt")
    output_path.write_text(text, encoding="utf-8")

    print(f"--- Extracted text preview ---")
    print(text[:500])
    print(f"\nSaved to: {output_path}")
    print(f"Next step: re-run knowledge_base\\ingest.py")

if __name__ == "__main__":
    main()