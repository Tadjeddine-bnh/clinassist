import sys
from pathlib import Path
from config import settings
from knowledge_base.resources import get_whisper

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = settings.processed_dir    

MODEL_SIZE = settings.whisper_model      
LANGUAGE = "en"       # change to "ar" for Arabic, or remove for auto-detect


def main():
    if len(sys.argv) < 2:
        print('Usage: python transcribe.py path\\to\\audio.mp3')
        return

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model ({MODEL_SIZE})...")
    model = get_whisper()

    print(f"Transcribing {input_path.name}...")
    result = model.transcribe(str(input_path), language=LANGUAGE)
    text = result["text"].strip()

    output_path = PROCESSED_DIR / (input_path.stem + "_transcript.txt")
    output_path.write_text(text, encoding="utf-8")

    print(f"\n--- Transcript preview ---")
    print(text[:500])
    print(f"\nSaved to: {output_path}")
    print(f"Next step: re-run knowledge_base\\ingest.py")


if __name__ == "__main__":
    main()