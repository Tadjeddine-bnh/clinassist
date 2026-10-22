import hashlib
from pathlib import Path

from pypdf import PdfReader
import re
import unicodedata

from config import settings
from knowledge_base.resources import get_collection

DATA_DIRS = [settings.raw_dir, settings.processed_dir]

def clean_text(text: str) -> str:
    """Repair common PDF extraction damage before chunking."""
    # 1. "ﬁ rst" -> "first": ligature characters followed by a stray space
    text = re.sub(r"([\ufb00-\ufb04])\s+(?=[a-z])", r"\1", text)
    text = unicodedata.normalize("NFKC", text)               # ﬁ -> fi, odd spaces -> normal
    # 2. bullet symbols the PDF turned into codes like /L50776
    text = re.sub(r"/L\d{4,6}", " - ", text)
    # 3. page headers/footers: "PB2_ch5.indd 134" and "5/06/13 10:06 AM"
    text = re.sub(r"PB2_\S+?\.indd \d+", " ", text)
    text = re.sub(r"\d{1,2}/\d{2}/\d{2} \d{1,2}:\d{2} [AP]M", " ", text)
    # 4. letter-spaced words: "F e v e r" -> "Fever"
    text = re.sub(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)
    return text


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = settings.chunk_size, overlap: int = settings.chunk_overlap) -> list[str]:
    """Simple sliding-window chunker (unchanged)."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def index_document(text: str, source: str) -> tuple[str, int]:
    chunks = chunk_text(text)
    col = get_collection()
    doc_hash = hashlib.sha1(text.encode()).hexdigest()[:12]   # fingerprint of the content

    existing = col.get(where={"source": source}, include=["metadatas"])
    if existing["ids"] and existing["metadatas"][0].get("doc_hash") == doc_hash:
        return "unchanged", len(existing["ids"])

    if existing["ids"]:
        col.delete(ids=existing["ids"])
    if not chunks:
        return "empty", 0

    ids = [hashlib.sha1(f"{source}:{doc_hash}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    col.add(
        ids=ids,
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i, "doc_hash": doc_hash} for i in range(len(chunks))],
    )
    return "indexed", len(chunks)

def load_documents() -> list[tuple[str, str]]:
    docs = []
    for data_dir in DATA_DIRS:
        for path in data_dir.glob("*"):
            # NEW: skip "X_ocr.txt" when "X.pdf" exists, because the PDF is already indexed
            if path.name.endswith("_ocr.txt") and (settings.raw_dir / (path.name[:-8] + ".pdf")).exists():
                print(f"Skipped {path.name}: same text as the PDF")
                continue
            if path.suffix.lower() == ".pdf":
                docs.append((path.name, read_pdf(path)))
            elif path.suffix.lower() == ".txt":
                docs.append((path.name, read_txt(path)))
    return docs


def main():
    for d in DATA_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    if not docs:
        print(f"No .pdf or .txt files found in {DATA_DIRS}.")
        return

    # CHANGED: no embedder/client/collection here, no uuid4, no upsert.
    # index_document does all of that, the same way the API will.
    total_chunks = 0
    for source_name, text in docs:
        status, n = index_document(text, source_name)
        print(f"{status:9} {source_name}: {n} chunks")
    print(f"\nCollection holds {get_collection().count()} chunks.")


if __name__ == "__main__":
    main()