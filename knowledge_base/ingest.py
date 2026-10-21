import hashlib
from pathlib import Path

from pypdf import PdfReader

from config import settings
from knowledge_base.resources import get_collection

DATA_DIRS = [settings.raw_dir, settings.processed_dir]


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


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