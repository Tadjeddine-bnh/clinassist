"""
Phase 1: Knowledge base ingestion.

Reads every .pdf and .txt file from data/raw and data/processed, cleans the text,
splits it into overlapping chunks, and stores them in ChromaDB.

PDFs are read page by page with PyMuPDF so that:
- every chunk remembers its page number (for real citations, not invented ones)
- tables are kept as tables (rows intact) instead of a jumble of labels and numbers
"""

import hashlib
import re
import unicodedata
from pathlib import Path

import pymupdf

from config import settings
from knowledge_base.resources import get_bm25, get_collection

DATA_DIRS = [settings.raw_dir, settings.processed_dir]


def clean_text(text: str) -> str:
    """Repair common PDF extraction damage before chunking."""
    # 1. "ﬁ rst" -> "first": ligature characters followed by stray whitespace
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


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Tables come out as markdown with their rows intact."""
    pages = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            tables = page.find_tables().tables
            boxes = [pymupdf.Rect(t.bbox) for t in tables]
            # normal text blocks, skipping the ones inside a table (those come from the table below)
            text_parts = [
                block[4] for block in page.get_text("blocks")
                if block[6] == 0 and not any(pymupdf.Rect(block[:4]).intersects(box) for box in boxes)
            ]
            table_parts = [t.to_markdown() for t in tables]
            pages.append((page.number + 1, clean_text("\n".join(text_parts + table_parts))))
    return pages


def read_pdf(path: Path) -> str:
    """Whole PDF as one string (kept for code that just needs the text)."""
    return "\n".join(text for _, text in read_pdf_pages(path))


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = settings.chunk_size, overlap: int = settings.chunk_overlap) -> list[str]:
    """Simple sliding-window chunker."""
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


def index_pages(pages: list[tuple[int | None, str]], source: str) -> tuple[str, int]:
    """The ONLY place that writes to Chroma.

    pages: [(page_number, text), ...]. Use page_number None for files without pages (.txt).
    Chunks never cross a page boundary, so each chunk has exactly one page number.
    """
    chunks, metadatas = [], []
    for page, text in pages:
        for piece in chunk_text(text):
            meta = {"source": source, "chunk_index": len(chunks)}
            if page is not None:
                meta["page"] = page
            chunks.append(piece)
            metadatas.append(meta)

    full_text = "\n".join(text for _, text in pages)
    fingerprint = (f"{settings.chunk_size}:{settings.chunk_overlap}:{settings.embed_model}:"
                   f"{settings.embed_max_tokens}:{full_text}")
    doc_hash = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]

    col = get_collection()
    existing = col.get(where={"source": source}, include=["metadatas"])
    if existing["ids"] and existing["metadatas"][0].get("doc_hash") == doc_hash:
        return "unchanged", len(existing["ids"])

    if existing["ids"]:
        col.delete(ids=existing["ids"])
    if not chunks:
        return "empty", 0

    for meta in metadatas:
        meta["doc_hash"] = doc_hash
    ids = [hashlib.sha1(f"{source}:{doc_hash}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, metadatas=metadatas)
    get_bm25.cache_clear()   # keyword index is rebuilt on the next search
    return "indexed", len(chunks)


def index_document(text: str, source: str) -> tuple[str, int]:
    """Plain text with no pages (txt files, OCR, transcripts)."""
    return index_pages([(None, text)], source)


def load_documents() -> list[tuple[str, list[tuple[int | None, str]]]]:
    docs = []
    for data_dir in DATA_DIRS:
        for path in data_dir.glob("*"):
            # PDF already indexed: skip its OCR copy
            if path.name.endswith("_ocr.txt") and (settings.raw_dir / (path.name[:-8] + ".pdf")).exists():
                print(f"Skipped {path.name}: same text as the PDF")
                continue
            if path.suffix.lower() == ".pdf":
                docs.append((path.name, read_pdf_pages(path)))
            elif path.suffix.lower() == ".txt":
                docs.append((path.name, [(None, read_txt(path))]))
    return docs


def main():
    for d in DATA_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    if not docs:
        print(f"No .pdf or .txt files found in {DATA_DIRS}.")
        return

    for source_name, pages in docs:
        status, n = index_pages(pages, source_name)
        print(f"{status:9} {source_name}: {n} chunks")
    print(f"\nCollection holds {get_collection().count()} chunks.")


if __name__ == "__main__":
    main()