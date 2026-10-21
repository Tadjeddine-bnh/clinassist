"""
Phase 6 — FastAPI Backend.

Wraps the entire ClinAssist pipeline into a REST API.

Endpoints:
  POST /chat      — send a question, get a cited answer from the agent
  POST /ingest    — upload a file (PDF, image, audio, txt), auto-ingest it
  GET  /health    — confirms the service is running

Run with:
    uvicorn api.main:app --reload --port 8000

Or from the clinassist root:
    python -m uvicorn api.main:app --reload --port 8000

Then test at: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import os
import sys
import shutil
import uuid
from pathlib import Path 
 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from config import settings
from knowledge_base.resources import get_whisper

# Add project root to path so we can import from other phases
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.orchestrator import build_graph
from knowledge_base.ingest import load_documents, chunk_text, read_pdf, read_txt

import chromadb
from chromadb.utils import embedding_functions
import uuid as _uuid
from knowledge_base.ingest import index_document

from contextlib import asynccontextmanager
from knowledge_base.resources import get_collection

@asynccontextmanager
async def lifespan(app):
    print("Warming up models...")
    get_collection()           # embedder + DB, loaded once
    yield


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RAW_DIR = settings.raw_dir
PROCESSED_DIR = settings.processed_dir           
DB_DIR = settings.chroma_dir                     
COLLECTION_NAME = settings.collection_name       
EMBED_MODEL = settings.embed_model 

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt",           # direct ingest
    ".jpg", ".jpeg", ".png",  # triggers OCR
    ".mp3", ".wav", ".m4a",   # triggers ASR
}

app = FastAPI(title="ClinAssist API", lifespan=lifespan)   # keep your existing title/args, add lifespan

# Allow all origins for local dev — restrict this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent graph once at startup — not on every request
agent = build_graph()


# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    tools_used: list[str]

class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def ingest_text(text: str, source: str) -> int:
    _, n = index_document(text, source)
    return n

def process_uploaded_file(file_path: Path) -> tuple[str, int]:
    """
    Process an uploaded file based on its type.
    Returns (status_message, chunks_added).
    """
    suffix = file_path.suffix.lower()

    # Direct text ingestion
    if suffix == ".pdf":
        text = read_pdf(file_path)
        chunks = ingest_text(text, file_path.name)
        return f"PDF ingested directly", chunks

    if suffix == ".txt":
        text = read_txt(file_path)
        chunks = ingest_text(text, file_path.name)
        return f"Text file ingested directly", chunks

    # Image → OCR → ingest
    if suffix in {".jpg", ".jpeg", ".png"}:
        import pymupdf as fitz
        # Try PyMuPDF first (works for digital PDFs saved as images)
        # Fall back to a note to run OCR manually
        output_path = PROCESSED_DIR / (file_path.stem + "_ocr.txt")
        try:
            doc = fitz.open(str(file_path))
            text = "\n\n".join(page.get_text() for page in doc)
            if text.strip():
                output_path.write_text(text, encoding="utf-8")
                chunks = ingest_text(text, output_path.name)
                return "Image processed via text extraction", chunks
        except Exception:
            pass
        # If no text layer, use EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(str(file_path), detail=0)
            text = "\n".join(results)
            output_path.write_text(text, encoding="utf-8")
            chunks = ingest_text(text, output_path.name)
            return "Image processed via EasyOCR", chunks
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # Audio → ASR → ingest
    if suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
        try:
            model = get_whisper()
            result = model.transcribe(str(file_path), language="en")
            text = result["text"].strip()
            output_path = PROCESSED_DIR / (file_path.stem + "_transcript.txt")
            output_path.write_text(text, encoding="utf-8")
            chunks = ingest_text(text, output_path.name)
            return "Audio transcribed via Whisper", chunks
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ClinAssist API"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a clinical question to the agent.
    The agent decides which tools to use and returns a cited answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = agent.invoke({
        "question": request.question,
        "tools_needed": [],
        "rag_context": "",
        "tool_outputs": [],
        "final_answer": "",
    })

    return ChatResponse(
        question=request.question,
        answer=result["final_answer"],
        tools_used=result["tools_needed"],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """
    Upload a file (PDF, image, audio, txt).
    The API automatically extracts text and ingests it into the knowledge base.
    No manual ingest.py run needed.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Save uploaded file to data/raw/
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DIR / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    message, chunks = process_uploaded_file(save_path)

    return IngestResponse(
        filename=file.filename,
        chunks_added=chunks,
        message=message,
    )