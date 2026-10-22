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

import sys
from pathlib import Path 
 
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel 
from config import settings
from knowledge_base.resources import get_whisper

# Add project root to path so we can import from other phases
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.orchestrator import build_graph
from knowledge_base.ingest import read_pdf, read_txt

from knowledge_base.ingest import index_document

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from config import settings
from agent.orchestrator import build_graph
from knowledge_base.ingest import index_document, read_pdf, read_txt
from knowledge_base.resources import get_collection, get_whisper
from ocr.extract import extract_text
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from config import settings, GIT_COMMIT
from typing import Literal
from knowledge_base.ingest import index_document, index_pages, read_pdf_pages, read_txt
@asynccontextmanager
async def lifespan(app):
    print("Warming up models...")
    get_collection()
    yield

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RAW_DIR = settings.raw_dir
PROCESSED_DIR = settings.processed_dir    

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt",           # direct ingest
    ".jpg", ".jpeg", ".png",  # triggers OCR
    ".mp3", ".wav", ".m4a",   # triggers ASR
}

app = FastAPI(title="ClinAssist API", lifespan=lifespan)   # keep your existing title/args, add lifespan

# Allow all origins for local dev — restrict this in production

# Build the agent graph once at startup — not on every request
agent = build_graph()


# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    trace_id: str
    value: Literal[0, 1]          # 1 = helpful, 0 = not helpful; anything else gets a 422



class ChatResponse(BaseModel):
    question: str
    answer: str
    tools_used: list[str]
    trace_id: str | None = None

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
        _, chunks = index_pages(read_pdf_pages(file_path), file_path.name)
        return "PDF ingested", chunks

    if suffix == ".txt":
        text = read_txt(file_path)
        chunks = ingest_text(text, file_path.name)
        return f"Text file ingested directly", chunks

    # Image → OCR → ingest
    if suffix in {".jpg", ".jpeg", ".png"}:
        try:
            text = extract_text(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
        output_path = PROCESSED_DIR / (file_path.stem + "_ocr.txt")
        output_path.write_text(text, encoding="utf-8")
        chunks = ingest_text(text, output_path.name)
        return "Image processed via OCR", chunks
    # Audio → ASR → ingest
    if suffix in {".mp3", ".wav", ".m4a"}:
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
    q = request.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="chat") as span:
        span.update(
            input={"question": q},
            metadata={                      # the settings that produced this answer
                "llm_model": settings.llm_model,
                "embed_model": settings.embed_model,
                "top_k": settings.top_k,
                "chunk_size": settings.chunk_size,
                "git_commit": GIT_COMMIT,
            },
        )
        with propagate_attributes(tags=["api"]):
            handler = CallbackHandler()     # one per request, so trace IDs never mix
            result = agent.invoke(
                {"question": q, "tools_needed": [], "rag_context": "",
                 "tool_outputs": [], "final_answer": ""},
                config={"callbacks": [handler]},
            )
        span.update(output={"answer": result["final_answer"], "tools": result["tools_needed"]})
        trace_id = handler.last_trace_id

    return ChatResponse(question=q, answer=result["final_answer"],
                        tools_used=result["tools_needed"], trace_id=trace_id)
    
    
    
@app.post("/feedback")
def feedback(req: FeedbackRequest):
    get_client().create_score(trace_id=req.trace_id, name="user_feedback",
                              value=req.value, data_type="NUMERIC")
    return {"ok": True}

@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """Upload a file (PDF, image, audio, txt); it's extracted and ingested automatically."""
    # 1. Keep only the bare file name: "../../evil.txt" becomes "evil.txt"
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if not name or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Double-check the final path is really inside data/raw
    save_path = (RAW_DIR / name).resolve()
    if save_path.parent != RAW_DIR.resolve():
        raise HTTPException(400, "Bad filename")

    # 3. Never silently overwrite an existing file
    if save_path.exists():
        raise HTTPException(409, f"'{name}' already exists")

    # 4. Size limit: read at most max+1 bytes, so a huge upload can't fill memory
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
    save_path.write_bytes(data)

    try:
        message, chunks = process_uploaded_file(save_path)
    except Exception:
        save_path.unlink(missing_ok=True)
        raise
    return IngestResponse(filename=name, chunks_added=chunks, message=message)


# Serve the chat UI from the same address as the API. Must stay LAST.
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
