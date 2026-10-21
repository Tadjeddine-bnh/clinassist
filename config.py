import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    groq_api_key: str
    llm_model: str = "openai/gpt-oss-120b"
    embed_model: str = "paraphrase-multilingual-mpnet-base-v2"
    whisper_model: str = "base"

    chroma_dir: Path = BASE_DIR / "knowledge_base" / "chroma_db"
    collection_name: str = "clinical_kb"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    max_upload_mb: int = 25
    ffmpeg_dir: Path | None = None
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

settings = Settings()

if settings.ffmpeg_dir:
    os.environ["PATH"] += os.pathsep + str(settings.ffmpeg_dir)
    
    
import subprocess

# Langfuse reads these from the environment
for key in ("langfuse_public_key", "langfuse_secret_key", "langfuse_base_url"):
    if getattr(settings, key):
        os.environ.setdefault(key.upper(), getattr(settings, key))

# Which code version produced each trace, used to compare experiments in Step 3
try:
    GIT_COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=BASE_DIR).stdout.strip() or "unknown"
except Exception:
    GIT_COMMIT = "unknown"