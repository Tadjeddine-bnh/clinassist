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

settings = Settings()

if settings.ffmpeg_dir:
    os.environ["PATH"] += os.pathsep + str(settings.ffmpeg_dir)