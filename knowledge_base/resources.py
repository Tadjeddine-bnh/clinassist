from functools import lru_cache
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from config import settings

@lru_cache
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model
    )

@lru_cache
def get_collection():
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),   # removes the telemetry spam
    )
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=get_embedding_function(),
    )

@lru_cache
def get_whisper():
    import whisper
    return whisper.load_model(settings.whisper_model)