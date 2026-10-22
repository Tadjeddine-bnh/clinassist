from functools import lru_cache
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from config import settings
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
@lru_cache
class LocalEmbedder(EmbeddingFunction):
    """Turns text into vectors. Same model as before, but we control how much it reads."""

    def __init__(self, model_name: str, max_tokens: int):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_tokens      # the "glasses" setting

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input)).tolist()


@lru_cache
def get_embedding_function():
    return LocalEmbedder(settings.embed_model, settings.embed_max_tokens)

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
@lru_cache
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.rerank_model)

def _words(text: str) -> list[str]:
    """Split text into lowercase words: 'Give Diazepam!' -> ['give', 'diazepam']"""
    return re.findall(r"\w+", text.lower())


@lru_cache
def get_bm25():
    """Keyword index over every chunk, built in memory from the database (about a second)."""
    from rank_bm25 import BM25Okapi
    data = get_collection().get(include=["documents", "metadatas"])
    docs = list(zip(data["documents"], data["metadatas"]))
    return BM25Okapi([_words(d) for d, _ in docs]), docs


def keyword_search(question: str, n: int) -> list[tuple[str, dict]]:
    bm25, docs = get_bm25()
    scores = bm25.get_scores(_words(question))
    best = sorted(range(len(docs)), key=lambda i: -scores[i])[:n]
    return [docs[i] for i in best]


def fuse(*ranked_lists, c: int = 60) -> list[tuple[str, dict]]:
    """Reciprocal Rank Fusion: each list gives a chunk 1/(c + rank) points; add them up."""
    points, items = {}, {}
    for ranked in ranked_lists:
        for rank, (doc, meta) in enumerate(ranked, 1):
            key = (meta["source"], meta["chunk_index"])
            points[key] = points.get(key, 0) + 1 / (c + rank)
            items[key] = (doc, meta)
    return [items[key] for key in sorted(points, key=lambda x: -points[x])]

def search(question: str, k: int) -> list[tuple[str, dict]]:
    """The one search function:
    1. meaning search (embedder)            - always
    2. keyword search (BM25), merged in      - if settings.hybrid
    3. reranker re-sorts the candidates      - if settings.rerank_model
    """
    n = max(settings.rerank_candidates, k) if (settings.hybrid or settings.rerank_model) else k
    r = get_collection().query(query_texts=[question], n_results=n)
    results = list(zip(r["documents"][0], r["metadatas"][0]))

    if settings.hybrid:
        results = fuse(results, keyword_search(question, n))

    if settings.rerank_model:
        scores = get_reranker().predict([(question, doc) for doc, _ in results])
        results = [res for _, res in sorted(zip(scores, results), key=lambda x: -x[0])]

    return results[:k]