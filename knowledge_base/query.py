"""
Phase 1 — Retrieval + grounded generation.

Retrieves the most relevant chunks from the clinical_kb collection and asks
the LLM to answer ONLY from those chunks, citing which source each fact came
from. This citation-grounding is the single most important habit to build
now, before you add OCR/CV/ASR tools in later phases — in a healthcare
context, an answer with no traceable source is a liability, not a feature.

Usage:
    python query.py "What is the recommended dosage for X in pediatric patients?"
"""

import sys

from groq import Groq
from config import settings
from knowledge_base.resources import get_collection
from pathlib import Path

DB_DIR = Path(__file__).parent / "chroma_db"
DB_DIR = settings.chroma_dir                    
COLLECTION_NAME = settings.collection_name       
EMBED_MODEL = settings.embed_model              
TOP_K = settings.top_k                           
GROQ_MODEL = settings.llm_model
SYSTEM_PROMPT = """You are a clinical knowledge assistant. You must answer ONLY using the
provided context chunks below. Follow these rules strictly:

1. If the context does not contain the answer, say so explicitly — do NOT guess or use
   outside knowledge. This matters more here than in any other domain.
2. Every factual claim in your answer must be followed by a citation tag like [source: filename].
3. Keep the answer concise and direct.
4. If chunks conflict with each other, point out the conflict rather than picking one silently.
"""


def retrieve(query: str, collection) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=TOP_K)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        return

    query = sys.argv[1]

    collection = get_collection()

    chunks = retrieve(query, collection)
    if not chunks:
        print("No documents in the knowledge base yet — run ingest.py first.")
        return

    prompt = build_prompt(query, chunks)

    groq_client = Groq(api_key=settings.groq_api_key) 
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # low temperature — this is a grounding task, not a creative one
    )

    print("\n--- Answer ---")
    print(response.choices[0].message.content)

    print("\n--- Retrieved sources (for your own debugging) ---")
    for c in chunks:
        print(f"- {c['source']}")


if __name__ == "__main__":
    main()