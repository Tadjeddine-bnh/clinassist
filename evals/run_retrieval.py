"""Retrieval eval: for each golden question, where does the evidence passage rank?

No LLM calls. Uses the same search as the app (settings.top_k decides what the LLM sees).
Every run is appended to evals/results/retrieval.jsonl so runs can be compared later.
"""
import json
from datetime import datetime
from pathlib import Path

from config import settings, GIT_COMMIT
from knowledge_base.resources import get_collection
from knowledge_base.resources import search

def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main():
    lines = Path("evals/golden.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    items = [i for i in items if i.get("evidence")]   # refusal / x-ray questions have nothing to find
 

    ranks = []
    for it in items:
        results = search(it["question"], k=10)
        docs = [norm(doc) for doc, _ in results]
        ev = [norm(e) for e in it["evidence"]]
        rank = next((i + 1 for i, d in enumerate(docs) if any(e in d for e in ev)), None)
        ranks.append(rank)
        flag = "OK  " if rank and rank <= settings.top_k else "MISS"
        print(f"{flag} rank={str(rank or '-'):>2}  {it['id']:8} {it['question'][:60]}")

    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    summary = {
        "hit@1": round(hit(1), 3),
        f"hit@{settings.top_k}": round(hit(settings.top_k), 3),
        "hit@10": round(hit(10), 3),
        "mrr": round(mrr, 3),
        "n": n,
    }
    print("\n", summary)

    out = Path("evals/results/retrieval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "commit": GIT_COMMIT,
            "chunk_size": settings.chunk_size,
            "top_k": settings.top_k,
            "embed_model": settings.embed_model,
            **summary,
        }) + "\n")


if __name__ == "__main__":
    main()