"""Check that every evidence phrase in evals/golden.jsonl exists in the knowledge base.

0 matches   -> the phrase doesn't exist as written: fix it with scripts/find_evidence.py
>20 matches -> the phrase is too generic: retrieval will look better than it really is
"""
import json
from pathlib import Path

from knowledge_base.resources import get_collection


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main():
    lines = Path("evals/golden.jsonl").read_text(encoding="utf-8").splitlines()
    items = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Line {n} is broken: {e}")
            print(f"  starts with: {line[:100]}")
            return
    docs = [norm(d) for d in get_collection().get(include=["documents"])["documents"]]

    problems = 0
    for it in items:
        for phrase in it.get("evidence", []):
            n = sum(norm(phrase) in d for d in docs)
            if n == 0:
                status, problems = "FIX  (not found)", problems + 1
            elif n > 20:
                status, problems = "WEAK (too generic)", problems + 1
            else:
                status = "ok"
            print(f"{status:18} {n:>4} chunks  {it['id']:8} '{phrase}'")

    print(f"\n{len(items)} questions, {problems} evidence phrases need attention")


if __name__ == "__main__":
    main()