import sys
from knowledge_base.resources import get_collection

words = [w.lower() for w in sys.argv[1:]]
data = get_collection().get(include=["documents", "metadatas"])

hits = [(d, m) for d, m in zip(data["documents"], data["metadatas"])
        if all(w in d.lower() for w in words)]

print(f"{len(hits)} chunks contain all of {words}")
for doc, m in hits[:8]:
    i = doc.lower().find(words[0])
    print(f"- {m['source'][:40]} #{m['chunk_index']}: ...{doc[max(0, i - 80): i + 160]}...")