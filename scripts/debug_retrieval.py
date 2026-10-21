from knowledge_base.resources import get_collection

col = get_collection()
q = "What are the signs of severe dehydration in children?"

hits = col.get(where_document={"$contains": "severe dehydration"})
print("Chunks containing 'severe dehydration':", len(hits["ids"]))

r = col.query(query_texts=[q], n_results=10, include=["metadatas", "distances", "documents"])
for m, d, doc in zip(r["metadatas"][0], r["distances"][0], r["documents"][0]):
    print(f"{d:.3f}  {m['source'][:35]:35}  #{m['chunk_index']:<5} {doc[:70]!r}")