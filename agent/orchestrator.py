"""
Phase 5 — Agent Orchestrator.

A LangGraph state machine that receives a clinician's question,
decides which tools to call (RAG, OCR, ASR, CV), calls them,
and composes a final cited answer.

Usage:
    python orchestrator.py "What does the chest xray show and is it consistent with WHO guidelines?"
    python orchestrator.py "Summarize John Smith's discharge summary"
    python orchestrator.py "What did the clinician dictate in the audio note?"
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated
import operator
from config import settings
from config import settings
from knowledge_base.resources import get_collection
 

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langfuse import get_client

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = settings.processed_dir         
DB_DIR = settings.chroma_dir                    
COLLECTION_NAME = settings.collection_name       
EMBED_MODEL = settings.embed_model               
GROQ_MODEL = settings.llm_model                 

# ─────────────────────────────────────────
# STATE
# Every node reads from and writes to this.
# ─────────────────────────────────────────
class AgentState(TypedDict):
    question: str                          # original clinician question
    tools_needed: list[str]                # router decision: ["rag"], ["rag","cv"], etc.
    rag_context: str                       # retrieved chunks from ChromaDB
    tool_outputs: Annotated[list[str], operator.add]  # outputs from OCR/ASR/CV tools
    final_answer: str                      # composed answer


# ─────────────────────────────────────────
# TOOLS
# Each tool is a plain function. The agent
# calls them as needed — they don't know
# about the agent, the agent knows about them.
# ─────────────────────────────────────────

def tool_rag(question: str) -> str:
    """Retrieve relevant chunks from the clinical knowledge base."""
    collection = get_collection()
    with get_client().start_as_current_observation(as_type="span", name="retrieval") as span:
        results = collection.query(
            query_texts=[question], n_results=settings.top_k,
            include=["documents", "metadatas", "distances"],
        )
        span.update(
            input={"question": question, "top_k": settings.top_k},
            output=[
                {"source": m["source"], "chunk": m["chunk_index"], "distance": round(d, 3), "preview": doc[:150]}
                for doc, m, d in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ],
        )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(f"[source: {meta['source']}]\n{doc}")
    return "\n\n".join(chunks)


def tool_cv(image_path: str = None) -> str:
    """Run CV classifier on the most recent image in data/processed/ if no path given."""

    # Find the most recent classification result already saved
    classification_files = list(PROCESSED_DIR.glob("*_classification.txt"))
    if classification_files:
        latest = max(classification_files, key=lambda f: f.stat().st_mtime)
        return latest.read_text(encoding="utf-8")

    return "No CV classification results found. Run cv/classify.py on an image first."


def tool_asr() -> str:
    """Return the most recent ASR transcript from data/processed/."""
    transcript_files = list(PROCESSED_DIR.glob("*_transcript.txt"))
    if transcript_files:
        latest = max(transcript_files, key=lambda f: f.stat().st_mtime)
        return f"[source: {latest.name}]\n{latest.read_text(encoding='utf-8')}"
    return "No transcripts found. Run asr/transcribe.py on an audio file first."


def tool_ocr() -> str:
    """Return the most recent OCR extraction from data/processed/."""
    ocr_files = list(PROCESSED_DIR.glob("*_ocr.txt"))
    if ocr_files:
        latest = max(ocr_files, key=lambda f: f.stat().st_mtime)
        return f"[source: {latest.name}]\n{latest.read_text(encoding='utf-8')}"
    return "No OCR extractions found. Run ocr/extract.py on a document first."


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────
llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=settings.groq_api_key)


def node_router(state: AgentState) -> AgentState:
    """
    Node 1 — Router.
    Asks the LLM to decide which tools are needed based on the question.
    Returns a list like ["rag"], ["rag", "cv"], ["rag", "asr", "ocr"], etc.
    """
    prompt = f"""You are a clinical AI assistant router. Given a clinician's question,
decide which tools are needed to answer it. Respond with ONLY a comma-separated list
of tool names, nothing else.

Available tools:
- rag: search the clinical knowledge base (always include this)
- cv: get computer vision classification result for a medical image
- asr: get the latest audio transcription/dictation
- ocr: get the latest OCR-extracted document text

Examples:
"What does the chest xray show?" → rag,cv
"What did the doctor dictate?" → rag,asr
"Summarize the discharge summary" → rag,ocr
"What is the treatment for pneumonia?" → rag
"Analyze the xray and compare with the audio note" → rag,cv,asr

Question: {state['question']}
Tools needed:"""

    response = llm.invoke(prompt)
    raw = response.content.strip().lower()
    tools = [t.strip() for t in raw.split(",") if t.strip() in ["rag", "cv", "asr", "ocr"]]
    if not tools:
        tools = ["rag"]  # always fall back to RAG

    print(f"\n[Router] Tools selected: {tools}")
    return {**state, "tools_needed": tools}


def node_tool_caller(state: AgentState) -> AgentState:
    """
    Node 2 — Tool caller.
    Calls each selected tool and collects their outputs.
    RAG output goes to rag_context, everything else to tool_outputs.
    """
    rag_context = ""
    tool_outputs = []

    for tool in state["tools_needed"]:
        if tool == "rag":
            print("[Tool] Running RAG retrieval...")
            rag_context = tool_rag(state["question"])
        elif tool == "cv":
            print("[Tool] Running CV classifier...")
            tool_outputs.append("=== CV Classification ===\n" + tool_cv())
        elif tool == "asr":
            print("[Tool] Reading ASR transcript...")
            tool_outputs.append("=== ASR Transcript ===\n" + tool_asr())
        elif tool == "ocr":
            print("[Tool] Reading OCR extraction...")
            tool_outputs.append("=== OCR Document ===\n" + tool_ocr())

    return {**state, "rag_context": rag_context, "tool_outputs": tool_outputs}


def node_answer_generator(state: AgentState) -> AgentState:
    """
    Node 3 — Answer generator.
    Combines RAG context + tool outputs and generates a final cited answer.
    """
    tool_section = "\n\n".join(state["tool_outputs"]) if state["tool_outputs"] else ""

    prompt = f"""You are a clinical AI assistant. Answer the clinician's question using
ONLY the provided context below. Cite every factual claim with its source.
If the context doesn't contain the answer, say so explicitly — never guess.

=== Knowledge Base Context ===
{state['rag_context']}

{f"=== Additional Tool Outputs ==={chr(10)}{tool_section}" if tool_section else ""}

Question: {state['question']}

Answer:"""

    response = llm.invoke(prompt)
    return {**state, "final_answer": response.content.strip()}


# ─────────────────────────────────────────
# GRAPH
# Wire the nodes together into a state machine.
# ─────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", node_router)
    graph.add_node("tool_caller", node_tool_caller)
    graph.add_node("answer_generator", node_answer_generator)

    graph.set_entry_point("router")
    graph.add_edge("router", "tool_caller")
    graph.add_edge("tool_caller", "answer_generator")
    graph.add_edge("answer_generator", END)

    return graph.compile()


def main():
    if len(sys.argv) < 2:
        print('Usage: python orchestrator.py "your clinical question here"')
        return

    question = sys.argv[1]
    print(f"\nQuestion: {question}")

    app = build_graph()
    result = app.invoke({
        "question": question,
        "tools_needed": [],
        "rag_context": "",
        "tool_outputs": [],
        "final_answer": "",
    })

    print(f"\n{'='*50}")
    print("FINAL ANSWER")
    print('='*50)
    print(result["final_answer"])


if __name__ == "__main__":
    main()