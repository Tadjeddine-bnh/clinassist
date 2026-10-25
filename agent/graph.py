"""
Phase 5, version 2: an agent with real tool calling.

Instead of a separate "router" call that writes "rag,cv" as text, the LLM is given a list of tools.
It decides itself which tool to call and with what input, reads the result, and either calls another
tool or writes the answer. LangGraph runs that loop.

    question -> agent (LLM) --wants a tool?--> tools -> back to agent -> ... -> answer

Usage:
    uv run python -m agent.graph "How much IV fluid do I give a child with severe dehydration?"
"""
import sys
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import settings
from knowledge_base.resources import search


# ─────────────────────────────────────────
# TOOLS
# The docstring is what the LLM reads to decide when to use a tool, so it matters.
# ─────────────────────────────────────────
@tool
def search_guidelines(query: str) -> str:
    """Search the clinical knowledge base: the WHO Pocket Book of Hospital Care for Children
    and any indexed patient documents. Use it for every clinical question.
    Returns passages, each starting with a label like [source: file.pdf, p. 159]."""
    results = search(query, k=settings.top_k)
    parts = []
    for doc, meta in results:
        label = f"{meta['source']}, p. {meta['page']}" if "page" in meta else meta["source"]
        parts.append(f"[source: {label}]\n{doc}")
    return "\n\n".join(parts) or "No passages found."


TOOLS = [search_guidelines]


# ─────────────────────────────────────────
# THE AGENT
# ─────────────────────────────────────────
SYSTEM = """You are a clinical AI assistant for clinicians.
Always use your tools to find information before answering. Answer ONLY from what the tools return.
If the tools don't give you the answer, say so explicitly. Never guess.
If a passage mentions a table, chart or treatment plan you need but whose content you don't see,
search again specifically for it. You can search up to 3 times.

Citations:
- Every passage from a tool starts with a label like [source: file.pdf, p. 159].
- After every factual claim, copy the label of the passage it came from, exactly as written.
- Never write a page number that is not in a label.

Tables:
- Some passages contain tables written as |cell|cell|. Read which row and column each value is in,
  and copy numbers exactly as they appear. Never estimate or invent a value the table doesn't show."""


class State(TypedDict):
    # The whole conversation: question, tool calls, tool results, answer.
    # add_messages means each node's new messages are APPENDED, not replacing the list.
    messages: Annotated[list[AnyMessage], add_messages]

MAX_SEARCHES = 3

llm = ChatGroq(model=settings.llm_model, temperature=0, api_key=settings.groq_api_key)
llm_with_tools = llm.bind_tools(TOOLS)


def force_answer(state: State):
    """Out of searches: give the LLM the question and all passages, with no tools to call."""
    question = state["messages"][0].content
    passages = "\n\n".join(m.content for m in state["messages"] if m.type == "tool")
    prompt = f"""{SYSTEM}

You have used all your searches. Do not search again. Answer now using only the passages below.
If they don't fully answer the question, say clearly what is missing.

=== Passages ===
{passages}

Question: {question}

Answer:"""
    return llm.invoke(prompt)


def agent(state: State) -> dict:
    """Ask the LLM what to do next: call a tool, or answer."""
    searches_done = sum(1 for m in state["messages"] if m.type == "tool")
    if searches_done >= MAX_SEARCHES:
        return {"messages": [force_answer(state)]}
    reply = llm_with_tools.invoke([SystemMessage(SYSTEM)] + state["messages"])
    return {"messages": [reply]}
def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))           # runs whatever tools the LLM asked for
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)   # tool calls? -> "tools", otherwise -> end
    graph.add_edge("tools", "agent")                   # after a tool, back to the LLM
    return graph.compile()


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python -m agent.graph "your clinical question"')
        return

    app = build_graph()
    result = app.invoke(
        {"messages": [HumanMessage(sys.argv[1])]},
        config={"recursion_limit": 8, "callbacks": [CallbackHandler()]},   # max 8 steps: no endless loops
    )
    for message in result["messages"]:
        message.pretty_print()        # shows every step: question, tool call, tool result, answer
    get_client().flush()              # send the Langfuse trace before the script exits


if __name__ == "__main__":
    main()