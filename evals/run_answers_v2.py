"""Collect answers from the NEW tool-calling agent (agent/graph.py).

Writes evals/results/answers_v2.jsonl in the SAME row format as run_answers.py,
so score_rules.py and judge.py work on it unchanged.

Usage:
    uv run python -m evals.run_answers_v2                  # every golden question
    uv run python -m evals.run_answers_v2 doc cv who-02    # only ids that start with these

Resumable: ids already in answers_v2.jsonl are skipped, so if Groq's daily limit stops you,
run the same command again later. Delete the file to start over.

Groq free tier is ~8,000 tokens/minute and one question now uses 3,000-6,000 tokens
(each LLM call re-reads the whole conversation), so we wait between questions.
"""
import json
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from config import GIT_COMMIT, settings

OUT = Path("evals/results/answers_v2.jsonl")
WAIT_BETWEEN_QUESTIONS = 45            # seconds
CONTEXT_TOOLS = {"search_guidelines", "read_file"}   # what the writer read: the judge's "documents"
RESULT_TOOLS = {"classify_xray"}                      # tool results: the judge's "tool results"
HELPER_TOOLS = {"list_files", "list_images"}          # menus: not worth listing as "tools used"


class DailyLimit(Exception):
    pass


def ask(agent, question: str) -> dict:
    for attempt in (1, 2):
        try:
            return agent.invoke({"messages": [HumanMessage(question)]}, config={"recursion_limit": 14})
        except Exception as e:
            msg = str(e)
            if "tokens per day" in msg or "(TPD)" in msg:
                raise DailyLimit(msg)
            if attempt == 1 and ("rate" in msg.lower() or "429" in msg):
                print("   rate limit hit, waiting 60 s and retrying once...")
                time.sleep(60)
                continue
            raise


def to_row(item: dict, result: dict, seconds: float) -> dict:
    messages = result["messages"]
    tool_msgs = [m for m in messages if m.type == "tool"]
    return {
        "id": item["id"],
        "question": item["question"],
        "answer": messages[-1].content,                       # the last message is the final answer
        "tools": list(dict.fromkeys(m.name for m in tool_msgs if m.name not in HELPER_TOOLS)),
        "context": "\n\n".join(m.content for m in tool_msgs if m.name in CONTEXT_TOOLS),
        "tool_outputs": [m.content for m in tool_msgs if m.name in RESULT_TOOLS],
        "seconds": seconds,
        "model": settings.llm_model,
        "commit": GIT_COMMIT,
        "agent": "tool-calling",
    }


def main():
    prefixes = tuple(sys.argv[1:])
    lines = Path("evals/golden.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if prefixes:
        items = [i for i in items if i["id"].startswith(prefixes)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()}

    agent = build_graph()
    with OUT.open("a", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                print(f"[{n}/{len(items)}] {item['id']:8} already done, skipping")
                continue
            t0 = time.time()
            try:
                result = ask(agent, item["question"])
            except DailyLimit:
                print("\nGroq's free daily token limit is used up. Answers so far are saved.")
                print("Run the same command again later: finished questions are skipped.")
                return
            except Exception as e:
                print(f"[{n}/{len(items)}] {item['id']:8} FAILED ({str(e)[:80]}), will retry next run")
                continue

            row = to_row(item, result, round(time.time() - t0, 2))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(items)}] {item['id']:8} {row['seconds']:5.1f}s  tools={row['tools']}")
            if n < len(items):
                time.sleep(WAIT_BETWEEN_QUESTIONS)

    print(f"\nDone. Answers are in {OUT}")


if __name__ == "__main__":
    main()