"""Part 2, step 1: ask the real agent every golden question and save the answers.

No scoring here. Running the agent is the slow part (LLM calls), so we do it once,
save the answers, and score them separately as many times as we like.

Groq free tier: ~8,000 tokens/minute and one question uses ~2,000 tokens,
so we wait between questions. A full run takes about 10 minutes.
"""
import json
import time
from pathlib import Path

from agent.orchestrator import build_graph
from config import settings, GIT_COMMIT

WAIT_BETWEEN_QUESTIONS = 15   # seconds; keeps us under Groq's tokens-per-minute limit


def ask(agent, question: str) -> dict:
    state = {"question": question, "tools_needed": [], "rag_context": "",
             "tool_outputs": [], "final_answer": ""}
    try:
        return agent.invoke(state)
    except Exception as e:
        if "rate" in str(e).lower() or "429" in str(e):
            print("   rate limit hit, waiting 60 s and retrying once...")
            time.sleep(60)
            return agent.invoke(state)
        raise


def main():
    lines = Path("evals/golden.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    agent = build_graph()

    out = Path("evals/results/answers.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            t0 = time.time()
            result = ask(agent, it["question"])
            secs = round(time.time() - t0, 2)

            row = {
                "id": it["id"],
                "question": it["question"],
                "answer": result["final_answer"],
                "tools": result["tools_needed"],
                "seconds": secs,
                "model": settings.llm_model,
                "commit": GIT_COMMIT,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()   # saved line by line, so a crash halfway doesn't lose everything

            print(f"[{n}/{len(items)}] {it['id']:8} {secs:5.1f}s  tools={result['tools_needed']}")
            if n < len(items):
                time.sleep(WAIT_BETWEEN_QUESTIONS)

    print(f"\nSaved {len(items)} answers to {out}")


if __name__ == "__main__":
    main()