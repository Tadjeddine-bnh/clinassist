"""Part 2, step 3: an LLM judge checks whether each answer sticks to its sources.

Version 2, after checking version 1 against hand labels:
- v1 missed answers that conclude more than a tool can measure ("X-ray: no TB")
  -> tool results now get their own section and their own yes/no question
- v1 raised a false alarm on a claim that WAS in the sources
  -> the judge must now check every claim and quote the source text that supports it
- v1 sometimes returned no JSON
  -> up to 3 attempts per answer
- the final verdict is computed by this code from the judge's checks, not taken from the judge

Version 3, after v2's run:
- v2 often wrote broken JSON (quotes copied from sources broke it)
  -> Groq strict structured outputs: the API itself guarantees valid JSON in our schema
- v2 flagged honest "not in the sources" answers as unsupported (false alarms)
  -> answers that refuse are skipped here; score_rules already scores them

Version 4, after v3 hit Groq's free daily limit (200k tokens/day for gpt-oss-20b):
- reasoning_effort="low": the judge thinks less, so it uses far fewer tokens per answer
- cache: an answer that was already judged (same question, sources, answer) is never sent again
- daily limit reached -> stop and save, instead of retrying for nothing

Different model from the writer (gpt-oss-20b vs gpt-oss-120b), temperature 0.
Groq free tier: we wait between calls. A full run takes about 7-8 minutes.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

from groq import Groq

from config import settings
from evals.score_rules import is_refusal

JUDGE_MODEL = "openai/gpt-oss-20b"
WAIT_BETWEEN_CALLS = 15   # seconds, stays under Groq's tokens-per-minute limit
MAX_ATTEMPTS = 3
JUDGE_VERSION = "v4"      # change this when the instructions change, so old cached verdicts aren't reused
CACHE_FILE = Path("evals/results/judge_cache.json")


class DailyLimitReached(Exception):
    pass

INSTRUCTIONS = """You are checking a clinical assistant's answer against its sources.
The assistant was only allowed to use the DOCUMENTS and TOOL RESULTS below.

Step 1. Split the ANSWER into its individual factual claims.
For each claim, search the DOCUMENTS and TOOL RESULTS for text that states it.
- If you find it, copy a short quote (max 15 words) as evidence and mark supported = true.
- If you cannot find it, set evidence to "none" and supported = false.
Rewording, summarising and translating count as supported.
Saying that information is missing from the sources counts as supported.
Changing what a source says (for example correcting a misspelled drug name, or stating an unclear
dose or label as if the source said it clearly) counts as NOT supported.
Invented details such as page numbers count as NOT supported.

Step 2. Tool overreach. TOOL RESULTS (for example an image classifier) only support conclusions about
what the tool actually outputs. If a classifier only has the labels NORMAL and PNEUMONIA, it says nothing
about tuberculosis or any other disease. Guideline text in DOCUMENTS describes diseases in general;
it is not a finding about this patient's image.
Set tool_overreach = true if the answer concludes anything about the patient that the tool's labels
cannot show. Otherwise false. If there are no tool results, false.

Fill in the claims list, tool_overreach, and tool_overreach_reason (one sentence, or empty)."""

# Groq strict mode makes the model's reply always match this shape
SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "evidence", "supported"],
                "additionalProperties": False,
            },
        },
        "tool_overreach": {"type": "boolean"},
        "tool_overreach_reason": {"type": "string"},
    },
    "required": ["claims", "tool_overreach", "tool_overreach_reason"],
    "additionalProperties": False,
}


def load_jsonl(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_message(ans: dict) -> str:
    documents = ans.get("context", "").strip() or "(none)"
    tools = "\n\n".join(dict.fromkeys(ans.get("tool_outputs", []))).strip() or "(none)"
    return (f"{INSTRUCTIONS}\n\nQUESTION:\n{ans['question']}\n\n"
            f"DOCUMENTS:\n{documents}\n\nTOOL RESULTS:\n{tools}\n\n"
            f"ANSWER:\n{ans['answer']}")


def judge(client: Groq, ans: dict) -> dict:
    message = build_message(ans)
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                reasoning_effort="low",
                messages=[{"role": "user", "content": message}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "judgement", "strict": True, "schema": SCHEMA},
                },
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last_error = str(e)
            if "tokens per day" in last_error or "(TPD)" in last_error:
                raise DailyLimitReached(last_error)
            wait = 60 if ("rate" in last_error.lower() or "429" in last_error) else 5
            if attempt < MAX_ATTEMPTS:
                print(f"   attempt {attempt} failed ({last_error[:60]}), retrying in {wait} s...")
                time.sleep(wait)
    return {"error": last_error}


def cache_key(ans: dict) -> str:
    raw = JUDGE_VERSION + ans["question"] + build_message(ans)[len(INSTRUCTIONS):]
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def verdict_from(result: dict) -> tuple[str, list[str]]:
    """The code decides the verdict from the judge's individual checks."""
    if "error" in result:
        return "ERROR", [f"JUDGE ERROR: {result['error']}"]
    problems = [f"unsupported: {c.get('claim', '')}"
                for c in result.get("claims", []) if not c.get("supported", False)]
    if result.get("tool_overreach"):
        problems.append(f"tool overreach: {result.get('tool_overreach_reason', '')}")
    return ("NOT_SUPPORTED" if problems else "SUPPORTED"), problems


def main():
    golden = {g["id"]: g for g in load_jsonl("evals/golden.jsonl")}
    answers_path = sys.argv[1] if len(sys.argv) > 1 else "evals/results/answers.jsonl"
    answers = [a for a in load_jsonl(answers_path)
               if not golden[a["id"]].get("should_refuse")]
    client = Groq(api_key=settings.groq_api_key)
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}

    rows, counts = [], {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "ERROR": 0, "SKIPPED": 0}
    for n, ans in enumerate(answers, 1):
        if is_refusal(ans["answer"]):
            # "not in the sources" makes no claims to check; score_rules handles it
            counts["SKIPPED"] += 1
            rows.append({"id": ans["id"], "verdict": "SKIPPED", "problems": ["answer says info is missing"]})
            print(f"[{n}/{len(answers)}] -   {ans['id']} (skipped: answer says info is missing)")
            continue
        key = cache_key(ans)
        from_cache = key in cache
        if from_cache:
            result = cache[key]
        else:
            try:
                result = judge(client, ans)
            except DailyLimitReached:
                print("\nGroq's free daily token limit for the judge model is used up.")
                print("Results so far are saved. Run again after the limit resets:")
                print("answers already judged come from the cache and cost nothing.")
                break
            if "error" not in result:
                cache[key] = result
                CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        verdict, problems = verdict_from(result)
        counts[verdict] += 1
        rows.append({"id": ans["id"], "verdict": verdict, "problems": problems, "raw": result})

        label = {"SUPPORTED": "OK ", "NOT_SUPPORTED": "BAD", "ERROR": "ERR"}[verdict]
        print(f"[{n}/{len(answers)}] {label} {ans['id']}{'  (cached)' if from_cache else ''}")
        for p in problems:
            print(f"        - {p}")
        if n < len(answers) and not from_cache:
            time.sleep(WAIT_BETWEEN_CALLS)

    out = Path(f"evals/results/judge_{Path(answers_path).stem}.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    total = len(rows)
    print(f"\nSupported: {counts['SUPPORTED']}/{total}   "
          f"Not supported: {counts['NOT_SUPPORTED']}/{total}   "
          f"Skipped (refusals): {counts['SKIPPED']}   Errors: {counts['ERROR']}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()