"""Part 2, step 2: score the saved answers with simple rules (free, instant, no LLM).

Reads  evals/golden.jsonl            (what a good answer needs)
       evals/results/answers.jsonl   (what the agent actually said)
Checks each answer, prints PASS/FAIL per question, and a summary per check and per category.
"""
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# A citation in any of the formats the model uses: [source: x], 【x】, 【source: x】
CITATION_RE = re.compile(r"\[\s*source:|【", re.IGNORECASE)

# Fancy characters the model uses that look normal but aren't: ‑ – — − and special spaces
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2007"), " ")


def clean(s: str) -> str:
    """Lowercase and replace look-alike hyphens/spaces so '1975‑04‑30' matches '1975-04-30'."""
    return s.translate(_DASHES).translate(_SPACES).lower()


# Ways the model says "the answer isn't in my sources"
REFUSAL_RE = re.compile(
    r"not (contain|include|provide|state|mention)"
    r"|not (available|provided|included|mentioned|given)"
    r"|cannot be (derived|determined|found|answered)"
    r"|no information",
    re.IGNORECASE,
)
# ...but only when the same sentence talks about the sources, so that
# "if artesunate is not available, give quinine" is NOT counted as giving up
CONTEXT_RE = re.compile(r"context|excerpt|provided|supplied|source|document|report", re.IGNORECASE)


def is_refusal(text: str) -> bool:
    text = re.sub(r"【[^】]*】|\[\s*source:[^\]]*\]", "", text)   # citations say "source", ignore them
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if REFUSAL_RE.search(sentence) and CONTEXT_RE.search(sentence):
            return True
    return False


def load_jsonl(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score(item: dict, ans: dict) -> dict[str, bool]:
    """Return {check_name: passed} for the checks that apply to this question."""
    text = ans["answer"]
    refused = is_refusal(text)
    checks = {}

    if item.get("should_refuse"):
        checks["refused"] = refused                       # must say "not in my sources"
    else:
        checks["answered"] = not refused                  # must not give up
        checks["cited"] = bool(CITATION_RE.search(text))  # must name a source
        if item.get("must_contain"):                      # must contain the key facts
            checks["has_facts"] = all(clean(k) in clean(text) for k in item["must_contain"])

    if item.get("expected_tools"):                        # must use the right tool
        checks["right_tools"] = all(t in ans["tools"] for t in item["expected_tools"])

    return checks


def main():
    golden = {g["id"]: g for g in load_jsonl("evals/golden.jsonl")}
    answers = load_jsonl("evals/results/answers.jsonl")

    per_check = defaultdict(lambda: [0, 0])      # check -> [passed, total]
    per_category = defaultdict(lambda: [0, 0])   # category -> [passed, total]
    passed_all = 0

    for ans in answers:
        item = golden[ans["id"]]
        checks = score(item, ans)
        ok = all(checks.values())
        passed_all += ok

        for name, result in checks.items():
            per_check[name][0] += result
            per_check[name][1] += 1
        per_category[item["category"]][0] += ok
        per_category[item["category"]][1] += 1

        failed = [name for name, result in checks.items() if not result]
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {ans['id']:8} {'' if ok else 'failed: ' + ', '.join(failed)}")

    n = len(answers)
    print(f"\nOverall: {passed_all}/{n} questions pass every check ({passed_all / n:.0%})")
    print("\nBy check:")
    for name, (p, t) in per_check.items():
        print(f"  {name:12} {p}/{t}")
    print("\nBy category:")
    for cat, (p, t) in sorted(per_category.items()):
        print(f"  {cat:17} {p}/{t}")

    out = Path("evals/results/answer_scores.jsonl")
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": answers[0].get("model"),
            "commit": answers[0].get("commit"),
            "overall": round(passed_all / n, 3),
            "by_check": {k: f"{p}/{t}" for k, (p, t) in per_check.items()},
            "by_category": {k: f"{p}/{t}" for k, (p, t) in per_category.items()},
        }) + "\n")


if __name__ == "__main__":
    main()