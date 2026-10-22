import json
import sys

sys.stdout.reconfigure(encoding="utf-8")   # so Arabic prints correctly in PowerShell

prefixes = tuple(sys.argv[1:]) or ("",)    # e.g. "oos" "cv"; nothing = show all
for line in open("evals/results/answers.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].startswith(prefixes):
        print(f"\n=== {r['id']}  ({r['seconds']} s, tools={r['tools']})")
        print("Q:", r["question"])
        print("A:", r["answer"])