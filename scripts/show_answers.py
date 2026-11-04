import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

args = sys.argv[1:]
path = "evals/results/answers.jsonl"
if args and args[0].endswith(".jsonl"):
    path, args = args[0], args[1:]
prefixes = tuple(args) or ("",)

for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r["id"].startswith(prefixes):
        print(f"\n=== {r['id']}  ({r['seconds']} s, tools={r['tools']})")
        print("Q:", r["question"])
        print("A:", r["answer"])