"""List the cells that still need collecting by comparing every results file with the sizes in
config. Expected: main file at n (x11 for ju2025), `rep2` at ceil(0.2 n), and a `temp0` file of 20
for bozdag2025 on local models. Exits 1 if anything is missing or short. Read-only.

    python scripts/gaps.py            # the gap list, one line per problem, then a summary
    python scripts/gaps.py --all      # every cell, including the complete ones
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402

RESULTS = registry.ROOT / "data" / "results"
ROWS_PER_ITEM = {"ju2025": 11}


def read(path):
    ids, dup, refused, bad = set(), 0, 0, 0
    if not path.exists():
        return None
    for i, line in enumerate(open(path, encoding="utf-8")):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "item_id" not in r:
            continue
        # unit key: (strategy, item) for ju2025, else the item id
        key = (r["strategy"], r["item"]) if "strategy" in r and "item" in r else r["item_id"]
        if key in ids:
            dup += 1
            continue
        ids.add(key)
        refused += bool(r.get("refused"))
        bad += not r.get("parse_ok", True)
    return {"n": len(ids), "dup": dup, "refused": refused, "bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cfg = registry.load()
    local = list(cfg["models"].get("local", {}))
    models = list(registry.all_models(cfg))
    problems, notes, complete = [], [], 0
    for m, mc in cfg["methods"].items():
        n = mc["n"]
        if not isinstance(n, int):
            continue
        k = ROWS_PER_ITEM.get(m, 1)
        want = {"": n * k, "rep2": max(4, math.ceil(0.2 * n)) * k}
        for model in models:
            checks = dict(want)
            if m == "bozdag2025" and model in local:
                checks["temp0"] = 20
            for tag, exp in checks.items():
                f = RESULTS / m / (f"{model}__{tag}.jsonl" if tag else f"{model}.jsonl")
                r = read(f)
                cell = f"{m:22s} {model:18s} {tag or 'main':5s}"
                if r is None:
                    problems.append(f"MISSING  {cell} 0/{exp}")
                    continue
                if r["n"] < exp:
                    problems.append(f"SHORT    {cell} {r['n']}/{exp}")
                else:
                    complete += 1
                    if a.all:
                        print(f"ok       {cell} {r['n']}/{exp}")
                if r["dup"]:
                    notes.append(f"DUP      {cell} {r['dup']} duplicate row(s), ignored")
                if r["n"] and r["refused"] == r["n"]:
                    notes.append(f"ALLREF   {cell} every item refused")
                elif r["n"] and (r["refused"] + r["bad"]) > 0.25 * r["n"]:
                    notes.append(f"LOSSY    {cell} refused {r['refused']} unparsed {r['bad']} of {r['n']}")
    for p in problems:
        print(p)
    for p in notes:
        print(p)
    print(f"\n{complete} complete, {len(problems)} missing/short, {len(notes)} notes")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
