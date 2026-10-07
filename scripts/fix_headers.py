"""Rebuild the header counts of every result file from its rows (`n_written`, `refusals`,
`parse_failures`) and `served_model` from data/results/_model_lock.json. Scores are unchanged.
Writes a <name>.pre_header.bak backup where the header changes.

    python scripts/fix_headers.py            # report
    python scripts/fix_headers.py --apply
"""
import argparse, datetime as dt, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402

RESULTS = registry.ROOT / "data" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    lock = json.load(open(RESULTS / "_model_lock.json", encoding="utf-8"))
    n_changed = 0
    for p in sorted(RESULTS.glob("*/*.jsonl")):
        lines = [l for l in open(p, encoding="utf-8") if l.strip()]
        if not lines or "__meta__" not in lines[0]:
            continue
        meta = json.loads(lines[0])["__meta__"]
        rows = []
        for l in lines[1:]:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "item_id" in r:
                rows.append(r)
        alias = meta.get("model")
        new = dict(meta)
        new.update(n_written=len(rows), refusals=sum(bool(r.get("refused")) for r in rows),
                   parse_failures=sum(r.get("parse_ok") is False for r in rows))
        served = (lock.get(alias) or {}).get("served")
        if served:
            new["served_model"] = served
        if new != meta:
            n_changed += 1
            diff = {k: (meta.get(k), new[k]) for k in new if meta.get(k) != new[k]}
            print(f"  {p.relative_to(RESULTS)}: {diff}")
            if a.apply:
                new["header_rebuilt"] = dt.date.today().isoformat()
                shutil.copy2(p, p.with_suffix(".jsonl.pre_header.bak"))
                lines[0] = json.dumps({"__meta__": new}) + "\n"
                open(p, "w", encoding="utf-8").writelines(l if l.endswith("\n") else l + "\n" for l in lines)
    print(f"\n{'APPLIED' if a.apply else 'dry run'}: {n_changed} header(s) differ from their rows.")


if __name__ == "__main__":
    main()
