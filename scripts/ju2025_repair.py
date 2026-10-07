"""ju2025 repair: rewrite the fifteen main ju2025 result files, dropping rows that repeat a
(strategy, item) pair and rows flagged `refused` by the "I can't help but ..." idiom. The runner
then collects the missing pairs. Each file is backed up first as <name>.pre_repair.bak.

    python scripts/ju2025_repair.py            # report only
    python scripts/ju2025_repair.py --apply    # back up and rewrite
"""
import argparse, json, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402

IDIOM = re.compile(r"help but", re.I)
D = registry.ROOT / "data" / "results" / "ju2025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cfg = registry.load()
    total_dup = total_idiom = 0
    for model in registry.all_models(cfg):
        p = D / f"{model}.jsonl"
        if not p.exists():
            continue
        lines = [l for l in open(p, encoding="utf-8") if l.strip()]
        keep, seen, dup, idiom = [], set(), 0, 0
        for l in lines:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "item_id" not in r:
                keep.append(l); continue
            k = (r.get("strategy"), r.get("item"))
            if k in seen:
                dup += 1; continue
            if r.get("refused") and IDIOM.search(r.get("response") or ""):
                idiom += 1; continue
            seen.add(k); keep.append(l)
        total_dup += dup; total_idiom += idiom
        print(f"  {model:20s} rows {len(lines) - 1:5d}  distinct pairs {len(seen):5d}  dropped: duplicates {dup:4d}  idiom refusals {idiom:3d}")
        if a.apply and (dup or idiom):
            shutil.copy2(p, p.with_suffix(".jsonl.pre_repair.bak"))
            with open(p, "w", encoding="utf-8") as fh:
                fh.writelines(l if l.endswith("\n") else l + "\n" for l in keep)
    print(f"\n{'APPLIED' if a.apply else 'dry run'}: duplicates {total_dup}, idiom refusals {total_idiom}."
          + ("" if a.apply else " Re-run with --apply to rewrite (backups kept)."))
    if a.apply:
        print("Now re-collect: for each model, python -m core.runner --method ju2025 --model <alias>")


if __name__ == "__main__":
    main()
