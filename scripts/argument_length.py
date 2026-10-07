"""Length of each model's visible adversary arguments in amayuelas2024: adversary turns, mean and
median words per turn, mean words of each item's first turn, and the share of empty turns. Read-only.

    python scripts/argument_length.py
"""
import json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402


def main():
    cfg = registry.load()
    d = registry.ROOT / "data" / "results" / "amayuelas2024"
    print(f"{'model':20s} {'items':>5s} {'turns':>5s} {'mean_w':>7s} {'median':>7s} {'first_w':>8s} {'empty':>6s}")
    for model in registry.all_models(cfg):
        f = d / f"{model}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        rows = [r for r in rows if "item_id" in r]
        words, first, empty, turns = [], [], 0, 0
        for r in rows:
            adv = [m for m in (r.get("transcript") or {}).get("adversary", []) if m.get("role") == "assistant"]
            for k, m in enumerate(adv):
                turns += 1
                w = len((m.get("content") or "").split())
                if w == 0:
                    empty += 1
                else:
                    words.append(w)
                    if k == 0:
                        first.append(w)
        if not turns:
            continue
        print(f"{model:20s} {len(rows):5d} {turns:5d} {statistics.mean(words) if words else 0:7.0f} "
              f"{statistics.median(words) if words else 0:7.0f} {statistics.mean(first) if first else 0:8.0f} "
              f"{empty / turns:6.1%}")


if __name__ == "__main__":
    main()
