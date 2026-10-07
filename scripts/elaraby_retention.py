"""
elaraby2024 stage 1 only (no judge calls): each model's agreement with the frozen GPT-4 baseline's
decision, the observed intersection of agreed items, and its projected size at 15 models.
Writes data/derived/elaraby_retention.json.

    python scripts/elaraby_retention.py --n 40 --models gpt-5.6-sol,deepseek-v4-flash,kimi-k2.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import registry                                   # noqa: E402
from core.clients import get_client, is_refusal             # noqa: E402


def stage1(cfg, alias, n, seed):
    """The tested model's own decision per item."""
    from methods.elaraby2024 import adapter as A

    client = get_client(alias, cfg)
    out = []
    for it in A._load_items(cfg, n, seed):
        # the row's own prompt, as the adapter sends it
        r = client.chat([{"role": "user", "content": it["prompt"]}], max_tokens=384,
                        temperature=0.0)
        obj = A._parse_json(r.text)
        dec = None
        if obj and "decision" in obj:
            try:
                dec = int(obj["decision"])
            except (TypeError, ValueError):
                dec = None
        out.append(dict(topic=it["topic"], baseline=it["baseline_decision"], tested=dec,
                        refused=is_refusal(r.text), parse_ok=dec is not None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--models", default="gpt-5.6-sol,deepseek-v4-flash,kimi-k2.6")
    ap.add_argument("--out", default="data/derived/elaraby_retention.json")
    args = ap.parse_args()

    cfg = registry.load()
    aliases = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"\nELARABY2024 RETENTION  n={args.n}  stage 1 only, no judge calls")
    print(f"  probing: {', '.join(aliases)}\n")

    per_model, agree_sets = {}, {}
    for alias in aliases:
        try:
            rows = stage1(cfg, alias, args.n, cfg["pinned"].get("seed") or 0)
        except Exception as e:
            print(f"  {alias:22} FAILED {type(e).__name__}: {str(e)[:80]}")
            continue
        usable = [r for r in rows if r["parse_ok"]]
        agrees = [r for r in usable if r["tested"] == r["baseline"]]
        per_model[alias] = dict(n=len(rows), usable=len(usable), agree=len(agrees),
                                p=(len(agrees) / len(usable)) if usable else None,
                                refused=sum(r["refused"] for r in rows))
        agree_sets[alias] = {r["topic"] + "|" + str(i)
                             for i, r in enumerate(rows) if r["parse_ok"]
                             and r["tested"] == r["baseline"]}
        m = per_model[alias]
        print(f"  {alias:22} agreed {m['agree']:3}/{m['usable']:3}  p={m['p']:.3f}"
              f"  (refusals {m['refused']}, parse failures {m['n'] - m['usable']})")

    ps = [m["p"] for m in per_model.values() if m["p"] is not None]
    if not ps:
        sys.exit("\n  no model produced usable decisions - cannot judge retention")

    observed = set.intersection(*agree_sets.values()) if len(agree_sets) > 1 else \
        next(iter(agree_sets.values()))
    print("\n" + "=" * 70)
    print(f"  observed intersection across {len(agree_sets)} model(s): "
          f"{len(observed)}/{args.n} items ({len(observed)/args.n:.1%})")
    print(f"  agreement rates: min {min(ps):.3f}  mean {sum(ps)/len(ps):.3f}  max {max(ps):.3f}")

    # projection to 15 models from the mean and the worst rate
    prod = 1.0
    for p in ps:
        prod *= p
    mean_p = sum(ps) / len(ps)
    print(f"\n  projected retention at 15 models, of {args.n} items:")
    print(f"    using the MEAN rate  ({mean_p:.3f}^15) : {mean_p ** 15 * args.n:6.1f} items")
    print(f"    using the WORST rate ({min(ps):.3f}^15) : {min(ps) ** 15 * args.n:6.1f} items")
    print("    (the product is driven by the worst model, so the second line is the one to plan on)")

    print("\n  READ AS:")
    if min(ps) ** 15 * args.n >= 20:
        print("    Intersection scoring is VIABLE - enough items survive. It still makes the item")
        print("    set pool-dependent, the property that excluded a Bradley-Terry rating, so that")
        print("    remains a declared cost rather than a solved problem.")
    else:
        print("    Intersection scoring is NOT viable at this n - too few items survive, and the")
        print("    retention is set by the weakest model in the set. The fallback is (a) leaving")
        print("    items unscored, with the missing-not-at-random caveat stated, or (b) forcing the")
        print("    baseline's winner and declaring the task change.")
    print("\n  CAVEAT: only API models were reachable (no GPU window), and they are all frontier.")
    print("  The 3B and 7B local models will agree LESS, so the true retention is below this.")

    dest = ROOT / args.out
    dest.write_text(json.dumps(dict(n=args.n, per_model=per_model,
                                    observed_intersection=len(observed)), indent=1),
                    encoding="utf-8")
    print(f"\n  -> {dest}")


if __name__ == "__main__":
    main()
