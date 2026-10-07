"""
Check that the pinned persuadee separates persuaders of different strength: run the method's
adapter with three probe models as the persuader and report the spread of their means against the
within-probe noise, with floor and ceiling warnings.

    python scripts/persuadee_check.py --method bozdag2025 --n 12
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import registry                      # noqa: E402
from core.clients import get_client            # noqa: E402

# probe persuaders of different strength, all on the OpenAI key
PROBES = [
    ("weak", "gpt-4.1-nano-2025-04-14"),
    ("mid", "gpt-4o-mini-2024-07-18"),
    ("strong", "gpt-5.6-sol"),
]


def run_probe(cfg, method: str, persuader_id: str, n: int) -> list[float]:
    """Run the method's own adapter, with the tested model swapped for a probe."""
    import importlib

    from core import clients as C
    from core.registry import all_models

    anchor = next(iter(all_models(cfg)))          # any tested model; discarded immediately
    spec = registry.resolve(cfg, method, anchor)

    agents = {role: C.get_client(a, cfg, is_pinned=True) if isinstance(a, str)
              else [C.get_client(x, cfg, is_pinned=True) for x in a]
              for role, a in spec.agents.items()}
    spec.extra["agents_resolved"] = agents
    spec.extra["test_mode"] = False
    spec.extra["cfg"] = cfg
    spec.n = n

    client = C._OpenAICompatible(f"probe:{persuader_id}", persuader_id,
                                 api_key_env="OPENAI_API_KEY", reasoning=None)

    mod = importlib.import_module(f"methods.{method}.adapter")
    scores = []
    for row in mod.run(client, spec):
        if row.get("score") is not None:
            scores.append(float(row["score"]))
        if len(scores) >= n:
            break
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="bozdag2025",
                    help="a dialogue method with a pinned persuadee")
    ap.add_argument("--n", type=int, default=12, help="items per probe")
    args = ap.parse_args()

    cfg = registry.load()
    pinned = cfg["pinned"]
    print(f"\nPERSUADEE CHECK  method={args.method}  n={args.n} per probe")
    print(f"  pinned persuadee: {pinned.get('persuadee')} "
          f"-> {cfg['models']['pinned_agents'][pinned['persuadee']]['id']}")
    print(f"  probes: {', '.join(f'{k}={v}' for k, v in PROBES)}\n")

    results = {}
    for label, mid in PROBES:
        try:
            s = run_probe(cfg, args.method, mid, args.n)
        except Exception as e:
            print(f"  {label:7} {mid:26} FAILED {type(e).__name__}: {str(e)[:90]}")
            continue
        if not s:
            print(f"  {label:7} {mid:26} no scored items")
            continue
        results[label] = s
        sd = st.stdev(s) if len(s) > 1 else 0.0
        print(f"  {label:7} {mid:26} mean {st.mean(s):.3f}  sd {sd:.3f}  n {len(s)}")

    if len(results) < 2:
        sys.exit("\n  Not enough probes returned scores - cannot judge separability.")

    means = {k: st.mean(v) for k, v in results.items()}
    spread = max(means.values()) - min(means.values())
    # pooled within-probe SD, then the SE of a difference of means
    pooled = st.mean([st.stdev(v) for v in results.values() if len(v) > 1] or [0.0])
    se_diff = (2 * pooled ** 2 / args.n) ** 0.5 if pooled else 0.0
    sep = spread / se_diff if se_diff else float("inf")

    print("\n" + "=" * 66)
    print(f"  observed spread (max-min of means) : {spread:.3f}")
    print(f"  pooled within-probe SD             : {pooled:.3f}")
    print(f"  SE of a difference of means        : {se_diff:.3f}")
    print(f"  separability (spread / SE)         : {sep:.2f}")

    lo, hi = min(means.values()), max(means.values())
    scale = cfg["methods"][args.method].get("scale") or {}
    floor, ceil = tuple(scale.get("metric_range") or (None, None))

    print()
    if sep >= 2:
        print("  PASS - the persuadee separates persuaders of different strength.")
    else:
        print("  FAIL - the scores do not separate. This persuadee cannot rank models on this")
        print("         method, and eight columns depend on it. Do not start collection.")
    if isinstance(ceil, (int, float)) and hi >= ceil - 0.05 * (ceil - (floor or 0)):
        print(f"  WARNING ceiling effect: highest mean {hi:.3f} is at the scale ceiling {ceil}.")
    if isinstance(floor, (int, float)) and lo <= floor + 0.05 * ((ceil or 1) - floor):
        print(f"  WARNING floor effect: lowest mean {lo:.3f} is at the scale floor {floor}.")
    print("  Ordering:", " < ".join(k for k, _ in sorted(means.items(), key=lambda x: x[1])),
          "(expected: weak < mid < strong)")


if __name__ == "__main__":
    main()
