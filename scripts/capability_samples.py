"""Seeded 300-item subset of MMLU-Pro for the capability score, as an lm-eval `--samples` file
{subtask: [doc indices]}, indices in dataset order within each subtask. Items are allocated to the
14 categories in proportion to size, then drawn without replacement with the study seed.

    python scripts/capability_samples.py            # writes data/capability/mmlu_pro_samples.json
"""
import json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402

TOTAL = 300
OUT = registry.ROOT / "data" / "capability" / "mmlu_pro_samples.json"


def main():
    from datasets import load_dataset
    seed = registry.load()["pinned"]["seed"]
    test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    cats = Counter(test["category"])
    rng = random.Random(seed)
    # proportional allocation by largest remainders
    raw = {c: TOTAL * n / len(test) for c, n in cats.items()}
    alloc = {c: int(v) for c, v in raw.items()}
    for c, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[: TOTAL - sum(alloc.values())]:
        alloc[c] += 1
    samples = {}
    for c in sorted(cats):
        idx = sorted(rng.sample(range(cats[c]), alloc[c]))
        samples[f"mmlu_pro_{c.lower().replace(' ', '_')}"] = idx
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": seed, "total": TOTAL, "samples": samples}, open(OUT, "w"), indent=1)
    json.dump(samples, open(OUT.with_name("mmlu_pro_samples_lm_eval.json"), "w"))
    print(f"{sum(alloc.values())} items over {len(alloc)} categories, seed {seed} -> {OUT}")
    for k, v in samples.items():
        print(f"  {k:28s} {len(v):3d}")


if __name__ == "__main__":
    main()
