"""Re-extract MMLU-Pro answers from the stored lm-eval outputs with a format-aware parser. Keeps
lm-eval's letter where it found one, otherwise takes the last letter matched by PATTERNS. With
--write, rewrites the mmlu_pro_300 column of data/capability/summary.csv and keeps the original
as summary_lmeval.csv.

    python scripts/capability_rescore.py            # prints per-model recovery; --write updates summary.csv
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "data" / "capability"

PATTERNS = [                                                  # tried in order, last match wins within each
    r"answer is:?\s*\**\(?([A-J])\)?",
    r"[Aa]nswer\**:?\**\s*\(?\**([A-J])\**\)?",
    r"(?:choice|option|answer) is\s*\**\(?([A-J])\)?\**",
    r"\*\*\(?([A-J])\)?\*\*",
    r"[Oo]ption\s+\**([A-J])\**\b",
]


def extract(text: str):
    for pat in PATTERNS:
        hits = re.findall(pat, text)
        if hits:
            return hits[-1]
    return None


def rescore(model_dir: Path):
    n = recovered = correct_lmeval = correct_new = 0
    for f in sorted(model_dir.glob("*/samples_mmlu_pro_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            gold = r["doc"]["answer"]
            filt = str(r.get("filtered_resps", [""])[0] or "").strip()
            parsed = filt if re.fullmatch(r"[A-J]", filt) else None
            if parsed is None:
                parsed_new = extract(r["resps"][0][0] or "")
                recovered += parsed_new is not None
            else:
                parsed_new = parsed
            correct_lmeval += parsed == gold
            correct_new += parsed_new == gold
    return n, recovered, correct_lmeval / n if n else None, correct_new / n if n else None


def main():
    # compare against the lm-eval originals once --write has kept them
    src = CAP / "summary_lmeval.csv" if (CAP / "summary_lmeval.csv").exists() else CAP / "summary.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    out = []
    print(f"{'model':20s} {'n':>4s} {'recovered':>9s} {'lm-eval':>8s} {'rescored':>9s} {'summary.csv':>12s}")
    for row in rows:
        n, rec, acc0, acc1 = rescore(CAP / row["model"])
        print(f"{row['model']:20s} {n:4d} {rec:9d} {acc0:8.3f} {acc1:9.3f} {float(row['mmlu_pro_300']):12.3f}")
        assert abs(acc0 - float(row["mmlu_pro_300"])) < 1e-6, f"{row['model']}: lm-eval accuracy does not match summary.csv"
        out.append({**row, "mmlu_pro_300": acc1})
    if "--write" in sys.argv:
        orig = CAP / "summary_lmeval.csv"
        if not orig.exists():
            orig.write_text((CAP / "summary.csv").read_text(encoding="utf-8"), encoding="utf-8")
        with open(CAP / "summary.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {CAP / 'summary.csv'} (lm-eval original kept as summary_lmeval.csv)")


if __name__ == "__main__":
    main()
