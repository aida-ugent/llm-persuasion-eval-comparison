r"""
Pauli anchor: score the dataset's own annotated pairs through the adapter's `_score_pair` and
compare with the published human correlation (0.845) and length baseline (0.388). In-sample, so it
checks the wiring, not generalisation. Also reports the annotators' attainable ceiling and rho by
source and generator. Caches predictions to data/results/pauli2025/__pauli_anchor.jsonl and writes
data/derived/pauli_anchor.json.

    python scripts/pauli_anchor.py --limit 100      # ~3 min on CPU, a smoke test
    python scripts/pauli_anchor.py                  # all 2,697 pairs, ~66 min on CPU, resumable
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from methods.pauli2025.adapter import DATASET, SCORER, _load_scorer, _score_pair  # noqa: E402

OUT = ROOT / "data" / "results" / "pauli2025" / "__pauli_anchor.jsonl"
DEST = ROOT / "data" / "derived" / "pauli_anchor.json"
PAPER_RHO, PAPER_LENGTH_BASELINE = 0.845, 0.388
ANNOTATORS = ("score_1", "score_2", "score_3")


def spearman(a, b):
    """Spearman rho with average ranks for ties."""
    if len(a) < 3:
        return None

    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    x, y = rank(a), rank(b)
    mx, my = st.mean(x), st.mean(y)
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    return num / den if den else None


def boot_ci(pairs, seed, n_boot=1000):
    """95% row-bootstrap interval of rho."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        r = spearman([p for p, _ in draw], [h for _, h in draw])
        if r is not None:
            vals.append(r)
    vals.sort()
    q = lambda f: round(vals[int(f * (len(vals) - 1))], 4)   # noqa: E731
    return [q(0.025), q(0.975)] if vals else None


def load_rows(limit=None):
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train")
    rows = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        scores = [r[a] for a in ANNOTATORS]
        if any(s is None for s in scores):
            continue
        rows.append(dict(hf_row=i, text1=str(r["text1"]), text2=str(r["text2"]),
                         human=sum(scores) / len(scores), scores=scores,
                         source=r["source"], generator=r["model"],
                         gen_place=r["gen_place"], flip=r["flip"]))
    return rows


def predict(rows, args):
    """Cache-backed predictions from the adapter's `_score_pair`."""
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip() and '"__meta__"' not in line[:14]:
                d = json.loads(line)
                done[d["hf_row"]] = d["pred"]

    todo = [r for r in rows if r["hf_row"] not in done]
    if todo:
        tok, model, torch = _load_scorer()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if not OUT.exists():
            OUT.write_text(json.dumps(dict(__meta__=dict(
                script="scripts/pauli_anchor.py", scorer=SCORER, dataset=DATASET,
                scored_by="methods.pauli2025.adapter._score_pair, unchanged",
                note="IN-SAMPLE: the checkpoint was fitted on this dataset; the paper's 0.845 "
                     "is 10-fold cross-validated. This is a wiring check, not a replication.",
                written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))) + "\n",
                encoding="utf-8")
        t0 = time.time()
        with OUT.open("a", encoding="utf-8") as fh:
            for k, r in enumerate(todo, 1):
                p = _score_pair(tok, model, torch, r["text1"], r["text2"])
                done[r["hf_row"]] = p
                fh.write(json.dumps(dict(hf_row=r["hf_row"], pred=round(p, 6))) + "\n")
                fh.flush()
                if k % 50 == 0 or k == len(todo):
                    rate = (time.time() - t0) / k
                    print(f"  {k}/{len(todo)}  {rate:.2f}s/pair  "
                          f"~{rate * (len(todo) - k) / 60:.0f} min left", flush=True)
    for r in rows:
        r["pred"] = done.get(r["hf_row"])
    return [r for r in rows if r["pred"] is not None]


def subgroup(rows, key):
    out = {}
    for lv in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == lv]
        out[str(lv)] = dict(n=len(sub),
                            rho=round(spearman([r["pred"] for r in sub],
                                               [r["human"] for r in sub]) or 0, 4))
    return out


def analyse(rows, seed):
    pred = [r["pred"] for r in rows]
    human = [r["human"] for r in rows]
    length = [len(r["text2"].split()) - len(r["text1"].split()) for r in rows]

    # pairwise agreement between the three annotators
    inter = []
    for i in range(len(ANNOTATORS)):
        for j in range(i + 1, len(ANNOTATORS)):
            inter.append(spearman([r["scores"][i] for r in rows], [r["scores"][j] for r in rows]))
    inter = [v for v in inter if v is not None]

    # Spearman-Brown from one rater to the mean of k=3; its square root is the ceiling
    k, rbar = len(ANNOTATORS), (st.mean(inter) if inter else None)
    rel = (k * rbar / (1 + (k - 1) * rbar)) if rbar and rbar > 0 else None
    ceiling = rel ** 0.5 if rel else None

    rho = spearman(pred, human)
    rho_len = spearman(length, human)
    return dict(
        scorer=SCORER, dataset=DATASET, n_pairs=len(rows), seed=seed,
        validates="our CALL of the frozen scorer: column choice, pair orientation, and the "
                  "two-order arithmetic in _score_pair",
        does_not_validate=["generalisation - the checkpoint was fitted on this dataset, so this "
                           "is in-sample while the paper's 0.845 is 10-fold cross-validated",
                           "the study's own distribution - we score model rewrites of human "
                           "texts, not this dataset's human/LLM paraphrase pairs"],
        rho_scorer_human=round(rho, 4) if rho else None,
        rho_scorer_human_ci=boot_ci(list(zip(pred, human)), seed),
        paper_rho=PAPER_RHO,
        rho_length_human=round(rho_len, 4) if rho_len else None,
        paper_length_baseline=PAPER_LENGTH_BASELINE,
        annotator_agreement=dict(
            mean_pairwise_rho=round(st.mean(inter), 4) if inter else None,
            pairwise=[round(v, 4) for v in inter],
            mean_label_reliability=round(rel, 4) if rel else None,
            attainable_ceiling=round(ceiling, 4) if ceiling else None,
            note="pairwise = how well two annotators agree. mean_label_reliability = "
                 "Spearman-Brown of that to the 3-rater mean, which is the actual target. "
                 "attainable_ceiling = its square root: the highest correlation ANY predictor "
                 "could have with labels this noisy. Compare 0.845 against the ceiling, not 1."),
        by_source=subgroup(rows, "source"),
        by_generator=subgroup(rows, "generator"),
        # the training-free length baseline must reproduce the paper's 0.388
        verdict=("wiring confirmed by the training-free length baseline"
                 if rho_len is not None and abs(rho_len - PAPER_LENGTH_BASELINE) < 0.05 else
                 "LENGTH BASELINE OFF - check the column choice and pair orientation"),
        in_sample_note=(
            f"rho(scorer, human) = {round(rho, 3) if rho else None} is IN-SAMPLE and is NOT "
            f"comparable with the paper's held-out {PAPER_RHO}; the gap is the expected "
            f"train/test difference, not a better result."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, help="first N dataset rows (smoke test)")
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    rows = load_rows(args.limit)
    print(f"{len(rows)} annotated pairs from {DATASET}")
    rows = predict(rows, args)
    res = analyse(rows, args.seed)

    print(f"\nPauli anchor - {res['n_pairs']} pairs, scorer called through the adapter")
    print(f"  rho(scorer, human)   {res['rho_scorer_human']:+.3f}  "
          f"{res['rho_scorer_human_ci']}   paper (10-fold CV): {PAPER_RHO}  IN-SAMPLE, not comparable")
    print(f"  rho(length, human)   {res['rho_length_human']:+.3f}"
          f"{'':>17s}paper baseline:     {PAPER_LENGTH_BASELINE}  <- the decisive check")
    aa = res["annotator_agreement"]
    print(f"  annotator pairwise   {aa['mean_pairwise_rho']:+.3f}"
          f"{'':>17s}reliability of their mean: {aa['mean_label_reliability']}")
    print(f"  attainable ceiling   {aa['attainable_ceiling']:+.3f}"
          f"{'':>17s}(no predictor can beat this)")
    for key, label in (("by_source", "source corpus"), ("by_generator", "paraphrased by")):
        print(f"\n  {label:22s} {'n':>5s} {'rho':>7s}")
        for lv, v in res[key].items():
            print(f"  {lv:22s} {v['n']:5d} {v['rho']:+7.3f}")
    print(f"\n  IN-SAMPLE: the checkpoint was fitted on this data. This checks our call of the "
          f"scorer,\n  not its generalisation, and not its validity on model rewrites.")
    print(f"  -> {res['verdict']}")

    json.dump(res, open(DEST, "w"), indent=1)
    print(f"\nwrote {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
