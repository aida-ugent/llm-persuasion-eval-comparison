"""Reporting checks: whether each ju2025 strategy's and each amayuelas2024 dataset's model ranking
agrees with the pooled ranking, and whether pauli2025 scores track rewrite length (absolute and
relative to the source), with length-adjusted rankings. Uses the matrix builder's loader and
refusal policy. Writes data/derived/pooling_length_checks.json.

    python scripts/pooling_length_checks.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import registry                                                          # noqa: E402
from analysis.build_matrix import (RESULTS, apply_policy, load_run, study_rows,     # noqa: E402
                                   spearman, fisher_mean, _rank)


def scored_rows(cfg, method, policy):
    """{model: [(row, score)]} under the policy, as the matrix sees them."""
    spec = cfg["methods"][method]
    floor = (spec.get("scale") or {}).get("refusal_floor")
    out = {}
    for model in registry.all_models(cfg):
        path = RESULTS / method / f"{model}.jsonl"
        if not path.exists():
            continue
        _, rows = load_run(path)
        if rows:
            out[model] = apply_policy(study_rows(method, rows, "floor"), floor, policy)
    return out


def ranking(means):
    """{model: rank}, 1 = highest mean."""
    models = list(means)
    r = _rank([means[m] for m in models])
    return {m: len(models) + 1 - r[i] for i, m in enumerate(models)}


def rho(a: dict, b: dict):
    ms = [m for m in a if m in b]
    return spearman([a[m] for m in ms], [b[m] for m in ms])


def split_check(data, key):
    """Per-level model means and rankings for one grouping field, against the pooled ranking."""
    pooled = {m: st.mean(s for _, s in rows) for m, rows in data.items()}
    by_level = defaultdict(lambda: defaultdict(list))
    for m, rows in data.items():
        for r, s in rows:
            by_level[r.get(key)][m].append(s)
    levels = {lv: {m: st.mean(v) for m, v in per.items()} for lv, per in by_level.items()}
    rank_pooled = ranking(pooled)
    ranks = {lv: ranking(mm) for lv, mm in levels.items()}
    vs_pooled = {lv: rho(rk, rank_pooled) for lv, rk in ranks.items()}
    pairwise = {f"{a}|{b}": rho(ranks[a], ranks[b]) for a, b in combinations(sorted(ranks), 2)}
    n_rows = {lv: sum(len(v) for v in per.values()) for lv, per in by_level.items()}
    return dict(levels=sorted(levels), n_rows_per_level=n_rows,
                rho_with_pooled=vs_pooled,
                mean_rho_with_pooled=fisher_mean(list(vs_pooled.values())),
                min_rho_with_pooled=min(vs_pooled.values()),
                pairwise_between_levels=pairwise,
                mean_pairwise_between_levels=fisher_mean(list(pairwise.values())),
                means_per_level={lv: {m: round(x, 4) for m, x in mm.items()} for lv, mm in levels.items()},
                pooled_means={m: round(x, 4) for m, x in pooled.items()})


def _words(text):
    return len((text or "").split())


def _length_adjustment(data, measure):
    """Fit score on log(measure) over rows and remove it from each model's mean; None stays unadjusted."""
    per_model_x, per_model_score, xs, ys = {}, {}, [], []
    for m, rows in data.items():
        vals = [measure(r) for r, _ in rows]
        got = [v for v in vals if v]
        per_model_x[m] = st.mean(got) if got else 0.0
        per_model_score[m] = st.mean(s for _, s in rows)
        for v, (_, s) in zip(vals, rows):
            if v:
                xs.append(math.log(v)); ys.append(s)
    mx, my = st.mean(xs), st.mean(ys)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    adjusted = {}
    for m, rows in data.items():
        adjusted[m] = st.mean(s - (beta * (math.log(v) - mx) if v else 0.0)
                              for v, (_, s) in zip((measure(r) for r, _ in rows), rows))
    raw_rank, adj_rank = ranking(per_model_score), ranking(adjusted)
    return dict(model_level_rho=rho(per_model_x, per_model_score), row_level_rho=spearman(xs, ys),
                n_rows=len(xs), slope=beta, per_model_measure=per_model_x,
                raw_means=per_model_score, adjusted_means=adjusted,
                rho_raw_vs_adjusted_ranking=rho(raw_rank, adj_rank),
                raw_ranking=raw_rank, adjusted_ranking=adj_rank)


def length_check(data):
    """Rewrite length (words) against score, over models and over rows; length-adjusted ranking."""
    a = _length_adjustment(data, lambda r: None if r.get("refused") else (_words(r.get("response")) or None))
    src_len = st.mean(_words(r.get("source_text")) for rows in data.values() for r, _ in rows)
    return dict(model_level_rho_length_vs_score=a["model_level_rho"],
                row_level_rho_loglength_vs_score=a["row_level_rho"], n_rows=a["n_rows"],
                ols_slope_score_per_log_length=a["slope"],
                mean_source_length_words=src_len,
                mean_rewrite_length_words={m: round(v, 1) for m, v in a["per_model_measure"].items()},
                raw_means={m: round(v, 4) for m, v in a["raw_means"].items()},
                length_adjusted_means={m: round(v, 4) for m, v in a["adjusted_means"].items()},
                rho_raw_vs_adjusted_ranking=a["rho_raw_vs_adjusted_ranking"],
                raw_ranking=a["raw_ranking"], adjusted_ranking=a["adjusted_ranking"])


def length_ratio_check(data):
    """As length_check, with rewrite words / source words in place of length."""
    def ratio(r):
        n, k = _words(r.get("response")), _words(r.get("source_text"))
        return None if r.get("refused") or n == 0 or k == 0 else n / k
    a = _length_adjustment(data, ratio)
    return dict(model_level_rho_ratio_vs_score=a["model_level_rho"],
                row_level_rho_logratio_vs_score=a["row_level_rho"], n_rows=a["n_rows"],
                ols_slope_score_per_log_ratio=a["slope"],
                mean_length_ratio={m: round(v, 3) for m, v in a["per_model_measure"].items()},
                raw_means={m: round(v, 4) for m, v in a["raw_means"].items()},
                ratio_adjusted_means={m: round(v, 4) for m, v in a["adjusted_means"].items()},
                rho_raw_vs_adjusted_ranking=a["rho_raw_vs_adjusted_ranking"],
                raw_ranking=a["raw_ranking"], adjusted_ranking=a["adjusted_ranking"])


def main():
    cfg = registry.load()
    policy = (cfg["pinned"] or {}).get("refusal_policy", "floor_non_attempts")
    out = {"policy": policy}
    out["ju2025_by_strategy"] = split_check(scored_rows(cfg, "ju2025", policy), "strategy")
    out["amayuelas2024_by_dataset"] = split_check(scored_rows(cfg, "amayuelas2024", policy), "dataset")
    pauli = scored_rows(cfg, "pauli2025", policy)
    out["pauli2025_length"] = length_check(pauli)
    out["pauli2025_length_ratio"] = length_ratio_check(pauli)
    (ROOT / "data" / "derived" / "pooling_length_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    for key, label in (("ju2025_by_strategy", "False Claim, per strategy"),
                       ("amayuelas2024_by_dataset", "Mislead Debate, per dataset")):
        c = out[key]
        print(f"\n{label}: rho of each level's ranking with the pooled ranking")
        for lv in c["levels"]:
            print(f"  {str(lv):24s} n={c['n_rows_per_level'][lv]:5d}  rho={c['rho_with_pooled'][lv]:+.2f}")
        print(f"  mean {c['mean_rho_with_pooled']:.2f}, min {c['min_rho_with_pooled']:.2f}; "
              f"mean pairwise rho between levels {c['mean_pairwise_between_levels']:.2f}")
    p = out["pauli2025_length"]
    print("\nRewrite Text, length")
    print(f"  source texts: {p['mean_source_length_words']:.0f} words on average")
    print(f"  rho(mean rewrite length, mean score) over 15 models: {p['model_level_rho_length_vs_score']:+.2f}")
    print(f"  rho(log length, score) over {p['n_rows']} rows: {p['row_level_rho_loglength_vs_score']:+.2f}; "
          f"OLS slope {p['ols_slope_score_per_log_length']:+.2f} per log-word")
    print(f"  rho(raw ranking, length-adjusted ranking): {p['rho_raw_vs_adjusted_ranking']:.2f}")
    for m in sorted(p["raw_ranking"], key=p["raw_ranking"].get):
        print(f"    {m:20s} words {p['mean_rewrite_length_words'][m]:6.1f}  score {p['raw_means'][m]:+.3f}  "
              f"rank {p['raw_ranking'][m]:4.1f} -> adjusted {p['adjusted_ranking'][m]:4.1f}")
    q = out["pauli2025_length_ratio"]
    print("\nRewrite Text, length relative to the source (rewrite words / source words)")
    print(f"  rho(mean ratio, mean score) over 15 models: {q['model_level_rho_ratio_vs_score']:+.2f}")
    print(f"  rho(log ratio, score) over {q['n_rows']} rows: {q['row_level_rho_logratio_vs_score']:+.2f}; "
          f"OLS slope {q['ols_slope_score_per_log_ratio']:+.2f} per log-ratio")
    print(f"  rho(raw ranking, ratio-adjusted ranking): {q['rho_raw_vs_adjusted_ranking']:.2f}")
    for m in sorted(q["raw_ranking"], key=q["raw_ranking"].get):
        print(f"    {m:20s} ratio {q['mean_length_ratio'][m]:5.2f}  score {q['raw_means'][m]:+.3f}  "
              f"rank {q['raw_ranking'][m]:4.1f} -> adjusted {q['adjusted_ranking'][m]:4.1f}")
    print("\n-> data/derived/pooling_length_checks.json")


if __name__ == "__main__":
    main()
