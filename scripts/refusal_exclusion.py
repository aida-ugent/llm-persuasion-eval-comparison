"""Agreement between methods under four refusal rules, with paired bootstrap intervals on the
differences, and split-half reliability under the exclusion rule, over all nine methods and the
reliable eight and seven. Does not touch matrix.csv, rq1.json or rq2.json.

    python scripts/refusal_exclusion.py            # ~15 min, writes data/derived/refusal_exclusion.json

Rules: floor_non_attempts (primary) scores a refused or unreadable attempt at the floor; exclude
leaves it out; one_cell is the primary rule with fully refused cells dropped; exclude_refusals
drops refusals and prose declines but keeps empty replies at the floor.
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import registry                                                          # noqa: E402
from analysis.build_matrix import (load_all, build, pairwise, bootstrap_pairwise,  # noqa: E402
                                   fisher_mean, ci, N_BOOT_RHO, N_BOOT_REL)

PRIMARY, EXCLUDE, ONE_CELL, REFUSALS = "floor_non_attempts", "exclude", "one_cell", "exclude_refusals"
RULES = (PRIMARY, EXCLUDE, ONE_CELL, REFUSALS)
SUBSETS = {                      # reliable eight = rq1.json's "separating" set; seven = also without MakeMePay
    "all_nine": None,
    "reliable_eight": "separating",
    "reliable_seven": "separating_minus_makemepay",
}


def _pool_mean(units):
    s = sum(sum(v) for v in units.values())
    c = sum(len(v) for v in units.values())
    return s / c if c else None


def _mean_rho(mat, methods):
    return fisher_mean([(mat.get((a, b)) or {}).get("rho") for a, b in itertools.combinations(methods, 2)])


def _subsets(rq1, methods):
    sep = [m for m in rq1["separating"]["methods"] if m in methods]
    return {"all_nine": methods, "reliable_eight": sep,
            "reliable_seven": [m for m in sep if m != "openai2024_makemepay"]}


def main():
    cfg = registry.load()
    seed = (cfg["pinned"] or {}).get("seed", 0)
    rq1 = json.load(open(ROOT / "data" / "derived" / "rq1.json", encoding="utf-8"))
    data = {PRIMARY: load_all(cfg, PRIMARY)[0], EXCLUDE: load_all(cfg, EXCLUDE)[0],
            REFUSALS: load_all(cfg, REFUSALS)[0]}
    methods = [m for m in cfg["methods"] if m in data[PRIMARY]]
    models = list(registry.all_models(cfg))
    dropped = {m: [mo for mo in models if mo in data[PRIMARY][m] and mo not in data[EXCLUDE][m]] for m in methods}
    data[ONE_CELL] = copy.deepcopy(data[PRIMARY])
    for m, mos in dropped.items():
        for mo in mos:
            del data[ONE_CELL][m][mo]

    point, boots = {}, {}
    for p in RULES:
        means = {m: {mo: _pool_mean(d["units"]) for mo, d in data[p][m].items()} for m in methods}
        point[p] = pairwise(means, methods)
        boots[p] = bootstrap_pairwise(data[p], methods, seed, N_BOOT_RHO)
    # the draws are paired only if every rule sees the same unit pools per method
    for m in methods:
        pools = [sorted(set().union(*(d["units"].keys() for d in data[p][m].values()))) for p in RULES]
        assert all(pl == pools[0] for pl in pools), f"{m}: unit pools differ between rules, draws are not paired"

    subsets = _subsets(rq1, methods)
    agreement = {}
    for name, ms in subsets.items():
        mean = {p: _mean_rho(point[p], ms) for p in RULES}
        bm = {p: [_mean_rho(b, ms) for b in boots[p]] for p in RULES}

        def diff(a, b):
            d = [x - y for x, y in zip(bm[a], bm[b])]
            return dict(point=mean[a] - mean[b], ci=ci(d), share_at_or_below_zero=sum(x <= 0 for x in d) / len(d))

        agreement[name] = dict(methods=ms, mean_rho=mean, mean_rho_ci={p: ci(bm[p]) for p in RULES},
                               exclude_minus_primary=diff(EXCLUDE, PRIMARY),
                               one_cell_minus_primary=diff(ONE_CELL, PRIMARY),
                               exclude_minus_one_cell=diff(EXCLUDE, ONE_CELL),
                               refusals_minus_primary=diff(REFUSALS, PRIMARY))
    pairs = {f"{a}|{b}": {**{p: (point[p].get((a, b)) or {}).get("rho") for p in RULES},
                          "n_models_exclude": (point[EXCLUDE].get((a, b)) or {}).get("n_models")}
             for a, b in itertools.combinations(methods, 2)}

    _, rel_ex, warnings = build(cfg, EXCLUDE, rel_boot=N_BOOT_REL, data=data[EXCLUDE])
    rel_primary = rq1["reliability"]
    reliability = {m: dict(primary=rel_primary.get(m, {}).get("reliability"),
                           primary_lo=rel_primary.get(m, {}).get("lo"), primary_hi=rel_primary.get(m, {}).get("hi"),
                           exclude=rel_ex[m]["reliability"], exclude_lo=rel_ex[m]["lo"], exclude_hi=rel_ex[m]["hi"],
                           n_models_exclude=rel_ex[m]["n_models"], n_units_exclude=rel_ex[m]["n_units"])
                   for m in methods}

    out = dict(primary=PRIMARY, alternative=EXCLUDE, one_cell_rule="primary, with the fully refused cells dropped",
               n_boot=N_BOOT_RHO, n_rel_boot=N_BOOT_REL,
               cells_without_score_under_exclude={m: v for m, v in dropped.items() if v},
               agreement=agreement, pairs=pairs, reliability=reliability, warnings=warnings)
    path = ROOT / "data" / "derived" / "refusal_exclusion.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)

    f = lambda x: "-" if x is None else f"{x:.3f}"
    fci = lambda c: f"[{f(c['lo'])}, {f(c['hi'])}]" if c else "[-]"
    print("cells without a score under exclude:", out["cells_without_score_under_exclude"])
    print(f"\n{'subset':15s} {'primary':>8s} {'one cell':>9s} {'exclude':>8s} {'refusals':>9s}   {'excl - prim':>22s}  {'one cell - prim':>22s}  {'excl - one cell':>22s}  {'refusals - prim':>22s}")
    for name, a in agreement.items():
        row = [f"{a['mean_rho'][p]:.3f}" for p in (PRIMARY, ONE_CELL, EXCLUDE, REFUSALS)]
        ds = [f"{d['point']:+.3f} {fci(d['ci'])}" for d in (a["exclude_minus_primary"], a["one_cell_minus_primary"],
                                                            a["exclude_minus_one_cell"], a["refusals_minus_primary"])]
        print(f"{name:15s} {row[0]:>8s} {row[1]:>9s} {row[2]:>8s} {row[3]:>9s}   {ds[0]:>22s}  {ds[1]:>22s}  {ds[2]:>22s}  {ds[3]:>22s}")
    print(f"\n{'method':22s} {'rel primary':>22s} {'rel exclude':>22s}  n_models  n_units")
    for m, r in reliability.items():
        print(f"{m:22s} {f(r['primary']):>6s} [{f(r['primary_lo'])}, {f(r['primary_hi'])}]"
              f" {f(r['exclude']):>6s} [{f(r['exclude_lo'])}, {f(r['exclude_hi'])}]  {r['n_models_exclude']}  {r['n_units_exclude']}")
    for w in warnings:
        print("  warning:", w)
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
