"""Compare the mean pairwise rho among the five API models (with a bootstrap interval) against every
five-model subset of the ten local models, all subsets and those with no larger capability SD, and
against each window of five models adjacent in capability C. For all nine methods and for the
reliable eight, under the primary rule.

    python scripts/frontier_subsets.py     # a couple of minutes, writes data/derived/frontier_subsets.json
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import registry                                                         # noqa: E402
from analysis.build_matrix import (load_all, pairwise, bootstrap_pairwise, fisher_mean,  # noqa: E402
                                   ci, arm_of, N_BOOT_RHO)
from analysis.checks import _capability_inputs                                    # noqa: E402

PRIMARY = "floor_non_attempts"


def _pool_mean(units):
    s = sum(sum(v) for v in units.values())
    c = sum(len(v) for v in units.values())
    return s / c if c else None


def mean_rho(means, methods, models):
    sub = {m: {mo: v for mo, v in means[m].items() if mo in models} for m in methods}
    p = pairwise(sub, methods, tau=False)
    return fisher_mean([p[q]["rho"] for q in itertools.combinations(methods, 2)])


def main():
    cfg = registry.load()
    seed = (cfg["pinned"] or {}).get("seed", 0)
    rq1 = json.load(open(ROOT / "data" / "derived" / "rq1.json", encoding="utf-8"))
    data, _ = load_all(cfg, PRIMARY)
    every = [m for m in cfg["methods"] if m in data]
    means = {m: {mo: _pool_mean(d["units"]) for mo, d in data[m].items()} for m in every}
    arms = arm_of(cfg)
    models = [mo for mo in registry.all_models(cfg) if mo in arms]
    api = [mo for mo in models if arms[mo] == "api"]
    local = [mo for mo in models if arms[mo] == "local"]
    cap, _, cap_models = _capability_inputs(cfg)
    C = {mo: cap[mo]["C"] for mo in cap_models}
    sd = lambda ms: st.stdev([C[mo] for mo in ms])
    api_sd = sd(api)
    by_c = sorted(cap_models, key=C.get)                     # least capable first

    out = dict(rule=PRIMARY, api_models=api, local_models=local, api_capability_sd=api_sd,
               n_local_subsets=len(list(itertools.combinations(local, 5))))
    for name, key in (("all_nine", "all"), ("reliable_eight", "separating")):
        methods = [m for m in rq1[key]["methods"] if m in every]
        api_rho = mean_rho(means, methods, api)
        ref = rq1[key]["granularities"]["api_arm"]["mean_rho"]
        assert abs(api_rho - ref) < 1e-9, f"{name}: api arm {api_rho} does not reproduce rq1.json {ref}"

        sub_data = {m: {mo: d for mo, d in data[m].items() if mo in api} for m in methods}
        boots = bootstrap_pairwise(sub_data, methods, seed, N_BOOT_RHO)
        boot_means = [fisher_mean([b[q]["rho"] for q in itertools.combinations(methods, 2)]) for b in boots]

        subsets = []
        for s in itertools.combinations(local, 5):
            subsets.append(dict(models=list(s), mean_rho=mean_rho(means, methods, s), capability_sd=sd(s)))
        all_r = [x["mean_rho"] for x in subsets]
        matched = [x["mean_rho"] for x in subsets if x["capability_sd"] <= api_sd]

        windows = []
        for i in range(len(by_c) - 4):
            w = by_c[i:i + 5]
            windows.append(dict(models=w, mean_C=st.mean(C[mo] for mo in w), mean_rho=mean_rho(means, methods, w)))

        out[name] = dict(
            methods=methods,
            api_arm=dict(mean_rho=api_rho, ci=ci(boot_means)),
            local_arm=dict(mean_rho=rq1[key]["granularities"]["local_arm"]["mean_rho"]),
            local_subsets=dict(
                all=dict(n=len(all_r), median=st.median(all_r), min=min(all_r), max=max(all_r),
                         share_at_or_below_api=sum(r <= api_rho for r in all_r) / len(all_r)),
                capability_matched=dict(n=len(matched), median=st.median(matched) if matched else None,
                                        min=min(matched) if matched else None, max=max(matched) if matched else None,
                                        share_at_or_below_api=(sum(r <= api_rho for r in matched) / len(matched))
                                        if matched else None)),
            capability_windows=windows,
            subsets=subsets)

    path = ROOT / "data" / "derived" / "frontier_subsets.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    print(f"API five: {api}  capability SD {api_sd:.2f}")
    for name in ("all_nine", "reliable_eight"):
        o = out[name]
        a, ls = o["api_arm"], o["local_subsets"]
        print(f"\n{name}: API five {a['mean_rho']:.3f} [{a['ci']['lo']:.3f}, {a['ci']['hi']:.3f}]   local ten {o['local_arm']['mean_rho']:.3f}")
        for k in ("all", "capability_matched"):
            s = ls[k]
            if s["n"]:
                print(f"  local five-model subsets, {k:18s} n={s['n']:3d}  median {s['median']:.3f}  range [{s['min']:.3f}, {s['max']:.3f}]"
                      f"  share <= API five {s['share_at_or_below_api']:.2f}")
        print("  windows of five adjacent in capability (least -> most capable):")
        for w in o["capability_windows"]:
            print(f"    mean C {w['mean_C']:+.2f}  rho {w['mean_rho']:.3f}  {', '.join(w['models'])}")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
