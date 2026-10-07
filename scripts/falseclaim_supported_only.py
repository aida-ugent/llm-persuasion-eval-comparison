"""False Claim (ju2025) scored only on the paragraphs labelled A (supports the false answer) by
scripts/falseclaim_judge.py. Reports per-model means, the rho of the A-only ranking with the primary
ranking and with capability C, and the agreement with the other reliable methods. Primary rule
throughout. Nothing here changes the matrix.

    python scripts/falseclaim_supported_only.py    # about a minute, writes data/derived/falseclaim_supported.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import registry                                                         # noqa: E402
from analysis.build_matrix import (RESULTS, load_all, load_run, study_rows, apply_policy,  # noqa: E402
                                   spearman, fisher_mean)
from analysis.checks import _capability_inputs                                    # noqa: E402

M = "ju2025"
PRIMARY = "floor_non_attempts"
LABELS = RESULTS / M / "compliance_labels_v2.jsonl"


def _pool_mean(units):
    s = sum(sum(v) for v in units.values())
    c = sum(len(v) for v in units.values())
    return s / c if c else None


def main():
    cfg = registry.load()
    rq1 = json.load(open(ROOT / "data" / "derived" / "rq1.json", encoding="utf-8"))
    reliable = list(rq1["separating"]["methods"])
    others = [m for m in reliable if m != M]
    data, _ = load_all(cfg, PRIMARY)
    means = {m: {mo: _pool_mean(d["units"]) for mo, d in data[m].items()} for m in reliable}

    labels = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            labels[d["key"]] = d["label"]

    floor = (cfg["methods"][M].get("scale") or {}).get("refusal_floor")
    per = {}
    for model in registry.all_models(cfg):
        p = RESULTS / M / f"{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_run(p)
        kept = apply_policy(study_rows(M, rows), floor, PRIMARY)
        primary = sum(s for _, s in kept) / len(kept)
        assert abs(primary - means[M][model]) < 1e-9, f"{model}: does not reproduce the primary mean"
        a = [s for r, s in kept
             if not r.get("refused") and (r.get("response") or "").strip()
             and labels.get(f"{model}|{r.get('strategy')}|{r.get('item')}") == "A"]
        per[model] = dict(n=len(kept), n_supported=len(a), share_supported=len(a) / len(kept),
                          primary=primary, supported_only=sum(a) / len(a) if a else None)

    models = [mo for mo in per if per[mo]["supported_only"] is not None]
    cap, _, cap_models = _capability_inputs(cfg)
    C = {mo: cap[mo]["C"] for mo in cap_models}
    col = lambda key: [per[mo][key] for mo in models]                      # noqa: E731
    rho = dict(supported_vs_primary=spearman(col("supported_only"), col("primary")),
               supported_with_C=spearman(col("supported_only"), [C[mo] for mo in models]),
               primary_with_C=spearman(col("primary"), [C[mo] for mo in models]),
               share_supported_with_C=spearman(col("share_supported"), [C[mo] for mo in models]),
               n_models=len(models))

    def with_others(score):
        out = {}
        for o in others:
            shared = [mo for mo in models if means[o].get(mo) is not None]
            out[o] = spearman([score[mo] for mo in shared], [means[o][mo] for mo in shared])
        return out

    prim = with_others({mo: per[mo]["primary"] for mo in models})
    supp = with_others({mo: per[mo]["supported_only"] for mo in models})
    swapped = {**means, M: {mo: per[mo]["supported_only"] for mo in models}}

    def mean_pairwise(mm):
        rs = []
        for a, b in itertools.combinations(reliable, 2):
            shared = [mo for mo in mm[a] if mm[a][mo] is not None and mm[b].get(mo) is not None]
            rs.append(spearman([mm[a][mo] for mo in shared], [mm[b][mo] for mo in shared]))
        return fisher_mean(rs)

    agreement = dict(primary=prim, supported_only=supp,
                     mean_with_others_primary=fisher_mean(prim.values()),
                     mean_with_others_supported_only=fisher_mean(supp.values()),
                     reliable_eight_mean_primary=mean_pairwise(means),
                     reliable_eight_mean_supported_only=mean_pairwise(swapped))
    ref = rq1["separating"]
    out = dict(labeller="gpt-4.1-mini (also the listener), labels A supports / B corrects / C hedges / D refuses",
               rule=PRIMARY, per_model=per, rho=rho, agreement=agreement,
               check=dict(reliable_eight_mean_in_rq1=ref.get("mean_rho") or ref.get("mean")))
    path = ROOT / "data" / "derived" / "falseclaim_supported.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    print(f"{'model':20s} {'n':>5s} {'n_A':>5s} {'A share':>8s} {'primary':>8s} {'A only':>7s}")
    for mo in sorted(per, key=lambda m: C.get(m, 0)):
        d = per[mo]
        print(f"{mo:20s} {d['n']:5d} {d['n_supported']:5d} {d['share_supported']:8.3f} {d['primary']:8.3f} "
              f"{d['supported_only'] if d['supported_only'] is None else round(d['supported_only'], 3):>7}")
    print("\nrho:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rho.items()})
    print("rho with the other reliable methods (primary -> A only):")
    for o in others:
        print(f"  {o:22s} {prim[o]:+.2f} -> {supp[o]:+.2f}")
    print(f"  Fisher mean            {agreement['mean_with_others_primary']:+.3f} -> {agreement['mean_with_others_supported_only']:+.3f}")
    print(f"reliable-eight mean pairwise rho: {agreement['reliable_eight_mean_primary']:.3f} -> "
          f"{agreement['reliable_eight_mean_supported_only']:.3f}")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
