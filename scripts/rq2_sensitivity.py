"""RQ2 robustness over the eight reliable methods: the within/between gap of each grouping with its
exact two-sided permutation p and Westfall-Young adjusted p, on the primary correlations, the
capability-partialled ones and the exclusion rule; the intent grouping under each coding of
Benign Request (liu2025) and without it; and the intent and stake gaps with each method left out.

    python scripts/rq2_sensitivity.py      # a few minutes, writes data/derived/rq2_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.axes import FAMILY, gap, permutation_p, westfall_young, point_and_boots, reliable_methods  # noqa: E402

BENIGN = "liu2025"
LOO_AXES = ("intent", "stake")


def restrict(family, methods):
    return {a: {m: labels[m] for m in methods} for a, labels in family.items()}


def table(mat, family):
    """{axis: within, between, gap, p, p_wy, n_labelings} for every axis of a family."""
    raw, adj = westfall_young(mat, family)
    out = {}
    for a, labels in family.items():
        w, b, d, _, _ = gap(mat, labels)
        _, p2, n, _ = permutation_p(mat, labels)
        assert abs(p2 - raw[a]) < 1e-9, f"{a}: min-P disagrees with permutation_p"
        out[a] = dict(within=w, between=b, gap=d, p=p2, p_wy=adj[a], n_labelings=n)
    return out


def from_pairs(per_pair):
    """checks.json per_pair {"a|b": rho} -> the {(a, b): {"rho": r}} dict axes.py reads."""
    return {tuple(k.split("|")): {"rho": v} for k, v in per_pair.items()}


def main():
    methods = reliable_methods()
    family = restrict(FAMILY, methods)
    out = dict(methods=methods, p="two-sided exact permutation; p_wy = Westfall-Young single-step min-P")

    # primary, checked against rq2_family_reliable.json
    point, _, _ = point_and_boots(None, 1)
    out["primary"] = table(point, family)
    ref = json.load(open(ROOT / "data" / "derived" / "rq2_family_reliable.json", encoding="utf-8"))["axes"]
    for a, r in out["primary"].items():
        assert abs(r["gap"] - ref[a]["diff"]) < 1e-9 and abs(r["p"] - ref[a]["p_two_sided"]) < 1e-9 \
            and abs(r["p_wy"] - ref[a]["p_wy"]) < 1e-9, f"{a}: does not reproduce rq2_family_reliable.json"

    # capability-partialled correlations
    cap = json.load(open(ROOT / "data" / "derived" / "checks.json", encoding="utf-8"))["capability_partial_reliable"]
    assert sorted(cap["methods"]) == sorted(methods), "checks.json was run on a different method set"
    out["capability"] = table(from_pairs(cap["by_control"]["C"]["per_pair"]), family)
    out["capability_raw_check"] = table(from_pairs(cap["raw"]["per_pair"]), family)["intent"]   # rounded rhos

    # refused, empty and unreadable attempts excluded
    point_ex, _, _ = point_and_boots("exclude", 1)
    out["exclude"] = table(point_ex, family)

    # the Benign Request coding of the intent grouping
    codings = {"rational (paper)": "honest", "manipulative": "deceptive", "own group": "permitted"}
    br = {}
    for name, lab in codings.items():
        fam = {**family, "intent": {**family["intent"], BENIGN: lab}}
        br[name] = table(point, fam)["intent"]
    rest = [m for m in methods if m != BENIGN]
    br["left out"] = table(point, restrict(FAMILY, rest))["intent"]
    out["benign_request"] = br

    # leave one method out, intent and stake
    loo = {}
    for m in methods:
        keep = [x for x in methods if x != m]
        fam = restrict({a: FAMILY[a] for a in LOO_AXES}, keep)
        loo[m] = {}
        for a in LOO_AXES:
            w, b, d, _, _ = gap(point, fam[a])
            _, p2, n, _ = permutation_p(point, fam[a])
            loo[m][a] = dict(within=w, between=b, gap=d, p=p2, n_labelings=n)
    out["leave_one_out"] = loo

    path = ROOT / "data" / "derived" / "rq2_sensitivity.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    f = lambda x: "  -  " if x is None else f"{x:5.2f}"
    fp = lambda x: "  -  " if x is None else f"{x:5.3f}"
    print(f"{'axis':18s} | {'primary gap  p    p_wy':24s} | {'capability gap  p    p_wy':27s} | {'exclude gap  p    p_wy':24s}")
    for a in family:
        cells = [out[k][a] for k in ("primary", "capability", "exclude")]
        print(f"{a:18s} | " + " | ".join(f"{f(c['gap'])} {fp(c['p'])} {fp(c['p_wy'])}      " for c in cells))
    print("\nintent, Benign Request coded as:")
    for name, r in br.items():
        print(f"  {name:18s} within {f(r['within'])} between {f(r['between'])} gap {f(r['gap'])} p {fp(r['p'])} p_wy {fp(r['p_wy'])}")
    print("\nleave one method out (gap, p):")
    for m, r in loo.items():
        print(f"  without {m:22s} " + "  ".join(f"{a} {f(r[a]['gap'])} {fp(r[a]['p'])}" for a in LOO_AXES))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
