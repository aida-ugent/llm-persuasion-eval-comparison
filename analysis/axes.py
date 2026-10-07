"""RQ2 grouping axes: mean rho within vs between groups, a unit-bootstrap interval on the gap,
an exact permutation p, and over FAMILY the Benjamini-Hochberg q and Westfall-Young p.
Reads data/results/, data/derived/rq1.json and rq2.json; --family writes data/derived/rq2_family*.json.

    from analysis.axes import axis_report, axes_table
    axis_report({'pauli2025': 'text', 'elaraby2024': 'text', 'ju2025': 'one-shot', ...})
    axes_table({'my_axis': {...}, 'another': {...}})
    python -m analysis.axes                       # config axes, checked against rq2.json
    python -m analysis.axes --family              # FAMILY with both corrections
    python -m analysis.axes --family --reliable   # the methods with split-half >= 0.80
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import registry                                   # noqa: E402
from analysis.build_matrix import (load_all, pairwise, bootstrap_pairwise, fisher_mean, ci,  # noqa: E402
                                   N_BOOT_RHO)

_CACHE: dict = {}


def point_and_boots(policy=None, n_boot=N_BOOT_RHO):
    """(point, boots, methods): pairwise rho on the full data and per unit resample, cached."""
    cfg = registry.load()
    policy = policy or (cfg["pinned"] or {}).get("refusal_policy", "floor_non_attempts")
    key = (policy, n_boot)
    if key not in _CACHE:
        seed = (cfg["pinned"] or {}).get("seed", 0)
        data, _ = load_all(cfg, policy)
        methods = [m for m in cfg["methods"] if m in data]        # same order as build_matrix.main
        means = {m: {mo: _pool_mean(d["units"]) for mo, d in data[m].items()} for m in methods}
        point = pairwise(means, methods)
        boots = bootstrap_pairwise(data, methods, seed, n_boot)
        _CACHE[key] = (point, boots, methods)
    return _CACHE[key]


def _pool_mean(units):
    s = sum(sum(v) for v in units.values())
    c = sum(len(v) for v in units.values())
    return s / c if c else None


def _rho(mat, a, b):
    e = mat.get((a, b)) or mat.get((b, a))
    return None if e is None else e["rho"]


def gap(mat, labels):
    """(within, between, within - between, n_within, n_between) for one labelling."""
    within, between = [], []
    for a, b in itertools.combinations(labels, 2):
        r = _rho(mat, a, b)
        if r is None:
            continue
        (within if labels[a] == labels[b] else between).append(r)
    w, b = fisher_mean(within), fisher_mean(between)
    d = None if (w is None or b is None) else w - b
    return w, b, d, len(within), len(between)


def _arrangements(labs):
    """Every distinct assignment of the multiset `labs` to positions."""
    counts = Counter(labs)
    n = len(labs)
    out = []

    def rec(prefix):
        if len(prefix) == n:
            out.append(tuple(prefix))
            return
        for lab in counts:
            if counts[lab]:
                counts[lab] -= 1
                prefix.append(lab)
                rec(prefix)
                prefix.pop()
                counts[lab] += 1

    rec([])
    return out


def permutation_p(mat, labels):
    """Exact permutation p of the gap over every relabelling with the same group sizes:
    p_greater = share with gap >= observed, p_two_sided = share with |gap| >= |observed|."""
    methods = list(labels)
    obs = gap(mat, labels)[2]
    if obs is None:
        return None, None, 0, None
    ge = two = 0
    arr = _arrangements(list(labels.values()))
    for a in arr:
        d = gap(mat, dict(zip(methods, a)))[2]
        if d is None:
            continue
        ge += d >= obs - 1e-12
        two += abs(d) >= abs(obs) - 1e-12
    return ge / len(arr), two / len(arr), len(arr), obs


def axis_report(labels, policy=None, n_boot=N_BOOT_RHO, permutation=True):
    """One row for one labelling: the gap, its bootstrap interval and its permutation p."""
    point, boots, _ = point_and_boots(policy, n_boot)
    w, b, d, nw, nb = gap(point, labels)
    diffs = [gap(bm, labels)[2] for bm in boots]
    interval = ci(diffs) or {}
    row = dict(within=w, n_within=nw, between=b, n_between=nb, diff=d,
               ci_lo=interval.get("lo"), ci_hi=interval.get("hi"))
    if permutation:
        pg, p2, nl, _ = permutation_p(point, labels)
        row.update(p_greater=pg, p_two_sided=p2, n_labelings=nl)
    sizes = Counter(labels.values())
    row["groups"] = ", ".join(f"{k}:{v}" for k, v in sorted(sizes.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    return row


def axes_table(axes, **kw):
    """{axis_name: {method: label}} -> pandas DataFrame, one row per axis."""
    import pandas as pd
    rows = {name: axis_report(labels, **kw) for name, labels in axes.items()}
    cols = ["within", "n_within", "between", "n_between", "diff", "ci_lo", "ci_hi",
            "p_greater", "p_two_sided", "n_labelings", "groups"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df[[c for c in cols if c in df.columns]]


def group_means(labels, policy=None, n_boot=N_BOOT_RHO):
    """Per group of one labelling: its methods, number of within pairs, mean rho within."""
    point, _, _ = point_and_boots(policy, n_boot)
    rows = []
    for lab in sorted(set(labels.values()), key=str):
        ms = [m for m in labels if labels[m] == lab]
        rs = [_rho(point, a, b) for a, b in itertools.combinations(ms, 2)]
        rows.append(dict(group=lab, n_methods=len(ms), n_pairs=len(rs),
                         within=fisher_mean(rs), methods=", ".join(ms)))
    return rows


def groups_table(axes, **kw):
    """{axis_name: {method: label}} -> pandas DataFrame, one row per (axis, group)."""
    import pandas as pd
    rows = []
    for name, labels in axes.items():
        for r in group_means(labels, **kw):
            rows.append(dict(axis=name, **r))
    return pd.DataFrame(rows).set_index(["axis", "group"])[["n_methods", "n_pairs", "within", "methods"]]


_ORDER = ["pauli2025", "elaraby2024", "bozdag2025", "ju2025", "liu2025", "openai2024_makemepay",
          "openai2024_makemesay", "ahnert2025", "amayuelas2024"]


def _ax(*labels):
    return dict(zip(_ORDER, labels))


# Every RQ2 grouping, labels in _ORDER; the first eight are the paper's axes.
FAMILY = {
    "scored":           _ax("argument", "argument", "belief", "belief", "belief", "behaviour", "behaviour", "belief", "belief"),
    "scorer":           _ax("trained", "judge", "self", "self", "judge", "rule", "rule", "self", "truth"),
    "format":           _ax("dataset", "dataset", "dialogue", "dialogue", "dialogue", "dialogue", "dialogue", "dialogue", "debate"),
    "task":             _ax("shift", "shift", "shift", "falsehood", "comply", "comply", "comply", "comply", "falsehood"),
    "intent":           _ax("honest", "honest", "honest", "deceptive", "honest", "deceptive", "deceptive", "honest", "deceptive"),
    "topic":            _ax("politics", "politics", "politics", "facts", "everyday", "open", "open", "everyday", "facts"),
    "stake":            _ax("none", "none", "none", "truth", "personal", "money", "none", "personal", "truth"),
    "interaction":      _ax("static", "static", "multi", "single", "multi", "multi", "multi", "multi", "multi"),
    "intent_3way":      _ax("neutral", "neutral", "neutral", "deceptive", "benign", "deceptive", "deceptive", "benign", "deceptive"),
    "claims_actions":   _ax("claims", "claims", "claims", "claims", "action", "action", "action", "action", "claims"),
    "pinned":           _ax("nothing", "judge", "persuadee", "persuadee", "judge", "persuadee", "persuadee", "persuadee", "persuadee"),
    "static_persuadee": _ax("static", "static", "persuadee", "persuadee", "persuadee", "persuadee", "persuadee", "persuadee", "persuadee"),
    "turns":            _ax("single", "single", "short", "single", "long", "long", "long", "short", "short"),
    "length_4":         _ax("static", "static", "short", "single", "long", "long", "long", "short", "short"),
    "length_2":         _ax("short", "short", "short", "short", "long", "long", "long", "short", "short"),
    "granularity":      _ax("continuous", "discrete", "discrete", "binary", "discrete", "continuous", "binary", "discrete", "discrete"),
    "relative":         _ax("relative", "relative", "relative", "absolute", "absolute", "absolute", "absolute", "relative", "relative"),
    "resistance":       _ax("none", "none", "open", "open", "resistant", "resistant", "open", "open", "open"),
}
# resistance: "resistant" = the persuadee is told to hold firm or be cautious.


def bh(ps):
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q, running = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, ps[i] * m / rank)
        q[i] = running
    return q


def westfall_young(mat, axes):
    """Westfall-Young single-step min-P over all orderings of the methods: ({axis: p}, {axis: adjusted p})."""
    import numpy as np
    from analysis.build_matrix import FISHER_CLIP
    methods = list(next(iter(axes.values())))
    k = len(methods)
    pairs = list(itertools.combinations(range(k), 2))
    z = np.zeros((k, k))
    for i, j in pairs:
        z[i, j] = z[j, i] = np.arctanh(np.clip(_rho(mat, methods[i], methods[j]), -FISHER_CLIP, FISHER_CLIP))
    perms = np.array(list(itertools.permutations(range(k))), dtype=np.int8)   # row 0 = observed
    zp = np.stack([z[perms[:, i], perms[:, j]] for i, j in pairs], axis=1)
    n = len(perms)
    p = np.empty((n, len(axes)))
    for c, labels in enumerate(axes.values()):
        w = np.array([labels[methods[i]] == labels[methods[j]] for i, j in pairs])
        g = np.abs(np.tanh(zp[:, w].mean(1)) - np.tanh(zp[:, ~w].mean(1)))
        s = np.sort(g)
        p[:, c] = (n - np.searchsorted(s, g - 1e-12, "left")) / n
    min_p = p.min(1)
    raw = dict(zip(axes, p[0]))
    adj = {a: float((min_p <= p[0, c] + 1e-12).mean()) for c, a in enumerate(axes)}
    return {a: float(v) for a, v in raw.items()}, adj


def family_table(axes=None, **kw):
    """axes_table plus q_bh (Benjamini-Hochberg) and p_wy (Westfall-Young) over every axis in `axes`."""
    axes = axes or FAMILY
    df = axes_table(axes, **kw)
    point, _, _ = point_and_boots(kw.get("policy"), kw.get("n_boot", N_BOOT_RHO))
    raw, adj = westfall_young(point, axes)
    assert all(abs(raw[a] - df.loc[a, "p_two_sided"]) < 1e-9 for a in axes), "min-P disagrees with permutation_p"
    df["q_bh"] = bh(list(df["p_two_sided"]))
    df["p_wy"] = [adj[a] for a in df.index]
    return df


def config_axes():
    """The axes declared in config/methods.yaml, as {axis: {method: label}}."""
    cfg = registry.load()
    _, _, methods = point_and_boots()
    return {ax: {m: cfg["methods"][m].get(ax) for m in methods} for ax in registry.GROUPING_AXES}


def reliable_methods():
    """The methods with split-half reliability >= 0.80, from data/derived/rq1.json."""
    rq1 = json.load(open(ROOT / "data" / "derived" / "rq1.json", encoding="utf-8"))
    return list(rq1["separating"]["methods"])


def main_family(subset=None):
    import pandas as pd
    pd.set_option("display.width", 250)
    axes = FAMILY if subset is None else {a: {m: l for m, l in labs.items() if m in subset}
                                          for a, labs in FAMILY.items()}
    df = family_table(axes)
    print(df.drop(columns="groups").round(3).to_string())
    out = {a: {**{k: (None if pd.isna(v) else v) for k, v in row.items()}, "labels": axes[a]}
           for a, row in df.to_dict(orient="index").items()}
    path = ROOT / "data" / "derived" / ("rq2_family.json" if subset is None else "rq2_family_reliable.json")
    json.dump({"n_axes": len(axes), "p": "two-sided exact permutation",
               "methods": sorted(next(iter(axes.values()))), "axes": out},
              open(path, "w", encoding="utf-8"), indent=1, default=float)
    print(f"\nwrote {path.relative_to(ROOT)}")


def main():
    import pandas as pd
    pd.set_option("display.width", 200)
    if "--family" in sys.argv:
        return main_family(reliable_methods() if "--reliable" in sys.argv else None)
    df = axes_table(config_axes())
    print(df.round(3).to_string())
    rq2 = json.load(open(ROOT / "data" / "derived" / "rq2.json", encoding="utf-8"))["axes"]
    print("\ncheck against rq2.json (diff and its interval):")
    for ax in df.index:
        rep = rq2.get(ax, {}).get("difference", {})
        lo, hi = (rep.get("ci") or {}).get("lo"), (rep.get("ci") or {}).get("hi")
        same = all(abs((x or 0) - (y or 0)) < 1e-9 for x, y in
                   [(df.loc[ax, "diff"], rep.get("point")), (df.loc[ax, "ci_lo"], lo), (df.loc[ax, "ci_hi"], hi)])
        print(f"  {ax:16s} {'ok' if same else 'DIFFERS'}  report diff={rep.get('point')} ci=[{lo}, {hi}]")


if __name__ == "__main__":
    main()
