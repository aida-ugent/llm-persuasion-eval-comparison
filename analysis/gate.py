"""
The discriminability gate: per method, the spread of model means, the gap within a close pair,
and the n at which reliability s2_between / (s2_between + s2_within / n) reaches 0.90.
Reads data/results/<method>/<model>.jsonl; writes data/derived/gate.json.

    python analysis/gate.py --pair gemma-3-27b,qwen3-32b            # table to stdout
    python analysis/gate.py --pair gemma-3-27b,qwen3-32b --html     # also data/derived/gate.html
    python analysis/gate.py --models a,b,c                          # restrict to these models
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.build_matrix import RESULTS, apply_policy, load_run, unit_of   # noqa: E402
from core import registry                                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Z = 1.96 + 0.84          # alpha 0.05 two-sided, power 0.80
B = 1000
DEFAULT_GAP_FRACTION = 0.10
REL_TARGET = 0.90
K_REL = REL_TARGET / (1 - REL_TARGET)


def unit_means(rows, floor, policy, sampling_unit):
    units = defaultdict(list)
    for r, s in apply_policy(rows, floor, policy):
        units[unit_of(r, sampling_unit)].append(s)
    return [st.mean(v) for v in units.values()]


def boot_se(vals, rng):
    if len(vals) < 2:
        return float("nan")
    return st.pstdev([st.mean(rng.choices(vals, k=len(vals))) for _ in range(B)])


def n_for(sigma, delta):
    if not (sigma > 0 and delta and delta > 0):
        return None
    return int(math.ceil(2 * Z ** 2 * sigma ** 2 / delta ** 2))


def analyse(cfg, want, pair, policy):
    rng = random.Random((cfg["pinned"] or {}).get("seed", 0))
    out = {}
    for method, spec in cfg["methods"].items():
        sc = spec.get("scale") or {}
        floor, rng_m = sc.get("refusal_floor"), sc.get("metric_range")
        unit_key = spec.get("sampling_unit")
        per, n_rows, n_ref, n_bad = {}, 0, 0, 0
        d = RESULTS / method
        paths = sorted(d.glob("*.jsonl")) if d.exists() else []
        mains = {p.stem for p in paths if "__" not in p.stem}
        for path in paths:
            model = path.stem
            if "__" in model:
                # a __freechoice file stands in only when the main file is absent, marked with *
                base, tag = model.split("__", 1)
                if base in mains or tag not in ("freechoice",):
                    continue
                model = base + "*"
            if want and model.rstrip("*") not in want:
                continue
            _, rows = load_run(path)
            um = unit_means(rows, floor, policy, unit_key) if rows else []
            if len(um) < 2:
                continue
            per[model] = dict(mean=st.mean(um), sd=st.pstdev(um), se=boot_se(um, rng), units=len(um))
            n_rows += len(rows)
            n_ref += sum(1 for r in rows if r.get("refused"))
            n_bad += sum(1 for r in rows if r.get("parse_ok") is False)
        if len(per) < 2:
            out[method] = dict(per_model=per, note="fewer than 2 gate models with results")
            continue
        means = {m: v["mean"] for m, v in per.items()}
        spread = max(means.values()) - min(means.values())
        sigma = math.sqrt(st.mean([v["sd"] ** 2 for v in per.values()]))
        # cells with sd 0 (e.g. every row at the refusal floor) stay out of the variances
        live = {m: v for m, v in per.items() if v["sd"] > 0}
        s2_between = st.pvariance([v["mean"] for v in live.values()]) if len(live) >= 2 else None
        s2_within = st.mean([v["sd"] ** 2 for v in live.values()]) if live else None
        n_rel = (int(math.ceil(K_REL * s2_within / s2_between))
                 if s2_between and s2_within else None)
        # implied delta at the configured n (n_rel when none is set)
        n_cfg = spec.get("n") if isinstance(spec.get("n"), int) else n_rel
        implied_delta = (math.sqrt(2 * Z ** 2 * s2_within / n_cfg) if (n_cfg and s2_within) else None)
        adj = abs(means[pair[0]] - means[pair[1]]) if pair and all(p in means for p in pair) else None
        declared = spec.get("min_detectable_gap")
        if declared is None and rng_m:
            declared = DEFAULT_GAP_FRACTION * (rng_m[1] - rng_m[0])
        ceil = flo = None
        if rng_m:
            lo, hi = rng_m
            ceil = sum(1 for v in per.values() if v["mean"] + v["se"] >= hi) / len(per)
            flo = sum(1 for v in per.values() if v["mean"] - v["se"] <= lo) / len(per)
        out[method] = dict(
            per_model=per, spread=spread, sigma=sigma, adjacent_gap=adj,
            s2_between=s2_between, s2_within=s2_within, n_rel=n_rel, implied_delta=implied_delta,
            declared_gap=declared, n_declared=n_for(sigma, declared), n_observed=n_for(sigma, adj),
            ceiling_share=ceil, floor_share=flo,
            refusal_rate=n_ref / max(1, n_rows), parse_fail_rate=n_bad / max(1, n_rows),
            n_current=min(v["units"] for v in per.values()))
    return out


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (
        f"{x:.{d}f}" if isinstance(x, float) else str(x))


def print_table(res, pair):
    print(f"\nDISCRIMINABILITY GATE   close pair: {' vs '.join(pair) if pair else '(none declared)'}\n")
    hdr = (f"{'method':22s} {'models':>6s} {'n now':>5s} {'spread':>7s} {'adj gap':>8s} "
           f"{'n rel.9':>7s} {'delta':>7s} {'n 10%':>6s} {'n obs':>6s} {'ceil':>5s} {'floor':>5s} {'refuse':>6s} {'parse':>5s}")
    print(hdr); print("-" * len(hdr))
    for m, r in res.items():
        if "note" in r:
            print(f"{m:22s} {len(r['per_model']):>6d}   {r['note']}"); continue
        pct = lambda v: "-" if v is None else f"{v:.0%}"
        print(f"{m:22s} {len(r['per_model']):>6d} {r['n_current']:>5d} {fmt(r['spread']):>7s} "
              f"{fmt(r['adjacent_gap']):>8s} {fmt(r['n_rel']):>7s} {fmt(r['implied_delta']):>7s} "
              f"{fmt(r['n_declared']):>6s} {fmt(r['n_observed']):>6s} {pct(r['ceiling_share']):>5s} "
              f"{pct(r['floor_share']):>5s} {r['refusal_rate']:>6.0%} {r['parse_fail_rate']:>5.0%}")
        for mo, v in sorted(r["per_model"].items(), key=lambda kv: -kv[1]["mean"]):
            print(f"    {mo:22s} mean {v['mean']:7.3f}  se {v['se']:6.3f}  units {v['units']}")
    print("\nn rel.9 = items for reliability 0.90: 9 x within-model variance / between-model variance. USE THIS.")
    print("delta   = the gap that n detects at 80% power, in the metric's own units.")
    print("n 10%   = the 10%-of-range rule (undefined for pauli2025); n obs = observed close-pair gap, post hoc.")
    print("ceil/floor = share of gate models within one SE of the metric's bound; refuse/parse pooled over gate models.")


def write_html(res, pair, dest):
    rows = []
    for m, r in res.items():
        if "note" in r:
            rows.append(f"<tr><td>{html.escape(m)}</td><td colspan=12 class=muted>{html.escape(r['note'])}</td></tr>")
            continue
        models = "<br>".join(f"{html.escape(mo)} {v['mean']:.3f} ± {v['se']:.3f}"
                             for mo, v in sorted(r["per_model"].items(), key=lambda kv: -kv[1]["mean"]))
        pct = lambda v: "-" if v is None else f"{v:.0%}"
        warn = ""
        if r["ceiling_share"] and r["ceiling_share"] >= 0.5: warn += " ceiling"
        if r["refusal_rate"] >= 0.25: warn += " refusals"
        if r["n_rel"] is None: warn += " no-variance"
        rows.append(
            f"<tr><td><b>{html.escape(m)}</b>{('<br><span class=warn>' + warn.strip() + '</span>') if warn else ''}</td>"
            f"<td class=models>{models}</td><td>{r['n_current']}</td><td>{fmt(r['spread'])}</td>"
            f"<td>{fmt(r['adjacent_gap'])}</td><td class=big>{fmt(r['n_rel'])}</td>"
            f"<td>{fmt(r['implied_delta'])}</td><td class=muted>{fmt(r['n_declared'])}</td>"
            f"<td class=muted>{fmt(r['n_observed'])}</td>"
            f"<td>{pct(r['ceiling_share'])}</td><td>{pct(r['floor_share'])}</td>"
            f"<td>{r['refusal_rate']:.0%}</td><td>{r['parse_fail_rate']:.0%}</td></tr>")
    dest.write_text(
        "<meta charset='utf-8'><title>Discriminability gate</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:80rem;color:#1a1a1a}"
        "table{border-collapse:collapse;width:100%}th,td{padding:.35rem .6rem;border-bottom:1px solid #ddd;"
        "text-align:right;vertical-align:top}th:first-child,td:first-child,td.models{text-align:left}"
        "td.models{font-size:12px;color:#444}td.big{font-weight:700;font-size:16px}.muted{color:#888}"
        ".warn{color:#c62828;font-size:12px}p{max-width:60rem}</style>"
        "<h1>Discriminability gate</h1>"
        f"<p>Close pair: <b>{html.escape(' vs '.join(pair)) if pair else 'none declared'}</b>. "
        "<b>n (rel 0.90)</b> is the sample size to use: the items at which a "
        "method's reliability reaches 0.90, i.e. 9 x within-model variance / between-model variance from "
        "the gate. Scale-free and defined for every metric. <b>delta</b> is the gap that n detects at 80% "
        "power, in native units. The two grey columns are the 10%-of-range rule and the observed "
        "close-pair gap, for reference. A method flagged <span class=warn>ceiling</span> or "
        "<span class=warn>refusals</span> is NOT dropped; it is reported with the flag.</p>"
        "<table><tr><th>method</th><th>gate models (mean ± se)</th><th>n now</th><th>spread</th>"
        "<th>close-pair gap</th><th>n (rel 0.90)</th><th>delta</th><th>n (10% range)</th><th>n (observed)</th>"
        "<th>ceiling</th><th>floor</th><th>refused</th><th>unparsed</th></tr>" + "".join(rows) + "</table>",
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated aliases; default: all present")
    ap.add_argument("--pair", default=None, help="two aliases expected to be close, e.g. a,b")
    ap.add_argument("--policy", default=None, choices=["floor_non_attempts", "score_at_floor", "exclude"])
    ap.add_argument("--html", action="store_true", help="also write data/derived/gate.html")
    args = ap.parse_args()
    cfg = registry.load()
    policy = args.policy or (cfg["pinned"] or {}).get("refusal_policy", "score_at_floor")
    pair = tuple(args.pair.split(",")) if args.pair else None
    want = args.models.split(",") if args.models else None
    res = analyse(cfg, want, pair, policy)
    print_table(res, pair)
    (ROOT / "data" / "derived" / "gate.json").write_text(
        json.dumps(dict(policy=policy, pair=pair, methods=res), indent=1, default=str), encoding="utf-8")
    if args.html:
        dest = ROOT / "data" / "derived" / "gate.html"
        write_html(res, pair, dest)
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
