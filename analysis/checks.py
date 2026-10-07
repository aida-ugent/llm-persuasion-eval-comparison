"""Robustness checks beside the matrix: temperature 0, a second persuadee, capability (raw and
partialled out of the RQ1 correlations), design-factor splits and per-method rescorings.
Reads data/results/, data/capability/summary.csv, data/derived/matrix.csv and rq1.json;
writes data/derived/checks.json.

    python analysis/checks.py              # after build_matrix.py (needs data/derived/matrix.csv)
"""
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402
from analysis.build_matrix import RESULTS, apply_policy, load_run, spearman, study_rows  # noqa: E402

ANALYSIS = registry.ROOT / "data" / "derived"


def _policy(cfg, policy):
    """None -> the primary policy from pinned.yaml."""
    return policy or (cfg["pinned"] or {}).get("refusal_policy", "floor_non_attempts")


def cell_means(cfg, method, tag, policy=None):
    """{model: {item_id: score}} for <model>__<tag>.jsonl, or <model>.jsonl when tag is None."""
    policy = _policy(cfg, policy)
    floor = (cfg["methods"][method].get("scale") or {}).get("refusal_floor")
    out = {}
    for model in registry.all_models(cfg):
        p = RESULTS / method / (f"{model}__{tag}.jsonl" if tag else f"{model}.jsonl")
        if not p.exists():
            continue
        _, rows = load_run(p)
        rows = study_rows(method, rows)
        kept = apply_policy(rows, floor, policy)
        out[model] = {str(r["item_id"]): s for r, s in kept}
    return out


def _bozdag_stance(r):
    """Initial stance on the five-point scale: 1-2 opposed, 3 neutral, 4-5 supporting."""
    k = r.get("initial_rank")
    if k is None:
        return None
    return "opposed" if k <= 2 else "neutral" if k == 3 else "supporting"


# method -> [(design factor, its level in a row)]: conditions upstream varies within the method.
FACTORS = {
    "liu2025":       [("personality", lambda r: r.get("personality"))],
    "ju2025":        [("strategy", lambda r: r.get("strategy"))],
    "bozdag2025":    [("initial_stance", _bozdag_stance)],
    "amayuelas2024": [("dataset", lambda r: r.get("dataset"))],
    # the sampling unit is "<persona index>:<context>"; persona is its prefix
    "ahnert2025":    [("persona", lambda r: (r.get("unit") or "").split(":")[0] or None),
                      ("context", lambda r: r.get("context"))],
}


def factor_split(cfg, method, factor, level_of, policy=None, min_models=3):
    """Model means per factor level; Spearman between level rankings and of each level with the pooled one."""
    policy = _policy(cfg, policy)
    floor = (cfg["methods"][method].get("scale") or {}).get("refusal_floor")
    per_level: dict = {}                       # {level: {model: mean}}
    pooled: dict = {}                          # {model: mean}
    for model in registry.all_models(cfg):
        p = RESULTS / method / f"{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_run(p)
        rows = study_rows(method, rows)
        kept = apply_policy(rows, floor, policy)
        if not kept:
            continue
        pooled[model] = sum(s for _, s in kept) / len(kept)
        by: dict = {}
        for r, s in kept:
            lv = level_of(r)
            if lv is not None:
                by.setdefault(str(lv), []).append(s)
        for lv, ss in by.items():
            per_level.setdefault(lv, {})[model] = sum(ss) / len(ss)

    # models present at every level, so each ranking covers the same models
    models = sorted(m for m in pooled if all(m in per_level[lv] for lv in per_level))
    levels = sorted(lv for lv in per_level if len(per_level[lv]) >= min_models)
    if len(models) < min_models or len(levels) < 2:
        return dict(method=method, factor=factor, status=f"{len(models)} models, {len(levels)} levels; "
                                                         f"need {min_models} and 2")
    rank = {lv: [per_level[lv][m] for m in models] for lv in levels}
    pooled_v = [pooled[m] for m in models]

    between = [spearman(rank[a], rank[b]) for i, a in enumerate(levels) for b in levels[i + 1:]]
    between = [x for x in between if x is not None]
    vs_pooled = {lv: spearman(rank[lv], pooled_v) for lv in levels}
    n_rows = {lv: len(per_level[lv]) for lv in levels}
    return dict(
        method=method, factor=factor, status="ok", n_models=len(models), models=models,
        levels=levels, n_levels=len(levels),
        mean_rho_between_levels=None if not between else round(sum(between) / len(between), 3),
        min_rho_between_levels=None if not between else round(min(between), 3),
        max_rho_between_levels=None if not between else round(max(between), 3),
        rho_pooled_vs_level={lv: None if v is None else round(v, 3) for lv, v in vs_pooled.items()},
        level_means={lv: round(sum(rank[lv]) / len(models), 4) for lv in levels},
        n_models_per_level=n_rows,
        model_means={m: {lv: round(per_level[lv][m], 4) for lv in levels} for m in models})


def _ahnert_persona_labels(cfg):
    """Readable labels for ahnert2025's persona indices, from the target profiles in the clone."""
    try:
        from methods.ahnert2025 import adapter as ahn
        out = {}
        for c in ahn._conditions(cfg):
            i, t = c["unit"].split(":")[0], c["target"]
            out[i] = (f"{t.get('age')} {t.get('gender')}, {t.get('urbanization')}, "
                      f"{t.get('openness')}, {t.get('value_3')}")
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


LEVEL_LABELS = {("ahnert2025", "persona"): _ahnert_persona_labels}


def factor_splits(cfg):
    out = []
    for m, fs in FACTORS.items():
        for f, fn in fs:
            r = factor_split(cfg, m, f, fn)
            lab = LEVEL_LABELS.get((m, f))
            if lab and r.get("status") == "ok":
                r["level_labels"] = lab(cfg)
            out.append(r)
    return out


def paired(cfg, method, tag, exclude=()):
    """Main vs `tag` run on shared item ids: per-model means and Spearman rho; `exclude` drops models."""
    main, var = cell_means(cfg, method, None), cell_means(cfg, method, tag)
    rows = []
    for model in var:
        if model not in main or model in exclude:
            continue
        ids = [i for i in var[model] if i in main[model]]
        if len(ids) < 4:
            continue
        a = sum(main[model][i] for i in ids) / len(ids)
        b = sum(var[model][i] for i in ids) / len(ids)
        rows.append(dict(model=model, n=len(ids), main=round(a, 4), variant=round(b, 4), diff=round(b - a, 4)))
    rho = spearman([r["main"] for r in rows], [r["variant"] for r in rows]) if len(rows) >= 3 else None
    return dict(method=method, tag=tag, rows=rows, n_models=len(rows), rho=None if rho is None else round(rho, 3))


def repeat_on_same_models(cfg, check):
    """Run-to-run rho (main vs __rep2) on the models of `check` only."""
    rep = paired(cfg, check["method"], "rep2")
    models = {r["model"] for r in check["rows"]}
    rows = [r for r in rep["rows"] if r["model"] in models]
    rho = spearman([r["main"] for r in rows], [r["variant"] for r in rows]) if len(rows) >= 3 else None
    return dict(method=check["method"], vs=check["tag"], n_models=len(rows),
                rho=None if rho is None else round(rho, 4), rho_check=check["rho"])


CAP_COLS = ("mmlu_pro_300", "ifeval_prompt_strict")


def _capability_inputs(cfg):
    """(cap, means, models), or (None, None, error); cap[m]['C'] = mean of the benchmarks z-scored across models."""
    cap_p, mat_p = registry.ROOT / "data" / "capability" / "summary.csv", ANALYSIS / "matrix.csv"
    if not cap_p.exists() or not mat_p.exists():
        return None, None, f"missing {cap_p.name if not cap_p.exists() else mat_p.name}"
    cap = {r["model"]: r for r in csv.DictReader(open(cap_p, encoding="utf-8"))}
    means = {}
    for r in csv.DictReader(open(mat_p, encoding="utf-8")):
        if r.get("mean") not in (None, ""):
            means.setdefault(r["method"], {})[r["model"]] = float(r["mean"])
    models = [m for m in registry.all_models(cfg) if m in cap and all(cap[m].get(c) for c in CAP_COLS)]
    import statistics as st
    z = {}
    for c in CAP_COLS:
        vals = [float(cap[m][c]) for m in models]
        mu, sd = st.mean(vals), st.stdev(vals) if len(vals) > 1 else 1.0
        z[c] = {m: (float(cap[m][c]) - mu) / (sd or 1.0) for m in models}
    for m in models:
        cap[m]["C"] = sum(z[c][m] for c in CAP_COLS) / len(CAP_COLS)
    return cap, means, models


def capability(cfg):
    cap, means, models = _capability_inputs(cfg)
    if cap is None:
        return dict(error=models)
    out, ranks = {}, {m: [] for m in models}
    for method, mm in means.items():
        ms = [m for m in models if m in mm]
        if len(ms) < 5:
            continue
        x = [mm[m] for m in ms]
        r_c = spearman(x, [cap[m]["C"] for m in ms])
        out[method] = dict(n_models=len(ms),
                           rho_mmlu_pro=round(spearman(x, [float(cap[m]["mmlu_pro_300"]) for m in ms]), 3),
                           rho_ifeval=round(spearman(x, [float(cap[m]["ifeval_prompt_strict"]) for m in ms]), 3),
                           rho_C=round(r_c, 3),
                           # two-sided t approximation, uncorrected
                           p_C=round(_t_two_sided_p(r_c, len(ms)), 4))
        order = sorted(ms, key=lambda m: mm[m])
        for k, m in enumerate(order):
            ranks[m].append(k + 1)
    agg = [m for m in models if ranks[m]]
    if len(agg) >= 5:
        mr = [sum(ranks[m]) / len(ranks[m]) for m in agg]
        out["mean_rank_across_methods"] = dict(
            n_models=len(agg),
            rho_mmlu_pro=round(spearman(mr, [float(cap[m]["mmlu_pro_300"]) for m in agg]), 3),
            rho_ifeval=round(spearman(mr, [float(cap[m]["ifeval_prompt_strict"]) for m in agg]), 3),
            rho_C=round(spearman(mr, [cap[m]["C"] for m in agg]), 3))
    return out


def _ranks(values):
    """Average ranks, 1 = smallest (ties averaged), as Spearman uses."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _residualise(y, x):
    """Residuals of y after a least-squares line on x."""
    import statistics as st
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx else 0.0
    return [b - (my + slope * (a - mx)) for a, b in zip(x, y)]


def _pearson(x, y):
    """Pearson correlation; None when either side has no spread."""
    import statistics as st
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if not sx or not sy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _t_two_sided_p(r, n):
    """Two-sided p of correlation r over n pairs, t approximation with n - 2 df (as scipy.stats.spearmanr)."""
    import math
    if abs(r) >= 1:
        return 0.0
    df = n - 2
    x = 1 - r * r                               # = df / (df + t^2), with t^2 = r^2 df / (1 - r^2)
    a, b = df / 2, 0.5

    def cf(a, b, x):
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 / ((1 + aa * d) if abs(1 + aa * d) > tiny else tiny)
            c = (1 + aa / c) if abs(1 + aa / c) > tiny else tiny
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 / ((1 + aa * d) if abs(1 + aa * d) > tiny else tiny)
            c = (1 + aa / c) if abs(1 + aa / c) > tiny else tiny
            h *= d * c
            if abs(d * c - 1) < 3e-14:
                break
        return h

    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * cf(a, b, x) / a
    return 1 - front * cf(b, a, 1 - x) / b


def capability_partial(cfg, n_placebo=2000, subset=None):
    """RQ3: the RQ1 pairwise correlations with a capability ranking (C, each benchmark) partialled out.
    Partial Spearman: each method's ranks residualised on the capability ranks, Pearson between residuals.
    Placebo: `n_placebo` random orderings in place of C. `subset`: restrict to these methods."""
    from itertools import combinations
    import numpy as np
    from analysis.build_matrix import fisher_mean
    cap, means, models = _capability_inputs(cfg)
    if cap is None:
        return dict(error=models)
    methods = [m for m in cfg["methods"] if m in means and all(mo in means[m] for mo in models)
               and (subset is None or m in subset)]
    if len(methods) < 2 or len(models) < 5:
        return dict(error="too few methods or models")
    pairs = [f"{a}|{b}" for a, b in combinations(methods, 2)]
    rank_of = {m: _ranks([means[m][mo] for mo in models]) for m in methods}
    raw = {k: spearman(rank_of[k.split("|")[0]], rank_of[k.split("|")[1]]) for k in pairs}

    def partial(control_ranks):
        resid = {m: _residualise(rank_of[m], control_ranks) for m in methods}
        return {k: _pearson(resid[k.split("|")[0]], resid[k.split("|")[1]]) for k in pairs}

    out = dict(rule="standard partial Spearman: each method's model ranks residualised on the "
                    f"capability ranks, Pearson between the residuals; Fisher-z mean over the {len(pairs)} pairs",
               n_models=len(models), methods=methods,
               C={m: round(cap[m]["C"], 4) for m in models},
               raw=dict(mean_rho=round(fisher_mean(list(raw.values())), 4),
                        per_pair={k: round(v, 4) for k, v in raw.items()}),
               by_control={})
    for control in ("C",) + CAP_COLS:
        cr = _ranks([float(cap[mo][control]) for mo in models])
        pp = partial(cr)
        out["by_control"][control] = dict(
            mean_rho=round(fisher_mean(list(pp.values())), 4),
            per_pair={k: round(v, 4) for k, v in pp.items()},
            per_method_rho_with_control={m: round(spearman(rank_of[m], cr), 3) for m in methods})

    obs = partial(_ranks([cap[mo]["C"] for mo in models]))
    change = np.array([obs[k] - raw[k] for k in pairs])
    rng = np.random.default_rng(0)
    D, placebo_means = [], []
    for _ in range(n_placebo):
        pk = partial(_ranks([float(v) for v in rng.permutation(len(models))]))
        D.append([pk[k] - raw[k] for k in pairs])
        placebo_means.append(fisher_mean(list(pk.values())))
    D = np.array(D)
    p_pair = ((np.abs(D) >= np.abs(change) - 1e-12).sum(0) + 1) / (n_placebo + 1)
    mean_c = fisher_mean(list(obs.values()))
    p_mean = (sum(v <= mean_c + 1e-12 for v in placebo_means) + 1) / (n_placebo + 1)
    out["placebo_C"] = dict(
        n_orderings=n_placebo, rng="numpy default_rng(0), one permutation per draw",
        p_mean=round(float(p_mean), 4),
        n_pairs_beyond=int((p_pair < 0.05).sum()), n_pairs=len(pairs),
        expected_by_chance=round(0.05 * len(pairs), 1),
        per_pair_p={k: round(float(v), 4) for k, v in zip(pairs, p_pair)},
        note="p_mean: share of random orderings that lower the mean at least as far as C does; "
             "a pair is beyond the placebo range when fewer than 5% of orderings move it as far")
    n = len(models)
    rng = np.random.default_rng(0)
    d = np.argsort(rng.random((100_000, n)), axis=1) - np.arange(n)
    band = float(np.quantile(np.abs(1 - 6 * (d ** 2).sum(1) / (n * (n * n - 1))), 0.95))
    out["random_ordering_band"] = dict(
        abs_rho_95=round(band, 3), n_models=n, n_draws=100_000, rng="numpy default_rng(0)",
        note="95% of |Spearman rho| between a fixed ranking and a random ordering of the models")
    return out


def benign_offtask(cfg):
    """RQ1 mean rho with liu2025's unread judge replies rescored by `off_task_verdicts`, beside the primary."""
    from itertools import combinations
    import pandas as pd
    from analysis.build_matrix import off_task_verdicts, fisher_mean
    m = "liu2025"
    floor = (cfg["methods"][m].get("scale") or {}).get("refusal_floor")
    policy = _policy(cfg, None)
    means, n_rescored, n_zero = {}, 0, 0
    for model in registry.all_models(cfg):
        p = RESULTS / m / f"{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_run(p)
        rows = off_task_verdicts(study_rows(m, rows))
        n_rescored += sum(1 for r in rows if r.get("verdict_from_rejudge") is not None)
        n_zero += sum(1 for r in rows if r.get("verdict_from_rejudge") == 0)
        kept = apply_policy(rows, floor, policy)
        means[model] = sum(s for _, s in kept) / len(kept)
    mat = pd.read_csv(ANALYSIS / "matrix.csv").pivot(index="model", columns="method", values="mean")
    alt = mat.copy()
    alt[m] = pd.Series(means)
    reliable = json.load(open(ANALYSIS / "rq1.json", encoding="utf-8"))["separating"]["methods"]

    def mean_rho(x, methods):
        return fisher_mean([spearman(list(x[a]), list(x[b])) for a, b in combinations(methods, 2)])
    allm = [c for c in cfg["methods"] if c in mat.columns]
    return dict(n_rescored=n_rescored, n_zero=n_zero,
                mean_rho_all=dict(primary=round(mean_rho(mat, allm), 4), offtask_scored=round(mean_rho(alt, allm), 4)),
                mean_rho_reliable=dict(primary=round(mean_rho(mat, reliable), 4),
                                       offtask_scored=round(mean_rho(alt, reliable), 4)),
                rho_benign_with_primary=round(spearman(list(mat[m]), list(alt[m])), 3))


def makemepay_regex_only(cfg):
    """MakeMePay means counting only regex-detected donations, vs the study means."""
    d = RESULTS / "openai2024_makemepay"
    main, strict = [], []
    rows_out = []
    for model in registry.all_models(cfg):
        p = d / f"{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_run(p)
        rows = study_rows("openai2024_makemepay", rows)
        m = sum(r.get("score") or 0 for r in rows) / len(rows)
        s = sum((r.get("score") or 0) for r in rows if r.get("detected_via") != "summary") / len(rows)
        main.append(m); strict.append(s)
        rows_out.append(dict(model=model, mean=round(m, 2), regex_only=round(s, 2), summary_share=round(m - s, 2)))
    rho = spearman(main, strict) if len(main) >= 3 else None
    return dict(rows=rows_out, rho=None if rho is None else round(rho, 3))


def turn_length_sensitivity(cfg, method, role):
    """Per model: mean words per tested-model turn and mean score; Spearman between the two."""
    d = RESULTS / method
    L, S, rows_out = [], [], []
    for model in registry.all_models(cfg):
        p = d / f"{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_run(p)
        rows = study_rows(method, rows)
        words, scores = [], []
        for r in rows:
            t = r.get("transcript") or []
            msgs = t if isinstance(t, list) else []
            words += [len(str(m.get("content", "")).split()) for m in msgs
                      if isinstance(m, dict) and m.get("role") == role]
            if r.get("score") is not None:
                scores.append(r["score"])
        if not words or not scores:
            continue
        L.append(sum(words) / len(words)); S.append(sum(scores) / len(scores))
        rows_out.append(dict(model=model, mean_words=round(L[-1]), mean_score=round(S[-1], 3)))
    rho = spearman(L, S) if len(L) >= 3 else None
    return dict(method=method, rows=rows_out, rho=None if rho is None else round(rho, 3))


def _nca(initial, final):
    """Normalised change in agreement, as in bozdag2025 evaluate.py:168-171."""
    if final >= initial and initial != 5:
        return (final - initial) / (5 - initial)
    return (final - initial) / (initial - 1)


def bozdag_early_stop(cfg):
    """bozdag2025 with fill-with-5: a conversation that reached 5 mid-way scores as if its final
    rating were 5; `reverted` counts those that ended below 5."""
    d = RESULTS / "bozdag2025"
    main, resc, rows_out = [], [], []
    for model in registry.all_models(cfg):
        p = d / f"{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_run(p)
        rows = study_rows("bozdag2025", rows)
        a, b, reverted = [], [], 0
        for r in rows:
            pr = r.get("persuadee_ranks")
            if r.get("score") is None or not pr or any(x is None for x in pr):
                continue
            hit = len(pr) > 2 and max(pr[1:-1]) == 5
            a.append(float(r["score"]))
            b.append(_nca(pr[0], 5 if hit else pr[-1]))
            reverted += int(hit and pr[-1] < 5)
        if not a:
            continue
        main.append(sum(a) / len(a)); resc.append(sum(b) / len(b))
        rows_out.append(dict(model=model, n=len(a), mean=round(main[-1], 3),
                             early_stop=round(resc[-1], 3), reverted=reverted))
    rho = spearman(main, resc) if len(main) >= 3 else None
    return dict(rows=rows_out, rho=None if rho is None else round(rho, 3))


def bozdag_stable_claims(cfg):
    """bozdag2025 means over the claims whose initial rank is identical in every run, vs the headline."""
    d = RESULTS / "bozdag2025"
    per, init = {}, {}
    for model in registry.all_models(cfg):
        p = d / f"{model}.jsonl"
        if p.exists():
            _, rows = load_run(p)
            per[model] = study_rows("bozdag2025", rows)
    for rows in per.values():
        for r in rows:
            if r.get("initial_rank") is not None:
                init.setdefault(r["item_id"], set()).add(r["initial_rank"])
    stable = {k for k, v in init.items() if len(v) == 1}
    main, sub, rows_out = [], [], []
    for model, rows in per.items():
        a = [float(r["score"]) for r in rows if r.get("score") is not None]
        b = [float(r["score"]) for r in rows if r.get("score") is not None and r["item_id"] in stable]
        if not a or not b:
            continue
        main.append(sum(a) / len(a)); sub.append(sum(b) / len(b))
        rows_out.append(dict(model=model, mean=round(main[-1], 3),
                             stable_claims=round(sub[-1], 3), n_stable=len(b)))
    rho = spearman(main, sub) if len(main) >= 3 else None
    return dict(n_claims=len(init), n_stable=len(stable), rows=rows_out,
                rho=None if rho is None else round(rho, 3))


def validity_bound(anchor="pauli2025", rho_anchor_human=0.845):
    """Bound on each method's rho with humans from its rho with `anchor` (A), given rho(A, H):
    rho(B,H) in rho(AB)*rho(AH) +- sqrt((1 - rho(AB)^2) * (1 - rho(AH)^2)). Reads rq1.json."""
    src = ANALYSIS / "rq1.json"
    if not src.exists():
        return dict(error=f"{src.name} missing; run build_matrix.py --json first")
    pw = json.load(open(src))["all"]["pairwise"]

    def band(r_ab, r_ah):
        c = r_ab * r_ah
        w = (max(0.0, (1 - r_ab ** 2) * (1 - r_ah ** 2))) ** 0.5
        return [round(max(-1.0, c - w), 4), round(min(1.0, c + w), 4)]

    rows = []
    for key, v in pw.items():
        a, b = key.split("|")
        if anchor not in (a, b):
            continue
        other = b if a == anchor else a
        lo, hi = band(v["rho"], rho_anchor_human)
        rows.append(dict(method=other, rho_with_anchor=round(v["rho"], 4), n_models=v["n_models"],
                         bound=[lo, hi], informative=bool(lo > 0 or hi < 0),
                         sensitivity={str(h): band(v["rho"], h) for h in (0.6, 1.0)}))
    rows.sort(key=lambda r: -r["rho_with_anchor"])
    return dict(anchor=anchor, rho_anchor_human=rho_anchor_human,
                anchor_source="pauli2025 scorer, Spearman 0.845 under 10-fold CV "
                              "(length-only baseline 0.388)",
                assumptions=["the anchor's item-level scorer validation is read as a "
                             "ranking-level correlation with a human ranking - optimistic, so "
                             "the bound is a best case",
                             "the correlations are treated as population values; each is "
                             "estimated on 15 models (SE ~ 0.27)"],
                n_informative=sum(r["informative"] for r in rows), rows=rows)


def main():
    cfg = registry.load()
    # temp0 covers the local models; mistral-medium-3.5__temp0.jsonl is not a temperature-0 run
    res = dict(temp0=[paired(cfg, "bozdag2025", "temp0", exclude=("mistral-medium-3.5",))],
               persuadee2=[paired(cfg, m, "persuadee2") for m in ("bozdag2025", "liu2025")],
               capability=capability(cfg))
    for name in ("temp0", "persuadee2"):
        for c in res[name]:
            print(f"\n{name}: {c['method']}  models={c['n_models']}  Spearman rho(main, variant) = {c['rho']}")
            print(f"  {'model':20s} {'n':>3s} {'main':>8s} {'variant':>8s} {'diff':>8s}")
            for r in c["rows"]:
                print(f"  {r['model']:20s} {r['n']:3d} {r['main']:8.3f} {r['variant']:8.3f} {r['diff']:+8.3f}")
    res["temp0_vs_repeat"] = repeat_on_same_models(cfg, res["temp0"][0])
    t = res["temp0_vs_repeat"]
    print(f"\ntemp0 against chance, same {t['n_models']} models: rho(main, temp0) = {t['rho_check']}, "
          f"rho(main, repeat at the study temperature) = {t['rho']}")
    res["makemepay_regex_only"] = makemepay_regex_only(cfg)
    res["benign_offtask"] = benign_offtask(cfg)
    res["turn_length"] = [turn_length_sensitivity(cfg, "bozdag2025", "persuader"),
                          turn_length_sensitivity(cfg, "liu2025", "persuader")]
    mp = res["makemepay_regex_only"]
    print(f"\nmakemepay regex-only: Spearman rho(study ranking, regex-only ranking) = {mp['rho']}")
    for r in mp["rows"]:
        print(f"  {r['model']:20s} mean {r['mean']:6.2f}  regex-only {r['regex_only']:6.2f}  from summariser {r['summary_share']:5.2f}")
    for t in res["turn_length"]:
        print(f"\nturn length vs score, {t['method']}: Spearman rho over models = {t['rho']}")
        for r in t["rows"]:
            print(f"  {r['model']:20s} words/turn {r['mean_words']:5d}  score {r['mean_score']:6.3f}")
    res["factor_splits"] = factor_splits(cfg)
    print("\nwithin-method design factors: does the pooled column hide disagreeing sub-rankings?")
    print(f"  {'method':16s} {'factor':16s} {'lv':>3s} {'mean rho':>9s} {'min':>7s} {'max':>7s}  level means")
    for f in res["factor_splits"]:
        if f.get("status") != "ok":
            print(f"  {f['method']:16s} {f['factor']:16s} {f.get('status')}")
            continue
        g = lambda x: "     -" if x is None else f"{x:+6.2f}"   # noqa: E731
        means = "  ".join(f"{lv}={v:+.2f}" for lv, v in list(f["level_means"].items())[:6])
        print(f"  {f['method']:16s} {f['factor']:16s} {f['n_levels']:3d} {g(f['mean_rho_between_levels'])}"
              f"  {g(f['min_rho_between_levels'])} {g(f['max_rho_between_levels'])}  {means}")
    res["bozdag_early_stop"] = bozdag_early_stop(cfg)
    res["bozdag_stable_claims"] = bozdag_stable_claims(cfg)
    es = res["bozdag_early_stop"]
    print(f"\nbozdag2025 early stop: Spearman rho(headline, fill-with-5 rescoring) = {es['rho']}")
    for r in es["rows"]:
        print(f"  {r['model']:20s} mean {r['mean']:6.3f}  early-stop {r['early_stop']:6.3f}  reverted {r['reverted']:3d}")
    sc = res["bozdag_stable_claims"]
    print(f"\nbozdag2025 stable initial stance: {sc['n_stable']} of {sc['n_claims']} claims identical in every run; "
          f"Spearman rho(headline, stable-claim ranking) = {sc['rho']}")
    for r in sc["rows"]:
        print(f"  {r['model']:20s} mean {r['mean']:6.3f}  stable claims {r['stable_claims']:6.3f}  (n={r['n_stable']})")
    print("\ncapability: Spearman rho between each method's model means and the capability scores")
    print(f"  {'method':28s} {'n':>3s} {'MMLU-Pro':>9s} {'IFEval':>8s} {'C':>7s}")
    for k, v in res["capability"].items():
        if k == "error":
            print("  ", v); break
        print(f"  {k:28s} {v['n_models']:3d} {v['rho_mmlu_pro']:9.3f} {v['rho_ifeval']:8.3f} {v.get('rho_C', float('nan')):7.3f}")
    res["capability_partial"] = capability_partial(cfg)
    reliable = json.load(open(ANALYSIS / "rq1.json", encoding="utf-8"))["separating"]["methods"]
    res["capability_partial_reliable"] = capability_partial(cfg, subset=reliable)
    cp = res["capability_partial"]
    if "error" not in cp:
        print(f"\ncapability partialled out (RQ3): mean pairwise rho over {len(cp['methods'])} methods, {cp['n_models']} models")
        print(f"  raw {cp['raw']['mean_rho']:.3f}   " + "   ".join(f"{c} removed {v['mean_rho']:.3f}" for c, v in cp["by_control"].items()))
        pc, bd = cp["placebo_C"], cp["random_ordering_band"]
        print(f"  placebo for C: p = {pc['p_mean']:.3f}; {pc['n_pairs_beyond']} of {pc['n_pairs']} pairs beyond the "
              f"placebo range (about {pc['expected_by_chance']:.0f} by chance); random-ordering band |rho| < {bd['abs_rho_95']}")
    res["validity_bound"] = validity_bound()
    vb = res["validity_bound"]
    if "error" not in vb:
        print(f"\nhuman-validity bound from the one human-validated method ({vb['anchor']}, "
              f"rho with humans {vb['rho_anchor_human']}):")
        print(f"  {'method':22s} {'rho w/ anchor':>13s}  {'implied bound on rho(method, human)':>35s}")
        for r in vb["rows"]:
            mark = "" if r["informative"] else "   (spans zero - uninformative)"
            print(f"  {r['method']:22s} {r['rho_with_anchor']:+13.2f}  "
                  f"{'[%+.2f, %+.2f]' % tuple(r['bound']):>35s}{mark}")
        print(f"  {vb['n_informative']} of {len(vb['rows'])} bounds exclude zero. "
              f"Assumptions in checks.json.")
    dest = ANALYSIS / "checks.json"
    json.dump(res, open(dest, "w"), indent=1)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
