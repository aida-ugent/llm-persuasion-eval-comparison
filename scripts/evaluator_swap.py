"""
Evaluator swap: score the saved transcripts of bozdag2025, liu2025 and openai2024_makemepay with
liu2025's 1-5 effectiveness judge prompt, applied by the pinned judge at temperature 0, and compare
the resulting model rankings with each method's native score.

    python scripts/evaluator_swap.py --dry-run              # fake judge, prints one prompt per method
    python scripts/evaluator_swap.py --limit 5              # first 5 items of every file, live judge
    python scripts/evaluator_swap.py                        # everything, then the table + JSON

  E1   Spearman rho between the native and judge model rankings on identical items.
  E0   split-half reliability of each evaluator over sampling units (Spearman-Brown).
  H2   bootstrap of E1 minus the geometric mean of both E0s; rejected if the interval is below zero.

The template fields per method are in FIELDS_DOC. Native scores follow the primary refusal policy.
Judge rows go to data/results/<method>/<model>__judged.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics as st
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # adapters read their data files by repo-relative path

from core import registry                            # noqa: E402
from methods.bozdag2025 import adapter as boz        # noqa: E402
from methods.liu2025 import adapter as liu           # noqa: E402
from methods.openai2024_makemepay import adapter as mmp  # noqa: E402

RESULTS = ROOT / "data" / "results"
OUT_JSON = ROOT / "data" / "derived" / "evaluator_swap.json"
METHODS = ("bozdag2025", "liu2025", "openai2024_makemepay")
NATIVE = {"bozdag2025": "persuadee self-reported stance change (score)",
          "liu2025": "persuadee accept/reject (final_state == ACCEPT)",
          "openai2024_makemepay": "rule-based donation in USD (score)"}
JUDGE_MAX_TOKENS = 1000             # as upstream persuasiveness_eval.py
JUDGE_ATTEMPTS = 3                  # as methods/liu2025/adapter.py
N_SPLITS, N_BOOT, N_INNER_SPLITS = 100, 1000, 20
H2_MARGIN = 0.0   # H2 is rejected if the interval lies entirely below this


def read_jsonl(path: Path):
    """(meta, rows), skipping unreadable lines."""
    meta, rows = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "__meta__" in obj:
                meta = obj["__meta__"]
            elif "item_id" in obj:
                rows.append(obj)
    return meta, rows


def main_files(method: str, cfg: dict, tag: str | None = None) -> list[tuple[str, Path]]:
    """(model, data/results/<method>/<model>.jsonl) per tested model; `tag` reads <model>__<tag>.jsonl."""
    out = []
    for model in registry.all_models(cfg):
        p = RESULTS / method / (f"{model}__{tag}.jsonl" if tag else f"{model}.jsonl")
        if p.exists():
            out.append((model, p))
    return out


def judged_path(method: str, model: str, tag: str | None = None) -> Path:
    return RESULTS / method / (f"{model}__{tag}__judged.jsonl" if tag else f"{model}__judged.jsonl")


class Renderer:
    """Fills liu2025's judge template for any of the three methods."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.jdg = liu._judge_parts(cfg)          # system, user template, output_format
        self._scen = self._boz = self._mmp = None

    def _liu(self, row, meta):
        if self._scen is None:
            rp = ROOT / "methods" / "liu2025" / "repo" / "dataset"
            scen = json.loads((rp / "persuasion_scenarios_neutral_30_strong_rej.json")
                              .read_text(encoding="utf-8"))
            self._scen = {str(s.get("id", i)): s for i, s in enumerate(scen)}
        sid = row.get("scenario_id")
        if sid is None:
            sid = str(row["item_id"]).split("::")[0]
        s = self._scen[str(sid)]
        conv = "".join(f"{'Persuader' if t['role'] == 'persuader' else 'Persuadee'}: "
                       f"{t['content']}\n" for t in row["transcript"])
        return dict(persuasion_task=s["persuasion_task"], persuader_prompt=s["persuader_prompt"],
                    persuadee_prompt=s["persuadee_prompt"], conv_his=conv)

    _BOZ_ROLE = {"persuader": "Persuader", "persuadee": "Persuadee",
                 "persuadee_initial": "Persuadee", "persuadee_final": "Persuadee"}

    @staticmethod
    def _boz_message(text: str) -> str:
        """The <message> body as the adapter parses it, with any <support_ranking> tag removed."""
        msg = boz._parse(text)[0]
        return re.sub(r"<support_ranking>.*?</support_ranking>", "", msg, flags=re.S | re.I).strip()

    def _bozdag(self, row, meta):
        if self._boz is None:
            self._boz = boz._upstream(self.cfg)
        claim = row["claim"]
        conv = "".join(f"{self._BOZ_ROLE[t['role']]}: {self._boz_message(t['content'])}\n"
                       for t in row["transcript"])
        return dict(persuasion_task=claim, persuader_prompt=self._boz.persuader_prompt(claim),
                    persuadee_prompt=self._boz.persuadee_prompt(claim), conv_his=conv)

    def _makemepay(self, row, meta):
        if self._mmp is None:
            descs, _summary, withdraw = mmp._upstream(self.cfg)
            mcfg = self.cfg["methods"]["openai2024_makemepay"]
            variant = (meta or {}).get("variant") or {}
            if not variant:
                print("  [note] makemepay: file has no __meta__.variant; cap / prompt_version / "
                      "allow_withdraw taken from config/methods.yaml")
            version = variant.get("prompt_version") or mcfg.get("prompt_version", "balanced")
            cap = (variant.get("cap") or mcfg.get("cap") or {}).get("value", 6)
            allow_withdraw = variant.get("allow_withdraw", mcfg.get("allow_withdraw", True))
            con = descs[version]["conartist"].format(duration=f"{cap} turns")
            mark = descs[version]["mark"]
            if allow_withdraw:
                con += f"\n{withdraw}"
                mark += f"\n{withdraw}"
            self._mmp = (con, mark)
        con, mark = self._mmp
        conv = "".join(f"{'Persuadee' if t['role'] == 'mark' else 'Persuader'}: {t['content']}\n"
                       for t in row["transcript"])
        return dict(persuasion_task=con, persuader_prompt=con, persuadee_prompt=mark, conv_his=conv)

    def prompt(self, method: str, row: dict, meta: dict | None) -> tuple[str, str]:
        f = {"liu2025": self._liu, "bozdag2025": self._bozdag,
             "openai2024_makemepay": self._makemepay}[method](row, meta)
        user = liu._fill(self.jdg["user"], output_format=self.jdg["output_format"], **f)
        return self.jdg["system"], user


class FakeJudge:
    """--dry-run: liu2025's output format with a score that varies by prompt hash."""
    alias, model_id, served_model = "dry-run-fake-judge", "none", "none"

    def chat(self, messages, max_tokens=512, temperature=0.0, **_):
        h = int(hashlib.sha1(messages[-1]["content"].encode("utf-8")).hexdigest(), 16)
        return types.SimpleNamespace(
            text=f'{{ "Persuasion Effectiveness": [{1 + h % 5}, "dry run - fake judge"] }}\n')


def judge_once(judge, system: str, user: str):
    js, raw = None, ""
    for _ in range(JUDGE_ATTEMPTS):
        r = judge.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                       max_tokens=JUDGE_MAX_TOKENS, temperature=0.0)
        raw = r.text
        js = liu._parse_judge(raw)
        if js is not None:
            break
    return js, raw


def judge_file(method, model, path, rend: Renderer, judge, args, printed: set) -> dict:
    """Judge every transcript in one main file; return {item_id: judge row} (old + new)."""
    meta, rows = read_jsonl(path)
    out = judged_path(method, model, args.tag)
    done: dict = {}
    if out.exists():
        _, old = read_jsonl(out)
        done = {r["item_id"]: r for r in old}
    new, seen = [], 0
    for row in rows:
        if args.limit and seen >= args.limit:
            break
        if not row.get("transcript"):
            continue                              # nothing to judge
        seen += 1
        if row["item_id"] in done:
            continue
        if args.dry_run and method not in printed:
            system, user = rend.prompt(method, row, meta)
            printed.add(method)
            print(f"\n{'=' * 30} {method} / {model} / item {row['item_id']!r} - judge prompt "
                  f"{'=' * 30}\n[system]\n{system}\n\n[user]\n{user}\n{'=' * 90}\n")
        reuse = (method == "liu2025" and not args.recall_liu2025
                 and row.get("judge_score") is not None)
        if reuse:
            jr = dict(item_id=row["item_id"], judge_score=row["judge_score"],
                      judge_raw="reused main-run judge_score (same pinned judge, same prompt, "
                                "temperature 0)")
        else:
            system, user = rend.prompt(method, row, meta)
            js, raw = judge_once(judge, system, user)
            jr = dict(item_id=row["item_id"], judge_score=js, judge_raw=raw)
        new.append(jr)
        if not args.dry_run:
            _append(out, jr, method, model, path, judge, rend)
    calls = sum(1 for r in new if not str(r.get("judge_raw", "")).startswith("reused"))
    print(f"  {method:22s} {model:20s} rows {len(rows):4d}  judged now {len(new):4d} "
          f"({calls} calls)  already {len(done):4d}")
    return {**done, **{r["item_id"]: r for r in new}}


def _append(out: Path, jr: dict, method, model, src: Path, judge, rend: Renderer):
    fresh = not out.exists()
    with open(out, "a", encoding="utf-8") as fh:
        if fresh:
            meta = dict(method=method, model=model, source=src.name,
                        judge=judge.alias, judge_id=judge.model_id,
                        prompt="liu2025 persuasiveness_eval.py sys_eval_prompt + input_prompt, "
                               "extracted from the clone by methods/liu2025/adapter.py",
                        fields=FIELDS_DOC[method], temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,
                        started=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
            fh.write(json.dumps({"__meta__": meta}) + "\n")
        fh.write(json.dumps(jr, ensure_ascii=False) + "\n")


FIELDS_DOC = {
    "liu2025": "scenario fields persuasion_task / persuader_prompt / persuadee_prompt by scenario_id; "
               "transcript raw, as the adapter renders it",
    "bozdag2025": "persuasion_task = row claim; persuader_prompt / persuadee_prompt = clone "
                  "games/prompt.py persuader_prompt(claim) / persuadee_prompt(claim); transcript = "
                  "<message> bodies only (support_ranking withheld: it is the native score)",
    "openai2024_makemepay": "persuasion_task = persuader_prompt = con-artist system prompt as sent "
                            "(CONARTIST_TASK_DESC, duration '{cap} turns', + WITHDRAW_PROMPT); "
                            "persuadee_prompt = mark description (+ WITHDRAW_PROMPT); transcript raw",
}


def spearman(x, y):
    """Tie-aware Spearman rho (Pearson on average ranks); None if either side is constant."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def native_score(method: str, row: dict, floor: float) -> float:
    if method == "liu2025":
        return 1.0 if row.get("final_state") == "ACCEPT" else 0.0
    s = row.get("score")
    return floor if s is None else float(s)


def unit_of(method: str, row: dict):
    if method == "liu2025":
        sid = row.get("scenario_id")
        return str(sid if sid is not None else str(row["item_id"]).split("::")[0])
    return row["item_id"]                         # claim id / episode index


def collect(method, files, judged, floor) -> dict:
    """{model: {unit: [(native, judge)]}} on rows with both scores, after build_matrix.study_rows."""
    from analysis.build_matrix import study_rows
    data = {}
    for model, path in files:
        _, rows = read_jsonl(path)
        rows = study_rows(method, rows)
        per_unit: dict = {}
        for row in rows:
            if row.get("format_failure"):
                continue
            j = judged.get(model, {}).get(row["item_id"])
            if not j or j.get("judge_score") is None:
                continue
            per_unit.setdefault(unit_of(method, row), []).append(
                (native_score(method, row, floor), float(j["judge_score"])))
        if per_unit:
            data[model] = per_unit
    return data


def means(data, models, units, k):
    return [st.mean(v[k] for u in units for v in data[m][u]) for m in models]


def e1(data, models, units):
    return spearman(means(data, models, units, 0), means(data, models, units, 1))


def e0(data, models, units, k, rng, n_splits):
    """Split-half of one evaluator over units; Spearman-Brown applied only to a positive mean rho."""
    rhos = []
    for _ in range(n_splits):
        idx = list(units)
        rng.shuffle(idx)
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        r = spearman(means(data, models, a, k), means(data, models, b, k))
        if r is not None:
            rhos.append(r)
    if not rhos:
        return None
    half = st.mean(rhos)
    return min((2 * half) / (1 + half), 1.0) if half > 0 else half


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def analyse(method, data, seed) -> dict:
    models = sorted(m for m, u in data.items() if len(u) >= 4)
    if len(models) < 3:
        return dict(status=f"only {len(models)} model(s) with >= 4 judged units; need 3")
    units = sorted(set.intersection(*(set(data[m]) for m in models)), key=str)
    if len(units) < 4:
        return dict(status=f"only {len(units)} unit(s) shared across models; need 4")
    rng = random.Random(seed)
    E1 = e1(data, models, units)
    E0n = e0(data, models, units, 0, rng, N_SPLITS)
    E0j = e0(data, models, units, 1, rng, N_SPLITS)

    # comparator: geometric mean of both split-half reliabilities
    def comparator(en, ej):
        if en is None or ej is None or en <= 0 or ej <= 0:
            return None
        return math.sqrt(en * ej)

    E0c = comparator(E0n, E0j)
    diffs, e1s, e0s = [], [], []
    for _ in range(N_BOOT):
        ub = rng.choices(units, k=len(units))
        a = e1(data, models, ub)
        b = comparator(e0(data, models, ub, 0, rng, N_INNER_SPLITS),
                       e0(data, models, ub, 1, rng, N_INNER_SPLITS))
        if a is None or b is None:
            continue
        e1s.append(a); e0s.append(b); diffs.append(a - b)
    ci = [pct(diffs, .025), pct(diffs, .975)] if diffs else None
    if E1 is None or E0c is None:
        verdict = "undefined (a ranking is constant)"
    elif ci is None:
        verdict = "undefined (bootstrap produced no valid draws)"
    elif ci[1] < H2_MARGIN:
        verdict = "rejected"
    else:
        verdict = "supported"   # interval includes zero or lies entirely above it
    return dict(status="ok", models=models, n_models=len(models), n_units=len(units),
                n_rows=sum(len(v) for m in models for u in units for v in [data[m][u]]),
                E1=E1, E0_native=E0n, E0_judge=E0j, E0_comparator=E0c,
                E1_minus_E0_native=None if E1 is None or E0n is None else E1 - E0n,
                E1_minus_E0_comparator=None if E1 is None or E0c is None else E1 - E0c,
                ci95=ci, ci95_E1=[pct(e1s, .025), pct(e1s, .975)] if e1s else None,
                ci95_E0_native=[pct(e0s, .025), pct(e0s, .975)] if e0s else None,
                n_bootstrap_valid=len(diffs), verdict_H2=verdict,
                model_means={m: dict(native=means(data, [m], units, 0)[0],
                                     judge=means(data, [m], units, 1)[0]) for m in models})


def fmt(x, w=6):
    return f"{x:{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"


def print_table(res: dict):
    print(f"\n{'method':22s} {'models':>6s} {'units':>5s} {'E1':>6s} {'E0 nat':>7s} {'E0 jdg':>7s} "
          f"{'E1-E0':>7s}  {'95% CI':17s} H2")
    print("-" * 100)
    for method, r in res.items():
        if r.get("status") != "ok":
            print(f"{method:22s} {r.get('status')}")
            continue
        ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" if r["ci95"] else "-"
        print(f"{method:22s} {r['n_models']:6d} {r['n_units']:5d} {fmt(r['E1'])} "
              f"{fmt(r['E0_native'], 7)} {fmt(r['E0_judge'], 7)} "
              f"{fmt(r['E1_minus_E0_native'], 7)}  {ci:17s} {r['verdict_H2']}")
    ok = [r for r in res.values() if r.get("status") == "ok"]
    if ok:
        allv = [r["verdict_H2"] for r in ok]
        overall = ("supported" if all(v == "supported" for v in allv) and len(ok) == len(METHODS)
                   else "rejected" if any(v == "rejected" for v in allv) else "inconclusive")
        print(f"\nH2 (E1 >= E0 for all three methods, no slack): {overall}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--limit", type=int, help="items per file (testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake judge; print one rendered prompt per method; write nothing")
    ap.add_argument("--recall-liu2025", action="store_true",
                    help="call the judge for liu2025 instead of reusing the main run's judge_score")
    ap.add_argument("--tag", help="TESTING ONLY: read <model>__<tag>.jsonl instead of the main files")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = registry.load()
    seed = args.seed if args.seed is not None else cfg["pinned"].get("seed", 0)
    rend = Renderer(cfg)
    if args.dry_run:
        judge = FakeJudge()
        print("[dry run] fake judge; nothing is written to data/data/results/ or data/derived/")
    else:
        from core.clients import get_client
        judge = get_client(cfg["pinned"]["judge"], cfg, is_pinned=True)
    print(f"judge: {judge.alias} ({judge.model_id})  temperature 0.0  max_tokens {JUDGE_MAX_TOKENS}")

    results, printed = {}, set()
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method not in METHODS:
            sys.exit(f"unknown method {method!r}; choose from {METHODS}")
        files = main_files(method, cfg, args.tag)
        if not files:
            print(f"\n{method}: no {'main' if not args.tag else '__' + args.tag} result files")
            results[method] = dict(status="no result files")
            continue
        print(f"\n{method}: {len(files)} file(s)   native = {NATIVE[method]}")
        judged = {model: judge_file(method, model, path, rend, judge, args, printed)
                  for model, path in files}
        floor = float(cfg["methods"][method]["scale"]["refusal_floor"])
        results[method] = analyse(method, collect(method, files, judged, floor), seed)
        results[method]["native"] = NATIVE[method]
        results[method]["fields"] = FIELDS_DOC[method]

    print_table(results)
    if args.dry_run:
        print("\n[dry run] analysis above is on FAKE judge scores; JSON not written")
        return
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(
        generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        judge=dict(alias=judge.alias, id=judge.model_id, served=judge.served_model),
        settings=dict(n_splits=N_SPLITS, n_bootstrap=N_BOOT, inner_splits=N_INNER_SPLITS,
                      seed=seed,
                      refusal_policy=(cfg["pinned"] or {}).get("refusal_policy", "floor_non_attempts"),
                      h2_margin=H2_MARGIN,
                      limit=args.limit, tag=args.tag),
        methods=results), indent=1), encoding="utf-8")
    print(f"\nwritten {OUT_JSON}")


if __name__ == "__main__":
    main()
