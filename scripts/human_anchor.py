"""
Human anchor: the pinned judge, and upstream's gpt-4-turbo, against elaraby2024's human
annotations of which rationale is more persuasive. Validates the judge substitution only.

    python scripts/human_anchor.py --dry-run              # fake client, writes nothing
    python scripts/human_anchor.py --limit 5              # first 5 items per judge, ~4 cents
    python scripts/human_anchor.py                        # both judges, then the table + JSON
    python scripts/human_anchor.py --judges study         # pinned gpt-4o only (~2 USD)

Each item is judged twice with positions swapped, as the adapter does. Reports per-call and
consensus accuracy and kappa with bootstrap intervals over topics and items, and the paired
difference between the judges. Rows go to data/results/elaraby2024/__human_anchor__<alias>.jsonl
(resumable); statistics go to data/derived/human_anchor.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import statistics as st
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core import registry                              # noqa: E402
from core.adapters import upstream_text                # noqa: E402
from core.clients import _OpenAICompatible, _PROVIDERS  # noqa: E402
from methods.elaraby2024 import adapter as ela         # noqa: E402
from second_judge import kappa                         # noqa: E402

ANNOTATIONS = ("data", "annotated_data", "annotated_rationale_pairs.json")
RESULTS = ROOT / "data" / "results" / "elaraby2024"
DEST = ROOT / "data" / "derived" / "human_anchor.json"
CATS = ("[1]", "[2]", "EQUAL")
FLIP = {"[1]": "[2]", "[2]": "[1]", "EQUAL": "EQUAL"}

# which -> (alias, dated snapshot id, role); None reads the id from models.yaml
JUDGES = {
    "study": ("gpt-4o", None, "the pinned study judge"),
    "upstream": ("gpt-4-turbo", "gpt-4-turbo-2024-04-09", "upstream's judge, elaraby2024"),
}


def load_pairs(cfg) -> tuple[list[dict], list[dict]]:
    """(usable items, dropped rows) from the clone's annotation file."""
    raw = json.loads(upstream_text("elaraby2024", cfg, *ANNOTATIONS))
    items, dropped = [], []
    for i, r in enumerate(raw):
        winner = {"argument 1": 1, "argument 2": 2}.get(str(r["model decision"]).strip().lower())
        human = {"rationale 1": "[1]", "rationale 2": "[2]",
                 "equally persuasive": "EQUAL"}.get(str(r["persuasive rationale"]).strip().lower())
        if winner is None or human is None:
            dropped.append(dict(item_id=i, model_decision=r["model decision"],
                                persuasive_rationale=r["persuasive rationale"]))
            continue
        items.append(dict(item_id=i, topic=r["topic"], arg_1=r["argument 1"],
                          arg_2=r["argument 2"], winner=winner,
                          r1=str(r["rationale 1"]).strip(), r2=str(r["rationale 2"]).strip(),
                          human=human))
    return items, dropped


class FakeClient:
    """--dry-run: a deterministic verdict from the prompt hash."""
    alias, model_id, served_model = "dry-run-fake-anchor", "none", "none"

    def chat(self, messages, max_tokens=512, temperature=0.0, **_):
        h = int(hashlib.sha1(messages[-1]["content"].encode("utf-8")).hexdigest(), 16)
        return types.SimpleNamespace(
            text=json.dumps(dict(reasoning="dry run", decision=CATS[h % 3])))


def build_judge(which: str, cfg, dry_run: bool):
    alias, model_id, _ = JUDGES[which]
    if dry_run:
        return FakeClient()
    if which == "study":                       # the client the collection used
        from core.clients import get_client
        return get_client(alias, cfg, is_pinned=True)
    p = _PROVIDERS["openai"]
    return _OpenAICompatible(alias, model_id, base_url=p["base_url"],
                             api_key_env=p["key"], is_pinned=True)


def ask(judge, tpl, it, first, second):
    """One judge call, as the adapter makes it."""
    prompt = tpl.format(it["topic"], it["arg_1"], it["arg_2"], it["winner"], first, second)
    r = judge.chat([{"role": "system", "content": ela.JUDGE_SYSTEM},
                    {"role": "user", "content": prompt}],
                   max_tokens=ela.JUDGE_MAX_TOKENS, temperature=0.0)
    return ela._decision(ela._parse_json(r.text)), r.text


def consensus(a, b):
    """The adapter's consensus rule; a and b are in the file's frame."""
    if a is None or b is None:
        return None
    wins = (a == "[1]") + (b == "[1]")
    ties = (a == "EQUAL") + (b == "EQUAL")
    s = (wins + 0.5 * ties) / 2
    return "[1]" if s > 0.5 else ("[2]" if s < 0.5 else "EQUAL")


def judge_all(which, items, cfg, tpl, args) -> list[dict]:
    """Rows for one judge: resume from disk, judge what is missing, append as we go."""
    alias, _, _ = JUDGES[which]
    out = RESULTS / f"__human_anchor__{alias}.jsonl"
    done: dict = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip() and "__meta__" not in line[:12]:
                r = json.loads(line)
                done[r["item_id"]] = r

    judge = build_judge(which, cfg, args.dry_run)
    if not args.dry_run and not out.exists():
        RESULTS.mkdir(parents=True, exist_ok=True)
        meta = dict(__meta__=dict(
            script="scripts/human_anchor.py", judge_alias=alias,
            judge_id=getattr(judge, "model_id", None), role=JUDGES[which][2],
            source="/".join(("methods/elaraby2024/repo",) + ANNOTATIONS),
            prompt="src/prompts/pairwise_persuasion_ranking.md (clone) + adapter JUDGE_SYSTEM",
            max_tokens=ela.JUDGE_MAX_TOKENS, temperature=0.0,
            note="validates the judge substitution only; the winner-in-prompt change to the generation stage is "
                 "not tested here",
            written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
        out.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    rows, seen = [], 0
    for it in items:
        if args.limit and seen >= args.limit:
            break
        seen += 1
        if it["item_id"] in done:
            rows.append(done[it["item_id"]])
            continue
        dA, rawA = ask(judge, tpl, it, it["r1"], it["r2"])          # rationale 1 first
        dB, rawB = ask(judge, tpl, it, it["r2"], it["r1"])          # swapped
        dB = FLIP.get(dB) if dB else None                           # back into the file's frame
        row = dict(item_id=it["item_id"], topic=it["topic"], human=it["human"],
                   call_a=dA, call_b=dB, consensus=consensus(dA, dB),
                   raw=[rawA[:400], rawB[:400]])
        rows.append(row)
        if not args.dry_run:
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        if seen % 25 == 0:
            print(f"  {alias}: {seen}/{len(items)}")
    return rows


def _pairs(rows, mode):
    """(judge verdict, human verdict) pairs, dropping what the judge failed to produce."""
    out = []
    for r in rows:
        got = [r["call_a"], r["call_b"]] if mode == "per_call" else [r["consensus"]]
        out += [(g, r["human"], r["topic"]) for g in got if g is not None]
    return out


def _score(pairs):
    if not pairs:
        return dict(n=0, accuracy=None, kappa=None)
    acc = sum(a == b for a, b, _ in pairs) / len(pairs)
    return dict(n=len(pairs), accuracy=round(acc, 4),
                kappa=kappa([a for a, _, _ in pairs], [b for _, b, _ in pairs], CATS))


def _boot(pairs, by, seed, n_boot=1000):
    """Bootstrap accuracy and kappa. `by` is 'topic' (primary) or 'item'."""
    if not pairs:
        return {}
    rng = random.Random(seed)
    if by == "topic":
        groups: dict = {}
        for p in pairs:
            groups.setdefault(p[2], []).append(p)
        units = list(groups.values())
    else:
        units = [[p] for p in pairs]
    accs, kaps = [], []
    for _ in range(n_boot):
        draw = [p for _ in units for p in rng.choice(units)]
        s = _score(draw)
        accs.append(s["accuracy"])
        if s["kappa"] is not None:
            kaps.append(s["kappa"])
    q = lambda xs, f: round(sorted(xs)[int(f * (len(xs) - 1))], 4) if xs else None
    return dict(n_units=len(units),
                accuracy_ci=[q(accs, 0.025), q(accs, 0.975)],
                kappa_ci=[q(kaps, 0.025), q(kaps, 0.975)])


def analyse(by_judge: dict, items, dropped, seed) -> dict:
    labels = [it["human"] for it in items]
    majority = max(labels.count(c) for c in CATS) / len(labels) if labels else None
    res = dict(
        source="/".join(("methods/elaraby2024/repo",) + ANNOTATIONS),
        validates="the judge substitution only (methods.yaml deviation `judge`, severity green)",
        does_not_validate="the winner-in-prompt task deviation, which changed the generation stage",
        feeds_nothing="independent of checks.json `validity_bound`, which is anchored on "
                      "pauli2025's published 0.845 and never reads this file; item-level "
                      "agreement and a model-ranking bound cannot be chained",
        n_items=len(items), n_topics=len({it["topic"] for it in items}),
        dropped=dropped,
        human_label_counts={c: labels.count(c) for c in CATS},
        majority_class_baseline=round(majority, 4) if majority else None,
        judges={}, seed=seed)

    for which, rows in by_judge.items():
        alias, _, role = JUDGES[which]
        entry = dict(alias=alias, role=role,
                     unparsed_calls=sum((r["call_a"] is None) + (r["call_b"] is None) for r in rows),
                     order_disagreement=round(
                         sum(r["call_a"] != r["call_b"] for r in rows) / len(rows), 4) if rows else None)
        for mode in ("per_call", "consensus"):
            p = _pairs(rows, mode)
            entry[mode] = _score(p)
            entry[mode]["bootstrap_topic"] = _boot(p, "topic", seed)
            entry[mode]["bootstrap_item"] = _boot(p, "item", seed)
        res["judges"][which] = entry

    if len(by_judge) == 2:
        a = {r["item_id"]: r["consensus"] for r in by_judge["study"]}
        b = {r["item_id"]: r["consensus"] for r in by_judge["upstream"]}
        both = [(a[i], b[i]) for i in a if i in b and a[i] and b[i]]
        res["judge_vs_judge"] = dict(
            n=len(both),
            agreement=round(sum(x == y for x, y in both) / len(both), 4) if both else None,
            kappa=kappa([x for x, _ in both], [y for _, y in both], CATS) if both else None,
            note="consensus verdicts, same items; this is the ceiling any human agreement can "
                 "be read against")
        res["paired_difference"] = paired_difference(by_judge, seed)
    return res


def paired_difference(by_judge, seed, n_boot=1000):
    """Study minus upstream judge in kappa and accuracy on shared items, bootstrapped over topics."""
    st_rows = {r["item_id"]: r for r in by_judge["study"]}
    up_rows = {r["item_id"]: r for r in by_judge["upstream"]}
    shared = [(st_rows[i], up_rows[i]) for i in st_rows
              if i in up_rows and st_rows[i]["consensus"] and up_rows[i]["consensus"]]
    if not shared:
        return dict(error="no shared items with a consensus verdict from both judges")

    def stats(rows):
        hs = [s["human"] for s, _ in rows]
        ka = kappa([s["consensus"] for s, _ in rows], hs, CATS)
        kb = kappa([u["consensus"] for _, u in rows], hs, CATS)
        aa = sum(s["consensus"] == s["human"] for s, _ in rows) / len(rows)
        ab = sum(u["consensus"] == u["human"] for _, u in rows) / len(rows)
        return ka, kb, aa, ab

    ka, kb, aa, ab = stats(shared)

    groups: dict = {}
    for pair in shared:
        groups.setdefault(pair[0]["topic"], []).append(pair)
    units = list(groups.values())

    rng = random.Random(seed)
    dk, da = [], []
    for _ in range(n_boot):
        draw = [p for _ in units for p in rng.choice(units)]
        k1, k2, a1, a2 = stats(draw)
        if k1 is not None and k2 is not None:
            dk.append(k1 - k2)
        da.append(a1 - a2)
    q = lambda xs, f: round(sorted(xs)[int(f * (len(xs) - 1))], 4) if xs else None  # noqa: E731
    return dict(
        n_items=len(shared), n_units=len(units),
        kappa_study=round(ka, 4) if ka is not None else None,
        kappa_upstream=round(kb, 4) if kb is not None else None,
        kappa_difference=round(ka - kb, 4) if None not in (ka, kb) else None,
        kappa_difference_ci=[q(dk, 0.025), q(dk, 0.975)],
        accuracy_study=round(aa, 4), accuracy_upstream=round(ab, 4),
        accuracy_difference=round(aa - ab, 4),
        accuracy_difference_ci=[q(da, 0.025), q(da, 0.975)],
        excludes_zero=bool(dk and q(dk, 0.025) > 0),
        note="consensus verdicts on identical items; positive favours the pinned study judge. "
             "Bootstrap over topic, of which there are seven.")


def print_table(res):
    print(f"\nHuman anchor - elaraby2024, {res['n_items']} annotated items over "
          f"{res['n_topics']} topics")
    print(f"  human labels {res['human_label_counts']}; majority-class baseline "
          f"{res['majority_class_baseline']}")
    if res["dropped"]:
        print(f"  dropped {len(res['dropped'])}: {res['dropped']}")
    print(f"\n  {'judge':12s} {'mode':10s} {'n':>5s} {'acc':>6s} {'acc 95% (topic)':>18s} "
          f"{'kappa':>7s} {'kappa 95% (topic)':>19s}")
    for which, e in res["judges"].items():
        for mode in ("per_call", "consensus"):
            m = e[mode]
            b = m["bootstrap_topic"]
            k = f"{m['kappa']:.3f}" if m["kappa"] is not None else "n/a"
            ci = lambda x: f"[{x[0]}, {x[1]}]" if x else "n/a"
            print(f"  {e['alias']:12s} {mode:10s} {m['n']:5d} {m['accuracy']:6.3f} "
                  f"{ci(b.get('accuracy_ci')):>18s} {k:>7s} {ci(b.get('kappa_ci')):>19s}")
        print(f"  {'':12s} order-disagreement {e['order_disagreement']}, "
              f"unparsed calls {e['unparsed_calls']}")
    if "judge_vs_judge" in res:
        j = res["judge_vs_judge"]
        print(f"\n  judge vs judge (consensus, n={j['n']}): agreement {j['agreement']}, "
              f"kappa {j['kappa']:.3f}" if j["kappa"] is not None else "")
    print("\n  Validates the judge substitution ONLY. The generation-stage change is not "
          "tested by this and must not be described as validated.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judges", default="study,upstream",
                    help="study | upstream | study,upstream (default)")
    ap.add_argument("--limit", type=int, help="items per judge (testing)")
    ap.add_argument("--dry-run", action="store_true", help="fake client; write nothing")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = registry.load()
    seed = args.seed if args.seed is not None else cfg["pinned"].get("seed", 0)
    tpl = upstream_text("elaraby2024", cfg, "src", "prompts", "pairwise_persuasion_ranking.md")
    items, dropped = load_pairs(cfg)
    print(f"{len(items)} annotated items, {len(dropped)} dropped, "
          f"{len({it['topic'] for it in items})} topics")

    by_judge = {}
    for which in [w.strip() for w in args.judges.split(",") if w.strip()]:
        if which not in JUDGES:
            raise SystemExit(f"unknown judge '{which}'; pick from {list(JUDGES)}")
        print(f"\njudging with {JUDGES[which][0]} ({JUDGES[which][2]})")
        by_judge[which] = judge_all(which, items, cfg, tpl, args)

    res = analyse(by_judge, items[:args.limit] if args.limit else items, dropped, seed)
    print_table(res)
    if args.dry_run:
        print("[dry run] nothing written")
        return
    json.dump(res, open(DEST, "w"), indent=1)
    print(f"wrote {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
