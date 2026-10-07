"""
Runs one method against one model. Writes as it goes, resumes where it stopped.
A row must carry item_id, score, refused and parse_ok; multi-turn methods also `transcript`.

    python -m core.runner --method liu2025 --model llama-3.3-70b
    python -m core.runner --method liu2025 --model llama-3.3-70b --test   # 2 items, no write
    python -m core.runner --method liu2025 --model llama-3.3-70b --fresh  # ignore existing
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from core import clients, registry

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"

REQUIRED_KEYS = {"item_id", "score", "refused", "parse_ok"}


class AdapterError(RuntimeError):
    pass


def out_path(method: str, model: str, tag: str | None = None) -> Path:
    """Output file; a tagged run writes <model>__<tag>.jsonl beside the main file."""
    return RESULTS / method / (f"{model}__{tag}.jsonl" if tag else f"{model}.jsonl")


def read_done(path: Path) -> tuple[dict | None, set]:
    """Return (meta, set of completed item_ids). Tolerates a truncated final line."""
    if not path.exists():
        return None, set()
    meta, done = None, set()
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [resume] ignoring truncated line {i + 1}")
                continue
            if i == 0 and "__meta__" in row:
                meta = row["__meta__"]
            elif "item_id" in row:
                done.add(row["item_id"])
    return meta, done


def done_pairs(path: Path) -> set:
    """(strategy, item) pairs already on disk, for adapters whose unit is a pair (ju2025)."""
    pairs = set()
    if not path.exists():
        return pairs
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "strategy" in r and "item" in r:
            pairs.add((r["strategy"], r["item"]))
    return pairs


def config_matches(old: dict | None, new: dict) -> list[str]:
    """Settings that differ between the existing header and this run."""
    if not old:
        return []
    watch = ("turns", "temperature", "seed", "served_context", "agents",
             "headline_metric", "sampling_unit", "variant")
    drift = [f"{k}: existing={old.get(k)!r} now={new.get(k)!r}"
             for k in watch if k in old and old.get(k) != new.get(k)]
    # n may grow: the seeded sample of n items is contained in the sample at a larger n
    o, nn = old.get("n"), new.get("n")
    if "n" in old and o != nn and not (o in (None, "TBD") or (isinstance(o, int)
                                                                 and isinstance(nn, int) and nn >= o)):
        drift.append(f"n: existing={o!r} now={nn!r} (only TBD -> number or an increase is allowed)")
    return drift


def load_adapter(method: str):
    try:
        mod = importlib.import_module(f"methods.{method}.adapter")
    except ModuleNotFoundError as e:
        raise AdapterError(f"no adapter found at methods/{method}/adapter.py ({e})") from e
    if not hasattr(mod, "run"):
        raise AdapterError(f"methods/{method}/adapter.py has no run(client, spec)")
    return mod.run


def validate_row(row: dict, method: str, idx: int) -> list[str]:
    problems = []
    missing = REQUIRED_KEYS - set(row)
    if missing:
        problems.append(f"item {idx}: missing {sorted(missing)}")
    if row.get("score") is not None and not isinstance(row["score"], (int, float)):
        problems.append(f"item {idx}: score must be a number or null, got {type(row['score']).__name__}")
    if (row.get("refused") or row.get("parse_ok") is False) and row.get("score") is not None:
        problems.append(f"item {idx}: refused/unparseable rows must have score=null; the floor "
                        f"is applied by build_matrix.py, not at collection")
    return problems


def check_bounds(row: dict, spec) -> str | None:
    """A message if the score lies outside the declared metric_range, else None."""
    s, sc = row.get("score"), spec.scale or {}
    if not isinstance(s, (int, float)):
        return None
    rng = sc.get("metric_range")
    if not rng:
        return None                      # unbounded metric (pauli2025)
    lo, hi = rng
    if s > hi:
        return f"score {s} exceeds declared metric_range ceiling {hi}"
    if s < lo:
        return f"score {s} below declared metric_range floor {lo}"
    return None


def run_one(method: str, model: str, cfg: dict, test=False, fresh=False, limit=None,
            tag: str | None = None, overrides: dict | None = None) -> Path | None:
    spec = registry.resolve(cfg, method, model)
    if overrides:
        # Spec fields and pinned roles are set directly; anything else goes to extra
        for k, v in overrides.items():
            if k in ("temperature", "turns", "n") and isinstance(v, str):
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
            if k in spec.agents:
                # re-point a pinned role, e.g. --set persuadee=claude-haiku-4.5
                spec.agents[k] = v
            elif hasattr(spec, k) and k != "extra":
                setattr(spec, k, v)
            else:
                spec.extra[k] = v
    prov = spec.provenance()
    path = out_path(method, model, tag)

    done: set = set()
    if not test and not fresh:
        old_meta, done = read_done(path)
        drift = config_matches(old_meta, prov)
        if drift:
            print(f"[abort] {path.name} was collected under different settings:")
            for d in drift:
                print(f"    {d}")
            print("  Pooling those would mix two experiments. Use --fresh to discard and restart.")
            return None
        if done:
            print(f"[resume] {len(done)} item(s) already done")

    run_fn = load_adapter(method)
    tested = clients.get_client(model, cfg)
    if overrides and "strict_alternation" in overrides:
        tested.strict_alternation = bool(overrides["strict_alternation"])
    agents = {role: clients.get_client(a, cfg, is_pinned=True) if isinstance(a, str)
              else [clients.get_client(x, cfg, is_pinned=True) for x in a]
              for role, a in spec.agents.items()}
    spec.extra["agents_resolved"] = agents
    spec.extra["test_mode"] = test
    spec.extra["cfg"] = cfg
    spec.extra["done"] = done        # item ids adapters skip
    spec.extra["done_pairs"] = set() if (test or fresh) else done_pairs(path)   # ju2025
    spec.extra["limit"] = limit
    if test:
        limit = limit or 2

    # --limit sets the adapter's n
    if limit:
        spec.n = limit

    RESULTS.joinpath(method).mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    counts = dict(written=0, skipped=len(done), refused=0, parse_failures=0,
                  bound_violations=0)
    scores_seen, t0 = [], time.time()

    fh = None
    if not test:
        new_file = fresh or not path.exists()
        fh = open(path, "w" if new_file else "a", encoding="utf-8")
        if new_file:
            fh.write(json.dumps({"__meta__": {**prov, "started": started}}) + "\n")
            fh.flush()

    try:
        for i, row in enumerate(run_fn(tested, spec)):
            if row.get("item_id") in done:
                continue            # counted in `skipped`
            problems = validate_row(row, method, i)
            if problems:
                raise AdapterError("; ".join(problems))
            bound = check_bounds(row, spec)
            if bound:
                counts["bound_violations"] += 1
                row["bound_violation"] = bound
                print(f"  [warn] item {row['item_id']}: {bound}")
            counts["refused"] += bool(row.get("refused"))
            counts["parse_failures"] += (row.get("parse_ok") is False)
            if isinstance(row.get("score"), (int, float)):
                scores_seen.append(row["score"])
            if fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            counts["written"] += 1
            # no row cap: an adapter may yield more rows than n (ju2025: n items x 11 strategies)
    except Exception:
        if fh:
            fh.close()
        print(f"\n[error] {method}/{model} failed after {counts['written']} item(s)")
        traceback.print_exc()
        print(f"\n  Retry just this run:\n"
              f"    python -m core.runner --method {method} --model {model}")
        return None
    finally:
        if fh and not fh.closed:
            fh.close()

    elapsed = time.time() - t0
    n = counts["written"] + counts["skipped"]

    # degenerate run: no scores at all, or identical scores with many refusals/parse failures
    flag_rate = (counts["refused"] + counts["parse_failures"]) / max(1, counts["written"])
    if counts["written"] and not scores_seen:
        print(f"\n[FAIL] no item produced a score: {counts['refused']} refused, "
              f"{counts['parse_failures']} unparseable out of {counts['written']}. "
              f"Nothing was measured; do not put this in the matrix.")
    elif scores_seen and len(set(scores_seen)) == 1:
        msg = f"all {len(scores_seen)} scores are identical ({scores_seen[0]})"
        if flag_rate > 0.5:
            print(f"\n[FAIL] {msg} and {flag_rate:.0%} of items were refused or unparseable. "
                  f"This is a broken run, not a measurement.")
        else:
            print(f"\n[warn] {msg}, but flags are low ({flag_rate:.0%}). Could be genuine - "
                  f"a weak model may truly score at the floor throughout. Compare against other "
                  f"models on this method before trusting it.")

    print(f"\n[done] {method}/{model}  {n} items in {elapsed / 60:.1f} min")
    print(f"       refused={counts['refused']}  parse_failures={counts['parse_failures']}"
          f"  bound_violations={counts['bound_violations']}")
    if tested.served_model and tested.served_model != spec.model_id:
        print(f"       NOTE provider served '{tested.served_model}', we requested '{spec.model_id}'")

    if not test:
        # rewrite the header with the run-level counts
        rows = path.read_text(encoding="utf-8").split("\n")
        meta = json.loads(rows[0])["__meta__"]
        meta.update(finished=datetime.now(timezone.utc).isoformat(),
                    n_written=n, elapsed_sec=round(elapsed, 1),
                    served_model=tested.served_model,
                    refusals=counts["refused"], parse_failures=counts["parse_failures"],
                    bound_violations=counts["bound_violations"],
                    client_stats={"tested": tested.stats,
                                  **{r: (a.stats if hasattr(a, "stats") else
                                         [x.stats for x in a]) for r, a in agents.items()}})
        rows[0] = json.dumps({"__meta__": meta})
        path.write_text("\n".join(rows), encoding="utf-8")
        print(f"       -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", action="store_true", help="2 items, nothing written")
    ap.add_argument("--fresh", action="store_true", help="discard any existing output")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", help="variant run: writes <model>__<tag>.jsonl, never the main file")
    ap.add_argument("--set", action="append", default=[],
                    help="override a method setting for this run, e.g. --set winner_in_prompt=false")
    args = ap.parse_args()
    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = {"true": True, "false": False}.get(v.lower(), v)
    if overrides and not args.tag and not args.test:
        sys.exit("--set changes the experiment; give the run a --tag so it cannot overwrite study data")

    cfg = registry.load()
    problems = registry.validate(cfg) + registry.check_clones(cfg)
    if problems:
        print("[abort] config is invalid:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    r = run_one(cfg=cfg, method=args.method, model=args.model,
                test=args.test, fresh=args.fresh, limit=args.limit, tag=args.tag,
                overrides=overrides)
    sys.exit(0 if r or args.test else 1)


if __name__ == "__main__":
    main()
