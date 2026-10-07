"""
Shared helpers for adapters.

  upstream_text()   loads a prompt verbatim from the clone.
  stratified()      samples n items across a grouping key.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def repo_path(method: str, cfg: dict) -> Path:
    """Path of this method's clone; `upstream.repo_path` overrides the default."""
    declared = ((cfg["methods"].get(method) or {}).get("upstream") or {}).get("repo_path")
    if declared and declared != "none":
        return ROOT / declared
    return ROOT / "methods" / method / "repo"


def upstream_text(method: str, cfg: dict, *relative) -> str:
    """Read a file from the clone verbatim."""
    p = repo_path(method, cfg).joinpath(*relative)
    if not p.exists():
        raise FileNotFoundError(
            f"{method}: expected upstream file at {p}. Adapters must load prompts from the "
            f"clone, not restate them.")
    return p.read_text(encoding="utf-8")


def upstream_json(method: str, cfg: dict, *relative):
    return json.loads(upstream_text(method, cfg, *relative))


def upstream_module(method: str, cfg: dict, *relative):
    """Load one .py file from the clone as a standalone module, without its package."""
    import importlib.util
    p = repo_path(method, cfg).joinpath(*relative)
    if not p.exists():
        raise FileNotFoundError(f"{method}: expected upstream module at {p}")
    name = f"_upstream_{method}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stratified(items: list, n: int | None, key=None, seed: int = 20261002) -> list:
    """Take n items spread across the values of `key`; a seeded random sample if key is None."""
    if n is not None and not isinstance(n, int):
        raise TypeError(
            f"stratified: n must be an int or None, got {n!r}. A method whose n is still 'TBD' "
            f"cannot be run without --limit, which sets n explicitly.")
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    if key is None:
        out = list(items)
        rng.shuffle(out)
        return sorted(out[:n], key=lambda x: items.index(x))

    groups: dict = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    for g in groups.values():
        rng.shuffle(g)

    # round-robin across groups
    chosen, names = [], sorted(groups)
    i = 0
    while len(chosen) < n:
        progressed = False
        for gname in names:
            if len(chosen) >= n:
                break
            if i < len(groups[gname]):
                chosen.append(groups[gname][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return chosen


def done_ids(spec) -> set:
    """Item ids already on disk for this run; adapters skip them before calling any model."""
    return spec.extra.get("done") or set()


def row(item_id, score, *, refused=False, parse_ok=True, response=None,
        transcript=None, **extra) -> dict:
    """Build a result row; a refused or unparseable item gets score None."""
    if refused or not parse_ok:
        score = None
    r = dict(item_id=item_id, score=score, refused=bool(refused), parse_ok=bool(parse_ok))
    if response is not None:
        r["response"] = response
    if transcript is not None:
        r["transcript"] = transcript
    r.update(extra)
    return r
