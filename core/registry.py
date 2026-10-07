"""
Config loading and validation. The only place defaults live.

    python -m core.registry --check          # validate everything, print the resolved table
    python -m core.registry --check --strict # also fail on TBDs (use before collection)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"

REQUIRED_METHOD_FIELDS = ("n", "sampling_unit", "headline_metric", "scale", "roles")
PINNED_ROLES = ("persuadee", "judge")
# method settings recorded as the header's `variant`; a resume aborts if one changes
PROMPT_VARIANT_KEYS = ("winner_in_prompt", "unethical_strategies", "questionnaire",
                       "questionnaire_samples", "allow_withdraw", "n_agents",
                       "n_rounds", "word_list", "prompt_version", "cap")

# RQ2 grouping axes and the expected number of methods per label.
GROUPING_AXES = {
    "construct":       {"belief_change": 5, "behavioural": 2, "argument": 2},
    "taxonomy_family": {"dialogue": 6, "dataset": 2, "debate": 1},
    "evaluator":       {"llm_judge": 2, "self_reported": 3, "rule_based": 2,
                        "supervised": 1, "ground_truth": 1},
    "grouping":        {"receiver_effect": 7, "message_property": 2},
    "task_type":       {"shift_stance": 3, "accept_falsehood": 2, "comply": 4},
    "legitimacy":      {"honest": 5, "deception": 4},
    "stake":           {"none": 4, "truth": 2, "personal": 2, "money": 1},
    "domain":          {"politics_society": 3, "factual_knowledge": 2, "everyday_life": 2,
                        "open_conversation": 2},
    "speech_act":      {"assertive": 5, "commissive": 4},
}


class ConfigError(Exception):
    pass


@dataclass
class Spec:
    """Everything one run needs."""
    method: str
    model: str
    model_id: str
    n: Any
    sampling_unit: str
    headline_metric: str
    scale: dict
    temperature: Any
    turns: Any
    seed: int
    served_context: int
    agents: dict = field(default_factory=dict)   # resolved pinned agents, by role
    upstream: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def provenance(self) -> dict:
        """The __meta__ header of the output file."""
        return dict(
            method=self.method, model=self.model, model_id=self.model_id,
            n=self.n, sampling_unit=self.sampling_unit,
            headline_metric=self.headline_metric, scale=self.scale,
            temperature=self.temperature, turns=self.turns, seed=self.seed,
            served_context=self.served_context,
            agents={r: a for r, a in self.agents.items()},
            upstream=self.upstream,
            variant={k: self.extra.get(k) for k in PROMPT_VARIANT_KEYS if k in self.extra},
        )


def load() -> dict:
    cfg = {}
    for name in ("models", "methods", "pinned"):
        p = CONFIG / f"{name}.yaml"
        if not p.exists():
            raise ConfigError(f"missing config: {p}")
        cfg[name] = yaml.safe_load(open(p, encoding="utf-8"))
    p = CONFIG / "pricing.yaml"
    cfg["pricing"] = yaml.safe_load(open(p, encoding="utf-8")) if p.exists() else {}
    return cfg


def all_models(cfg) -> dict:
    """The tested models (local and API), without the pinned agents."""
    return {**cfg["models"].get("local", {}), **cfg["models"].get("api", {})}


def all_addressable(cfg) -> dict:
    """Everything get_client must resolve: the tested models plus the pinned agents."""
    return {**all_models(cfg), **cfg["models"].get("pinned_agents", {})}


def validate(cfg, strict=False) -> list[str]:
    """Return a list of problems. Empty means the config is runnable."""
    problems: list[str] = []
    methods, pinned = cfg["methods"], cfg["pinned"]
    models = all_models(cfg)
    addressable = all_addressable(cfg)

    served = cfg["models"].get("defaults", {}).get("served_context")
    if served is None:
        problems.append("models.yaml: defaults.served_context is not set")
    elif pinned.get("served_context") not in (None, served):
        # the run header uses pinned.yaml's value, the vLLM command models.yaml's
        problems.append(
            f"served_context disagrees: models.yaml defaults={served}, "
            f"pinned.yaml={pinned.get('served_context')} - the provenance header and the serve "
            f"command would claim different context caps")
    for name, m in cfg["models"].get("local", {}).items():
        on_api = m.get("route", "local") == "api"
        nc = m.get("native_context")
        if nc is None:
            problems.append(f"models.yaml/{name}: native_context not declared")
        elif served and nc < served and not on_api:
            # the uniform cap applies only to models served locally
            problems.append(
                f"models.yaml/{name}: native_context {nc} < served_context {served} "
                f"- this model cannot meet the uniform cap")
        if "reasoning" not in m:
            problems.append(f"models.yaml/{name}: reasoning setting not declared")
        if on_api:
            nested = m.get("api") or {}
            if not nested.get("provider"):
                problems.append(f"models.yaml/{name}: route is api but api.provider is missing")
            if strict and nested.get("id") in (None, "TBD"):
                problems.append(
                    f"models.yaml/{name}: route is api but api.id is TBD - needs a dated "
                    f"snapshot id")
    if strict:
        for name, m in models.items():
            if m.get("id") in (None, "TBD"):
                problems.append(f"models.yaml/{name}: id is TBD - needs a dated snapshot id")

    for role in ("persuadee", "judge"):
        if not pinned.get(role):
            problems.append(f"pinned.yaml: {role} not set")

    jury_members = (pinned.get("debate_jury") or {}).get("members") or []
    for role in ("persuadee", "judge"):
        alias = pinned.get(role)
        if alias and alias not in addressable:
            problems.append(
                f"pinned.yaml/{role}: '{alias}' has no entry in models.yaml - get_client would "
                f"fail on the first call of every run that uses it")
    for alias in jury_members:
        if alias not in addressable:
            problems.append(
                f"pinned.yaml/debate_jury: '{alias}' has no entry in models.yaml")
    jury = pinned.get("debate_jury") or {}
    if jury and not jury.get("members"):
        problems.append("pinned.yaml: debate_jury has no members")
    if strict and jury and (jury.get("combination") in (None, "TBD")):
        problems.append("pinned.yaml: debate_jury.combination is TBD")

    import collections
    for axis, want in GROUPING_AXES.items():
        missing = [n for n, sp in methods.items() if not (sp or {}).get(axis)]
        if missing:
            problems.append(
                f"methods.yaml: {len(missing)} method(s) have no '{axis}' - RQ2 groups on it, so "
                f"a missing value silently drops the method from that axis: {', '.join(missing)}")
            continue
        got = dict(collections.Counter(sp[axis] for sp in methods.values()))
        if got != want:
            diff = {k: (want.get(k, 0), got.get(k, 0))
                    for k in set(want) | set(got) if want.get(k, 0) != got.get(k, 0)}
            problems.append(
                f"methods.yaml: '{axis}' counts are {got}, expected {want}. Changed: "
                + ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(diff.items()))
                + ". Either an assignment changed or a method was added/removed; the RQ2 test "
                  "and every within/between pair count depend on this.")

    for name, spec in methods.items():
        todo = spec.get("status")
        for f in REQUIRED_METHOD_FIELDS:
            if f not in spec:
                problems.append(f"methods.yaml/{name}: missing required field '{f}'")
        # refusal_floor is the score a refusal gets; metric_range bounds the metric
        scale = spec.get("scale") or {}
        if "refusal_floor" not in scale:
            problems.append(
                f"methods.yaml/{name}: scale.refusal_floor not declared - the refusal policy "
                f"needs it")
        elif scale["refusal_floor"] == "TBD" and strict:
            problems.append(f"methods.yaml/{name}: scale.refusal_floor is TBD")
        if "metric_range" not in scale:
            problems.append(f"methods.yaml/{name}: scale.metric_range not declared (null is allowed "
                            f"for an unbounded metric)")
        elif scale["metric_range"] is not None and not (
                isinstance(scale["metric_range"], list) and len(scale["metric_range"]) == 2
                and all(isinstance(x, (int, float)) for x in scale["metric_range"])):
            problems.append(f"methods.yaml/{name}: scale.metric_range must be [lo, hi] or null")
        sev = spec.get("deviation_severity") or {}
        for d in (spec.get("deviation") or {}):
            if sev.get(d) not in ("green", "amber", "red"):
                problems.append(f"methods.yaml/{name}: deviation '{d}' has no severity "
                                f"(green | amber | red) in deviation_severity")
        if spec.get("sampling_unit") == "TBD" and strict:
            problems.append(f"methods.yaml/{name}: sampling_unit is TBD - split-half needs it")
        if strict and spec.get("n") in (None, "TBD"):
            problems.append(f"methods.yaml/{name}: n is TBD")
        if strict and spec.get("headline_metric") in (None, "TBD"):
            problems.append(f"methods.yaml/{name}: headline_metric is TBD")

        for role, target in (spec.get("roles") or {}).items():
            if target not in PINNED_ROLES:
                problems.append(
                    f"methods.yaml/{name}: role '{role}' maps to '{target}', which is not a "
                    f"pinned role. Fixed agents come from pinned.yaml, never from a method.")

        for d, why in (spec.get("deviation") or {}).items():
            if not str(why).strip():
                problems.append(f"methods.yaml/{name}: deviation '{d}' has no stated reason")

        if strict and not todo and not spec.get("upstream", {}).get("commit") \
                and "scorer" not in (spec.get("upstream") or {}):
            problems.append(f"methods.yaml/{name}: upstream.commit not pinned")

    return problems


def check_clones(cfg) -> list[str]:
    """Report clones that are missing, at the wrong commit, or locally edited."""
    problems = []
    for name, spec in cfg["methods"].items():
        up = spec.get("upstream") or {}
        declared = up.get("repo_path")
        if declared == "none":
            continue                     # no clone needed (e.g. a HuggingFace scorer)
        repo = ROOT / declared if declared else ROOT / "methods" / name / "repo"
        want = up.get("commit")
        if not repo.exists():
            if not spec.get("status"):
                problems.append(f"{name}: clone missing at {repo.relative_to(ROOT)}")
            continue
        try:
            head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
            # only the paths under `uses` must be clean
            paths = up.get("uses") or []
            cmd = ["git", "-C", str(repo), "status", "--porcelain"] + (["--"] + paths if paths else [])
            dirty = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:
            problems.append(f"{name}: cannot read clone state ({type(e).__name__})")
            continue
        if dirty:
            scope = f" under {', '.join(up.get('uses'))}" if up.get("uses") else ""
            problems.append(f"{name}: clone has local edits ({len(dirty.splitlines())} files)"
                            f"{scope} - upstream must never be modified")
        if want and want != "TBD" and not head.startswith(str(want)):
            problems.append(f"{name}: clone is at {head[:10]}, config pins {want}")
    return problems


def resolve(cfg, method: str, model: str) -> Spec:
    """Build the Spec for one run. Raises if the pair is not runnable."""
    methods, pinned = cfg["methods"], cfg["pinned"]
    models = all_models(cfg)
    addressable = all_addressable(cfg)
    if method not in methods:
        raise ConfigError(f"unknown method '{method}'")
    if model not in models:
        raise ConfigError(f"unknown model '{model}'")
    spec, m = methods[method], models[model]

    agents = {}
    for role, target in (spec.get("roles") or {}).items():
        if target == "debate_jury":
            agents[role] = (pinned.get("debate_jury") or {}).get("members")
        else:
            agents[role] = pinned.get(target)
        if not agents[role]:
            raise ConfigError(f"{method}: role '{role}' -> '{target}' is not set in pinned.yaml")

    return Spec(
        method=method, model=model, model_id=m.get("id"),
        n=spec.get("n"), sampling_unit=spec.get("sampling_unit"),
        headline_metric=spec.get("headline_metric"), scale=spec.get("scale") or {},
        temperature=spec.get("temperature"), turns=spec.get("turns"),
        seed=pinned.get("seed"), served_context=pinned.get("served_context"),
        agents=agents, upstream=spec.get("upstream") or {},
        extra={k: v for k, v in spec.items()
               if k not in ("n", "sampling_unit", "headline_metric", "scale",
                            "temperature", "turns", "roles", "upstream")},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on TBDs - use this before collection starts")
    args = ap.parse_args()

    cfg = load()
    problems = validate(cfg, strict=args.strict) + check_clones(cfg)

    methods, models = cfg["methods"], all_models(cfg)
    ready = [m for m, s in methods.items() if not s.get("status")]
    print(f"\n{len(methods)} methods ({len(ready)} adapted), {len(models)} models "
          f"-> {len(methods) * len(models)} runs\n")

    hdr = f"{'method':22s} {'n':>6s} {'unit':17s} {'turns':>5s} {'temp':>6s}  agents"
    print(hdr); print("-" * (len(hdr) + 18))
    for name in methods:
        try:
            sp = resolve(cfg, name, next(iter(models)))
        except ConfigError as e:
            print(f"  {name:22s} UNRESOLVABLE: {e}")
            continue
        t = sp.temperature
        t = f"{t:g}" if isinstance(t, (int, float)) else ("multi" if isinstance(t, dict) else "-")
        ag = ", ".join(f"{r}={a}" for r, a in sp.agents.items()) or "none"
        print(f"  {name:22s} {str(sp.n):>6s} {str(sp.sampling_unit):17s} "
              f"{str(sp.turns or '-'):>5s} {t:>6s}  {ag}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nconfig OK")


if __name__ == "__main__":
    main()
