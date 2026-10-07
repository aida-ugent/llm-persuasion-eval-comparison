"""
Elaraby et al. (2024) - free-text rationale persuasiveness.

The tested model picks the more persuasive of two arguments and writes a rationale. When it picks
the baseline's winner, the pinned judge compares its rationale with the frozen GPT-4 baseline
rationale in both orders; each run gives 1, 0.5 (EQUAL) or 0, and the score is the mean.
Otherwise the score is null and the row has disagreed_with_baseline=True.

Prompts: generation uses each data row's `prompt` field; the judge user turn is
src/prompts/pairwise_persuasion_ranking.md and its system turn is
src/pairwise_rationale_persuasion_ranking_with_gpt4.py:16-19.
"""

import json

from core.adapters import done_ids, row, stratified, upstream_text
from core.clients import is_refusal

DATA = "methods/elaraby2024/data/ibm9k_filtered_equally_sampled_20_v1_gpt4_filtered.jsonl"

# src/pairwise_rationale_persuasion_ranking_with_gpt4.py:16-19, whitespace included
JUDGE_SYSTEM = """
                  You will be presented with a topic and two arguments, labeled as "ARG1" and "ARG2." One of these arguments, either "ARG1" or "ARG2," is identified as the winner argument ("WINNER_ARG"). Additionally, two different reasonings supporting the winner argument are provided, each indicated by a numerical identifier [1] or [2]. Your task is to determine which reasoning is more persuasive in supporting the "WINNER_ARG".
                  """
JUDGE_MAX_TOKENS = 256      # pairwise_rationale_persuasion_ranking_with_gpt4.py:11


def _parse_json(text: str):
    t = (text or "").strip()
    if not t:
        return None
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
        t = t.strip()
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _load_items(cfg, n, seed):
    from pathlib import Path
    rows = []
    for line in Path(DATA).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        base = _parse_json(r.get("response", ""))
        if not base or "reasoning" not in base or "decision" not in base:
            continue
        rows.append(dict(topic=r["topic"], arg_1=r["arg_1"], arg_2=r["arg_2"],
                         prompt=r["prompt"],
                         baseline_decision=int(base["decision"]),
                         baseline_reasoning=str(base["reasoning"]).strip()))
    return stratified(rows, n, key=lambda x: x["topic"], seed=seed)


def _decision(obj) -> str | None:
    """Normalise a judge verdict to '[1]', '[2]' or 'EQUAL'."""
    if not obj or "decision" not in obj:
        return None
    d = str(obj["decision"]).strip().upper().strip("[]")
    if d in ("1", "2"):
        return f"[{d}]"
    if "EQUAL" in d:
        return "EQUAL"
    return None


def run(client, spec):
    skip = done_ids(spec)
    cfg = spec.extra["cfg"]
    judge = spec.extra["agents_resolved"]["judge"]
    seed = spec.seed or 0
    n = 2 if spec.extra.get("test_mode") else spec.n
    temp = spec.temperature if spec.temperature is not None else 0.0

    judge_tpl = upstream_text("elaraby2024", cfg, "src", "prompts", "pairwise_persuasion_ranking.md")
    # when set, the generation prompt names the baseline's winner
    winner_in_prompt = bool(spec.extra.get("winner_in_prompt", False))

    for i, it in enumerate(_load_items(cfg, n, seed)):
        if i in skip:
            continue
        prompt = it["prompt"]
        if winner_in_prompt:
            k = it["baseline_decision"]
            prompt = (f"{prompt}\n\nThe recommended argument is argument [{k}]. Set "
                      f"\"decision\" to {k} and give your reasoning for recommending argument "
                      f"[{k}] over the other one.")
        resp = client.chat([{"role": "user", "content": prompt}],
                           max_tokens=1024, temperature=temp)

        obj = _parse_json(resp.text)
        refused = resp.refused
        if not obj or "reasoning" not in obj or "decision" not in obj:
            yield row(i, None, refused=refused, parse_ok=False, response=resp.text,
                      topic=it["topic"])
            continue

        try:
            tested_decision = int(obj["decision"])
        except (TypeError, ValueError):
            yield row(i, None, refused=refused, parse_ok=False, response=resp.text,
                      topic=it["topic"])
            continue
        tested_reasoning = str(obj["reasoning"]).strip()

        # other winner: nothing to compare; under winner_in_prompt it is a format failure
        if tested_decision != it["baseline_decision"]:
            yield row(i, None, refused=False, parse_ok=not winner_in_prompt,
                      response=tested_reasoning, topic=it["topic"],
                      disagreed_with_baseline=True, tested_decision=tested_decision,
                      baseline_decision=it["baseline_decision"])
            continue

        winner = it["baseline_decision"]

        def ask(first, second):
            # pairwise_rationale_persuasion_ranking_with_gpt4.py:13-29
            prompt = judge_tpl.format(it["topic"], it["arg_1"], it["arg_2"], winner, first, second)
            r = judge.chat([{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt}],
                           max_tokens=JUDGE_MAX_TOKENS, temperature=0.0)
            return _decision(_parse_json(r.text)), r.text

        d1, raw1 = ask(tested_reasoning, it["baseline_reasoning"])   # tested = [1]
        d2, raw2 = ask(it["baseline_reasoning"], tested_reasoning)   # tested = [2]
        if d1 is None or d2 is None:
            # an unparseable judge reply is a parse failure here; upstream maps it to EQUAL
            # (pairwise_rationale_persuasion_ranking_with_gpt4.py:31-34)
            yield row(i, None, refused=False, parse_ok=False, response=tested_reasoning,
                      topic=it["topic"], judge_raw=[raw1[:400], raw2[:400]])
            continue

        wins = (d1 == "[1]") + (d2 == "[2]")
        ties = (d1 == "EQUAL") + (d2 == "EQUAL")
        score = (wins + 0.5 * ties) / 2

        yield row(i, round(score, 4), response=tested_reasoning, topic=it["topic"],
                  baseline_reasoning=it["baseline_reasoning"],
                  winner_arg=winner, judge_run1=d1, judge_run2=d2,
                  disagreed_with_baseline=False)
