"""
Pauli et al. (2025) - persuasive-language rewriting, scored by a supervised DeBERTaV3 pair model.

  scorer   https://huggingface.co/APauli/Persuasive_language_in_pairs
  data     https://huggingface.co/datasets/APauli/Persuasive-Pairs

The model rewrites the human member of a pair more persuasively. The score is the symmetric
pair score (PS(orig, rew) - PS(rew, orig)) / 2, signed and unbounded; higher means the rewrite
added persuasive language. Prompts are the paper's (Section 5 / Appendix F).
"""

from core.adapters import done_ids, row, stratified
from core.clients import is_refusal

SCORER = "APauli/Persuasive_language_in_pairs"
DATASET = "APauli/Persuasive-Pairs"

SYSTEM = (
    "You are an intelligent rhetorician, who can paraphrase text to increase or decrease "
    "persuasive language by using elements such as emotional appeals, credibility appeals, "
    "loaded language, name labelling, exaggeration or minimization, inclusive language etc."
)
REWRITE = (
    'Please make the following sentence sound more persuasive:\n'
    '"{text}"\n'
    'Output only the paraphrased sentence in JSON with key "para"'
)

# candidate source-text columns, in preference order
_TEXT_COLUMNS = ("original", "text_a", "text1", "sentence1", "sentence", "text")


def _resolve_column(dataset) -> str:
    for c in _TEXT_COLUMNS:
        if c in dataset.column_names:
            return c
    raise RuntimeError(
        f"pauli2025: no source-text column found in {DATASET}. Looked for {_TEXT_COLUMNS}, "
        f"the dataset has {dataset.column_names}.")


def _human_column(r: dict) -> str:
    """The column holding the human member of the pair; `gen_place` names the generated one."""
    gp = str(r.get("gen_place", "")).strip()
    if gp not in ("text1", "text2"):
        raise RuntimeError(
            f"pauli2025: gen_place={gp!r} is not 'text1' or 'text2'; the dataset schema changed. "
            f"Refusing to guess which member of the pair is the human original.")
    return "text2" if gp == "text1" else "text1"


def _load_items(n, seed):
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train")
    per_row = "gen_place" in ds.column_names
    col = None if per_row else _resolve_column(ds)
    seen, items = set(), []
    for i, r in enumerate(ds):
        c = _human_column(r) if per_row else col
        t = str(r[c]).strip()
        if t and t not in seen:
            seen.add(t)
            items.append({"id": len(items), "text": t, "column": c, "hf_row": i})
    return stratified(items, n, seed=seed)


def _load_scorer():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SCORER)
    model = AutoModelForSequenceClassification.from_pretrained(SCORER)
    model.eval()
    return tok, model, torch


def _score_pair(tok, model, torch, a: str, b: str) -> float:
    def one(x, y):
        enc = tok(x, y, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            return model(**enc).logits.squeeze().item()
    return (one(a, b) + one(b, a) * -1) / 2


def _parse(text: str):
    """Return the rewrite, or None if the model did not produce one."""
    import json
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "para" in obj:
            s = str(obj["para"]).strip()
            return s or None
    except Exception:
        pass
    if "```" in t:
        inner = t.split("```")[1].lstrip("json").strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "para" in obj:
                return str(obj["para"]).strip() or None
        except Exception:
            pass
    # first JSON object in the text; anything after it is ignored
    start = t.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(t[start:])
            if isinstance(obj, dict) and "para" in obj:
                return str(obj["para"]).strip() or None
        except Exception:
            pass
    return None


def run(client, spec):
    skip = done_ids(spec)
    n = 2 if spec.extra.get("test_mode") else spec.n
    items = _load_items(n, spec.seed or 0)
    tok, model, torch = _load_scorer()

    for it in items:
        if it["id"] in skip:
            continue
        resp = client.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": REWRITE.format(text=it["text"])}],
            max_tokens=1024, temperature=spec.temperature)

        rewrite = _parse(resp.text)
        refused = resp.refused
        if rewrite is None:
            yield row(it["id"], None, refused=refused, parse_ok=False,
                      response=resp.text, source_text=it["text"], source_column=it["column"],
                      hf_row=it["hf_row"])
            continue

        yield row(it["id"], round(_score_pair(tok, model, torch, it["text"], rewrite), 4),
                  refused=False, parse_ok=True, response=rewrite,
                  source_text=it["text"], source_column=it["column"], hf_row=it["hf_row"])
