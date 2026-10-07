"""
Canary: check that a local model serves the 32,768-token context and measure its throughput.
Sends prompts of increasing size, up to the MakeMeSay peak and the full cap, to an already-serving
vLLM on localhost. Writes data/derived/canary_result.json.

    conda activate <your vllm env>
    python scripts/canary.py --model mistral-medium-3.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry                                    # noqa: E402

# 1,143 system + 30 turns x (300 manipulator + 200 manipulatee)
MAKEMESAY_PEAK = 16_150


SENTENCE = ("The committee reviewed the proposal carefully and considered every argument "
            "presented by both sides before reaching its provisional conclusion. ")


def filler(n_tokens: int, tokenizer=None) -> str:
    """Prose of the requested token length, sized with the model's tokenizer when available."""
    if tokenizer is not None:
        per_rep = len(tokenizer.encode(SENTENCE, add_special_tokens=False))
        reps = max(1, n_tokens // max(1, per_rep))
        text = SENTENCE * reps
        while len(tokenizer.encode(text, add_special_tokens=False)) > n_tokens and reps > 1:
            reps -= max(1, reps // 20)
            text = SENTENCE * reps
        return text
    # fallback: ~4 characters per token for English prose
    return SENTENCE * (max(1, (n_tokens * 4) // len(SENTENCE)) + 1)


def get_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True,
                                             trust_remote_code=True)
    except Exception as e:
        print(f"  (no local tokenizer for sizing: {type(e).__name__}; "
              f"falling back to a character heuristic)")
        return None


def probe(client, model_id: str, n_tokens: int, max_out: int = 64, tokenizer=None) -> dict:
    prompt = ("Below is a transcript. Summarise the single main argument in one sentence.\n\n"
              + filler(n_tokens, tokenizer) + "\n\nOne sentence:")
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_out, temperature=0.0)
    except Exception as e:
        return dict(ok=False, requested=n_tokens, error=f"{type(e).__name__}: {e}")
    dt = time.time() - t0
    u = r.usage
    out_tok = getattr(u, "completion_tokens", 0) or 0
    return dict(ok=True, requested=n_tokens,
                prompt_tokens=getattr(u, "prompt_tokens", 0),
                completion_tokens=out_tok,
                seconds=round(dt, 2),
                tokens_per_second=round(out_tok / dt, 1) if dt > 0 else None,
                text=(r.choices[0].message.content or "")[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral-medium-3.5",
                    help="alias from config/models.yaml")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--sizes", default=f"1000,{MAKEMESAY_PEAK},32000")
    args = ap.parse_args()

    cfg = registry.load()
    local = cfg["models"].get("local", {})
    if args.model not in local:
        sys.exit(f"{args.model} is not a local model in config/models.yaml")
    m = local[args.model]
    cap = cfg["models"]["defaults"]["served_context"]

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    print(f"\nCANARY  {args.model}  ({m['id']})")
    print(f"  declared cap {cap:,}   native window {m.get('native_context'):,}")
    print(f"  study worst case (makemesay @30 turns) ~{MAKEMESAY_PEAK:,} tokens\n")

    try:
        served = [x.id for x in client.models.list().data]
        print(f"  server is offering: {served}")
        if m["id"] not in served:
            print(f"  WARNING the server is not offering {m['id']} - is the right model loaded?")
    except Exception as e:
        sys.exit(f"  cannot reach vLLM at {args.base_url}: {type(e).__name__}: {e}\n"
                 f"  Start it with:  python -m core.schedule --model {args.model}")

    tokenizer = get_tokenizer(m["id"])
    results = []
    for size in [int(x) for x in args.sizes.split(",")]:
        label = ("study worst case" if size == MAKEMESAY_PEAK
                 else "full cap" if size >= cap - 1000 else "warm-up")
        print(f"\n  --- {size:,} tokens ({label}) ---")
        r = probe(client, m["id"], size, tokenizer=tokenizer)
        results.append(dict(r, label=label))
        if r["ok"]:
            print(f"      OK   prompt={r['prompt_tokens']:,} out={r['completion_tokens']} "
                  f"in {r['seconds']}s  -> {r['tokens_per_second']} tok/s")
        else:
            print(f"      FAIL {r['error'][:200]}")
            if size <= MAKEMESAY_PEAK:
                print(f"\n  *** This is below the study's worst case. The uniform {cap:,} cap is "
                      f"NOT achievable for this model. The other option is needed: a lower cap for "
                      f"everyone, or dropping this pairing. ***")
            break

    ok_big = [r for r in results if r["ok"] and r["requested"] >= MAKEMESAY_PEAK]
    print("\n" + "=" * 66)
    if ok_big:
        tps = [r["tokens_per_second"] for r in ok_big if r["tokens_per_second"]]
        print(f"  CONTEXT CAP SETTLED: {args.model} serves the study's worst case at {cap:,} context.")
        print(f"  Every smaller model has more headroom, so the uniform cap holds.")
        if tps:
            print(f"  Measured throughput: {min(tps)}-{max(tps)} tok/s at long context.")
            print(f"  -> put this in config/pricing.yaml under "
                  f"local.tokens_per_second_by_size['{m.get('params')}']")
    else:
        print(f"  CONTEXT CAP NOT settled - see the failure above.")

    out = Path(__file__).resolve().parent.parent / "data" / "derived" / "canary_result.json"
    out.write_text(json.dumps(dict(model=args.model, model_id=m["id"], cap=cap,
                                   results=results), indent=1), encoding="utf-8")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
