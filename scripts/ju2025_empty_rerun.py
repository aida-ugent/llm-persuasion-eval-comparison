"""ju2025: for each row whose stored response is empty (not refused), rerun the tested model's
evidence call (the client escalates the cap on an empty reply), truncate the visible paragraph to
512 o200k_base tokens, and send it to the pinned listener as the adapter does. A reply still empty
at the ceiling stays a parse failure.

    python scripts/ju2025_empty_rerun.py --dry-run                 # counts only, no calls
    python scripts/ju2025_empty_rerun.py --model kimi-k2.6         # -> <model>__emptyrerun.jsonl
    python scripts/ju2025_empty_rerun.py --model kimi-k2.6 --apply # main <- rerun; the original
                                                                   # pass kept as <model>__cap512.jsonl
Both the main file and the rep2 replicate are processed.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import clients, registry           # noqa: E402
from core.adapters import row                # noqa: E402
from methods.ju2025 import adapter as A      # noqa: E402

M = "ju2025"
D_RES = registry.ROOT / "data" / "results" / M
ORIGINAL_CAP = 512            # the adapter's persuader cap; also the visible-length limit
ENCODING = "o200k_base"       # gpt-4.1-mini (the pinned listener)
TAG_OUT, TAG_KEEP = "emptyrerun", "cap512"


def is_empty_row(r) -> bool:
    return (r.get("score") is None and not r.get("refused") and r.get("parse_ok") is False
            and not (r.get("response") or "").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cfg = registry.load()
    models = [a.model] if a.model else list(registry.all_models(cfg))

    enc = None
    if not a.dry_run:
        import tiktoken
        enc = tiktoken.get_encoding(ENCODING)

    for model in models:
        spec = registry.resolve(cfg, M, model)
        tested = listener = None
        for tag in ("", "rep2"):
            src = D_RES / (f"{model}__{tag}.jsonl" if tag else f"{model}.jsonl")
            if not src.exists():
                continue
            lines = [l for l in open(src, encoding="utf-8") if l.strip()]
            rows = []
            for l in lines[1:]:
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError:
                    continue
            empties = [r for r in rows if is_empty_row(r)]
            if not empties:
                continue
            print(f"{src.name:28s} {len(empties):4d} empty rows of {len(rows)}"
                  + (" (dry run)" if a.dry_run else ""))
            if a.dry_run:
                continue
            if tested is None:
                tested = clients.get_client(model, cfg)
                listener = clients.get_client(spec.agents["listener"], cfg, is_pinned=True)

            dst = D_RES / (f"{model}__{tag + '_' if tag else ''}{TAG_OUT}.jsonl")
            done = set()
            if dst.exists():
                done = {json.loads(l)["item_id"] for l in open(dst, encoding="utf-8") if '"item_id"' in l}
            out = open(dst, "a", encoding="utf-8")
            if not done:
                meta = json.loads(lines[0])["__meta__"]
                out.write(json.dumps({"__meta__": {**meta, "repaired":
                          f"empty evidence rows rerun with the escalate-on-empty rule, visible text "
                          f"truncated to {ORIGINAL_CAP} {ENCODING} tokens before the listener"}}) + "\n")
            stats = dict(rerun=0, still_empty=0, refused=0, truncated=0, scored=0)
            for r in rows:
                if r["item_id"] in done:
                    continue
                if not is_empty_row(r):
                    out.write(json.dumps(r) + "\n"); continue
                strategy, it_prompt, target = r["strategy"], r["question"], r["target"]
                sys_p = A.PERSUADER_SYS.format(strategy=strategy, desc=A.STRATEGIES[strategy])
                usr_p = A.PERSUADER_USER.format(prompt=it_prompt, target=target)
                ev = tested.chat([{"role": "system", "content": sys_p},
                                  {"role": "user", "content": usr_p}],
                                 max_tokens=ORIGINAL_CAP, temperature=spec.temperature)
                stats["rerun"] += 1
                extra = dict(strategy=strategy, item=r.get("item"), question=it_prompt, target=target,
                             rerun_from_empty=True, rerun_budget=ev.token_budget,
                             rerun_completion_tokens=ev.completion_tokens,
                             rerun_finish_reason=ev.finish_reason)
                if ev.refused or not ev.text.strip():
                    stats["refused" if ev.refused else "still_empty"] += 1
                    out.write(json.dumps(row(r["item_id"], None, refused=ev.refused,
                                             parse_ok=bool(ev.text.strip()), response=ev.text,
                                             **extra)) + "\n")
                    continue
                ids = enc.encode(ev.text.strip())
                text = enc.decode(ids[:ORIGINAL_CAP]) if len(ids) > ORIGINAL_CAP else ev.text.strip()
                extra.update(rerun_visible_tokens=len(ids), rerun_truncated=len(ids) > ORIGINAL_CAP)
                stats["truncated"] += extra["rerun_truncated"]
                ans = listener.chat(
                    [{"role": "system", "content": A.LISTENER_SYS},
                     {"role": "user", "content": A.LISTENER_USER.format(evidence=text, question=it_prompt)}],
                    max_tokens=8, temperature=0.0, respect_cap=True)
                if not ans.text.strip():
                    out.write(json.dumps(row(r["item_id"], None, refused=False, parse_ok=False,
                                             response=text, listener_answer=ans.text, **extra)) + "\n")
                    continue
                success = target in ans.text                       # upstream's case-sensitive substring test
                stats["scored"] += 1
                out.write(json.dumps(row(r["item_id"], float(success), response=text,
                                         listener_answer=ans.text.strip(), **extra)) + "\n")
            out.close()
            print(f"    {stats}")
            if a.apply:
                keep = D_RES / (f"{model}__{tag + '_' if tag else ''}{TAG_KEEP}.jsonl")
                src.rename(keep); dst.rename(src)
                print(f"    -> {src.name} is now the repaired file; original kept as {keep.name}")


if __name__ == "__main__":
    main()
