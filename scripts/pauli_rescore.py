"""pauli2025 re-parse and re-score: parse each unparseable reply by its first JSON object and score
any rewrite found with the frozen scorer (needs torch and transformers). Writes a
<name>.pre_reparse.bak backup beside each rewritten file.

    python scripts/pauli_rescore.py            # report only
    python scripts/pauli_rescore.py --apply    # rewrite rows in place (backups kept)
"""
import argparse, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry  # noqa: E402
from methods.pauli2025 import adapter as A  # noqa: E402

D = registry.ROOT / "data" / "results" / "pauli2025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cfg = registry.load()
    scorer = None
    total = 0
    for model in registry.all_models(cfg):
        for tag in ("", "rep2"):
            p = D / (f"{model}__{tag}.jsonl" if tag else f"{model}.jsonl")
            if not p.exists():
                continue
            lines = [l for l in open(p, encoding="utf-8") if l.strip()]
            out, fixed = [], 0
            for l in lines:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r.get("item_id") is not None and r.get("parse_ok") is False and not r.get("refused"):
                    rewrite = A._parse(r.get("response") or "")
                    if rewrite:
                        fixed += 1
                        if a.apply:
                            if scorer is None:
                                scorer = A._load_scorer()
                            tok, mdl, torch = scorer
                            r.update(score=round(A._score_pair(tok, mdl, torch, r["source_text"], rewrite), 4),
                                     parse_ok=True, response=rewrite, reparsed_first_object=True)
                out.append(json.dumps(r) + "\n")
            total += fixed
            if fixed:
                print(f"  {p.name:36s} rescued {fixed}")
                if a.apply:
                    shutil.copy2(p, p.with_suffix(".jsonl.pre_reparse.bak"))
                    open(p, "w", encoding="utf-8").writelines(out)
    print(f"\n{'APPLIED' if a.apply else 'dry run'}: {total} row(s) rescued by the first-object parse.")


if __name__ == "__main__":
    main()
