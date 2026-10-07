"""
Generate the deviation table from config/methods.yaml and config/models.yaml.

    python scripts/deviation_table.py          # writes DEVIATIONS.md and prints it
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import registry   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
def main():
    cfg = registry.load()
    rows = []
    for m, spec in cfg["methods"].items():
        for k, text in (spec.get("deviation") or {}).items():
            rows.append((m, k, " ".join(str(text).split())))
    pinned = cfg["pinned"]
    out = [f"# Deviations from the original methods, generated {date.today()} from config/methods.yaml",
           "",
           "Do not edit by hand: `python scripts/deviation_table.py` regenerates it. Pinned agents: persuadee "
           f"`{pinned.get('persuadee')}`, judge `{pinned.get('judge')}`; pinned temperature "
           f"{pinned.get('pinned_temperature')}; seed {pinned.get('seed')}.",
           "",
           "| Method | Deviation | What |", "| --- | --- | --- |"]
    for m, k, text in rows:
        out.append(f"| {m} | {k} | {text} |")
    dest = ROOT / "DEVIATIONS.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:6]))
    print(f"... {len(rows)} rows -> {dest}")


if __name__ == "__main__":
    main()
