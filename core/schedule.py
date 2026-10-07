"""
Orders the 135 runs model-first: load one local model, run every method against it, unload.

    python -m core.schedule                 # print the plan
    python -m core.schedule --execute       # run it (local models need vLLM already serving)
    python -m core.schedule --only-api      # the 5 API models; no GPU, runs anytime
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from core import registry, runner

ROOT = Path(__file__).resolve().parent.parent


def plan(cfg, only_api=False, only_local=False, methods=None, models=None):
    """Return [(model, kind, [methods])] in execution order."""
    declared = cfg["models"].get("local", {})
    api = dict(cfg["models"].get("api", {}))
    # a local entry with `route: api` schedules with the API models
    local = {k: v for k, v in declared.items() if v.get("route", "local") != "api"}
    api.update({k: v for k, v in declared.items() if v.get("route", "local") == "api"})
    ready = [m for m, s in cfg["methods"].items() if not s.get("status")]
    if methods:
        ready = [m for m in ready if m in methods]

    order = []
    # API models first
    if not only_local:
        for name in api:
            if not models or name in models:
                order.append((name, "api", list(ready)))
    if not only_api:
        # local models smallest first
        def size(m):
            p = str(local[m].get("params", "0")).upper().replace("B", "").split()[0]
            try:
                return float(p)
            except ValueError:
                return 1e9
        for name in sorted(local, key=size):
            if not models or name in models:
                order.append((name, "local", list(ready)))
    return order


def serve_command(cfg, model: str) -> str:
    """The vLLM command for one local model, at the uniform context cap."""
    m = cfg["models"]["local"][model]
    ctx = cfg["models"]["defaults"]["served_context"]
    # env vllm4: vLLM 0.10.1.1 with transformers 4.57.6; HF_HUB_OFFLINE=1 for a read-only cache
    parts = [
        "conda activate vllm4 &&",
        "HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}",
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {m['id']}",
        f"--served-model-name {m['id']}",
        f"--max-model-len {ctx}",
        "--port ${PORT:-8000}",
    ]
    if m.get("quantization") == "fp8":
        parts.append("--quantization fp8")
    if "exaone" in model or "gemma-4" in model:
        parts.append("--trust-remote-code")
    # the biggest models need most of the card just for weights
    p = str(m.get("params", "")).upper()
    if any(p.startswith(x) for x in ("70", "120", "128")):
        parts.append("--gpu-memory-utilization 0.95")
    return " \\\n    ".join(parts)


def gpu_hours(cfg, model: str, methods: list[str]) -> float | None:
    """Rough estimate from declared throughput. Returns None if unknown."""
    pricing = cfg.get("pricing") or {}
    tps_by_size = (pricing.get("local") or {}).get("tokens_per_second_by_size") or {}
    m = cfg["models"]["local"].get(model)
    if not m:
        return None
    tps = tps_by_size.get(str(m.get("params", "")).strip())
    if not tps:
        return None
    # output tokens only; input is prefill and much faster
    out = 0
    for name in methods:
        spec = cfg["methods"][name]
        n = spec.get("n")
        if not isinstance(n, int):
            continue
        turns = spec.get("turns") or 1
        out += n * turns * 300          # ~300 output tokens per tested-model turn
    return out / tps / 3600 if out else None


def already_done(method: str, model: str) -> int:
    _, done = runner.read_done(runner.out_path(method, model))
    return len(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only-api", action="store_true")
    ap.add_argument("--only-local", action="store_true")
    ap.add_argument("--method", action="append")
    ap.add_argument("--model", action="append")
    args = ap.parse_args()

    cfg = registry.load()
    problems = registry.validate(cfg) + registry.check_clones(cfg)
    if problems:
        print("[abort] config is invalid:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    order = plan(cfg, args.only_api, args.only_local, args.method, args.model)
    total_runs = sum(len(ms) for _, ms in ((m, ms) for m, _, ms in order))
    print(f"\n{len(order)} models x methods = {total_runs} runs, model-first order\n")

    total_gpu = 0.0
    for model, kind, ms in order:
        done = sum(1 for x in ms if already_done(x, model))
        gh = gpu_hours(cfg, model, ms) if kind == "local" else None
        if gh:
            total_gpu += gh
        tag = f"{gh:5.1f} GPU-h" if gh else ("    API   " if kind == "api" else "     ?    ")
        print(f"  {model:22s} {kind:5s} {tag}  {len(ms)} methods"
              f"{f'  ({done} already complete)' if done else ''}")

    if total_gpu:
        ready = [m for m, s in cfg["methods"].items() if not s.get("status")]
        priced = [m for m in ready if isinstance(cfg["methods"][m].get("n"), int)]
        unpriced = [m for m in ready if m not in priced]
        print(f"\nEstimated GPU time: {total_gpu:.0f} hours "
              f"({total_gpu / 24:.1f} days of exclusive access)")
        if unpriced:
            print(f"  *** FLOOR, NOT AN ESTIMATE *** it covers {len(priced)} of {len(ready)} "
                  f"adapted methods.")
            print(f"  No n yet, so contributing zero: {', '.join(unpriced)}")
        print("  Throughput figures in config/pricing.yaml are unmeasured; a first local run should "
              "record real numbers for the 128B model.")

    if not args.execute:
        print("\nThis is a plan. Add --execute to run it.")
        print("Local models need vLLM already serving; see `serve` below.\n")
        for model, kind, _ in order:
            if kind == "local":
                print(f"# --- {model} ---\n{serve_command(cfg, model)}\n")
        return

    for model, kind, ms in order:
        print(f"\n{'=' * 70}\n{model} ({kind})\n{'=' * 70}")
        if kind == "local":
            print(f"  Ensure vLLM is serving {model} before continuing.")
        for method in ms:
            t0 = time.time()
            print(f"\n--- {method} / {model} ---")
            try:
                runner.run_one(method, model, cfg)
            except Exception as e:
                # one failed run does not stop the sweep
                print(f"  [skip] {method}/{model}: {type(e).__name__}: {e}")
            print(f"  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
